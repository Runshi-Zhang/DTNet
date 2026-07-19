import os
import sys
import random

import torch
import numpy as np
import datetime
import logging
import provider
import argparse
from pathlib import Path

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Dataset
from models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from model_v3 import PointTransformerV3
from earth_movers_distance.emd import EarthMoverDistance
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()

def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)
def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2
EMD = EarthMoverDistance()
def emd(pcs1, pcs2,pair_mask):
    dists = EMD(pcs1, pcs2)
    return torch.mean(dists[pair_mask])

def mse(pred, target):
    return torch.mean(torch.sqrt(torch.sum((pred - target) ** 2, dim=-1)))
def absloss(pred, target, weight=None):
    if(weight == None):
        return torch.mean(torch.abs(pred - target))
    else:
        loss = torch.abs(pred - target)
        return loss

from scipy.spatial.transform import Rotation



class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, A, B):
        """
        A: [B, N, C] (query)
        B: [B, M, C] (key/value)
        """
        Q = self.to_q(A)
        K = self.to_k(B)
        V = self.to_v(B)

        attn = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = self.proj(out)

        return out

class ToothAdjacentRelationModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_attn = CrossAttention(dim)

    def forward(self, feats):
        """
        feats: [B, T, N, C]
        """
        B, T, N, C = feats.shape
        out = feats.clone()
        T1 = T // 2

        for t in range(T1):
            if t < T1 - 1:
                out[:, t] += self.cross_attn(
                    feats[:, t], feats[:, t+1]
                )
                out[:, t + T1] += self.cross_attn(
                    feats[:, t + T1], feats[:, t + 1 + T1]
                )
            if t > 0:
                out[:, t] += self.cross_attn(
                    feats[:, t], feats[:, t-1]
                )
                out[:, t + T1] += self.cross_attn(
                    feats[:, t + T1], feats[:, t - 1 + T1]
                )

        return out

class ToothTopLowRelationModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_attn = CrossAttention(dim)

    def forward(self, feats):
        """
        feats: [B, T, N, C]
        """
        B, T, N, C = feats.shape
        out = feats.clone()
        T1 = T // 2

        for t in range(T1):
            out[:, t] += self.cross_attn(
                feats[:, t], feats[:, T - t - 1]
            )
            out[:, T - t - 1] += self.cross_attn(
                feats[:, T - t - 1], feats[:, t]
            )
        return out

def knn_gnn(x, k: int):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)

    outputs:
    - idx: int (neighbor_idx)
    """
    # x : (batch_size, feature_dim, num_points)
    # Retrieve nearest neighbor indices
    B, _, N = x.size()
    n = N
    dev = x.device
    #x = x.to(torch.float16)
    if not torch.cuda.is_available():
        from knn_cuda import KNN

        ref = x.transpose(2, 1).contiguous()  # (batch_size, num_points, feature_dim)
        query = ref
        _, sid = KNN(k=k, transpose_mode=True)(ref, query)
        idx = sid.clone()
    else:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        sid = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        idx = sid.clone()
    sid += torch.arange(B, device=dev).view(B, 1, 1) * N
    sid = sid.reshape(-1) # [B*n*k]
    tid = torch.arange(B * N, device=dev) # [B*n]
    tid = tid.view(-1, 1).repeat(1, k).view(-1) # [B*n*k]
    return idx, sid, tid, pairwise_distance # [B*n*k]
def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points
class PointPlus(nn.Module):  # PointNet++
    def __init__(self, in_channels, out_channels, first_layer=False):
        super(PointPlus, self).__init__()
        self.first_layer = first_layer
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x, B, n, sid_euc, tid_euc):
        # x[B*N, C] sid/tid[B*n*k]
        #sid_euc, tid_euc = id_euc
        k = int(sid_euc.size(0) / B / n)

        if self.first_layer:
            x, norm = x[:, :3], x[:, 3:]
            x_i, x_j = x[tid_euc], x[sid_euc] # [B*n*k, C]
            norm_j = norm[sid_euc] # [B*n*k, C]
            edge = torch.cat([x_j - x_i, norm_j], dim=-1) # [B*n*k, C]
        else:
            x_i, x_j = x[sid_euc], x[tid_euc] # [B*n*k, C]
            edge = x_j - x_i
        edge = edge.view(B, n, k, -1) # [B, n, k, C]
        edge = self.fc1(edge.transpose(1,-1)) # [B, n, k, C]
        y  = edge.max(-2)[0].transpose(1,2) # [B, n, C]
        y = y.contiguous().view(B*n, -1) # [B*n, C]

        return y
class Local_point(nn.Module):
    def __init__(self, if_noise=True, noise_dim=3, noise_stdv=1e-2):
        super(Local_point, self).__init__()
        self.feat_conv = nn.Sequential(
            nn.Conv1d(640, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 64, 1),
        )

        self.seg_head = nn.Sequential(
            nn.Conv1d(80, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 3, 1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pointplus1 = PointPlus(in_channels=64, out_channels=128)
        self.pointplus2 = PointPlus(in_channels=128, out_channels=256)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: Tensor, (B, 2048, 3)
        """
        # b, npoint, _ = point_cloud.shape
        device = point_cloud.device
        b, n, _ = point_cloud.shape
        point_cloud = point_cloud.transpose(1, 2)

        knn_idx, sid, tid, distance = knn_gnn(point_cloud, k=8)#8
        knn_x = index_points(point_cloud.permute(0, 2, 1), knn_idx)  # (B, N, 8, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        covariances = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1).permute(0, 2, 1)

        l0_points_raw = torch.cat([point_cloud, covariances], dim=1)  # (B, 12, N)
        l0_points_raw = self.conv1(l0_points_raw)  # (B, 64, N)
        l0_points_raw = torch.flatten(l0_points_raw.transpose(1, 2), start_dim=0, end_dim=1)
        l0_points1 = self.pointplus1(l0_points_raw, b, n, sid, tid)
        l0_points1 = self.pointplus2(l0_points1, b, n, sid, tid)
        l0_points1 = l0_points1.view(b, n, -1)#.transpose(1,2)

        return l0_points1
def rotation_6d_to_matrix(d6):
    """
    将 6D 旋转向量转换为 3x3 旋转矩阵
    输入 d6: [Batch, 6]
    输出: [Batch, 3, 3]
    """
    # 1. 拆分为两个向量 a1, a2
    a1 = d6[:, 0:3]
    a2 = d6[:, 3:6]

    # 2. 得到第一列向量 b1 (归一化)
    b1 = F.normalize(a1, dim=-1)

    # 3. 得到第二列向量 b2 (正交化)
    # b2 = normalize(a2 - (b1 · a2) * b1)
    dot_product = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = a2 - dot_product * b1
    b2 = F.normalize(b2, dim=-1)

    # 4. 得到第三列向量 b3 (叉乘)
    b3 = torch.cross(b1, b2, dim=-1)

    # 5. 堆叠成旋转矩阵 [B, 3, 3]
    return torch.stack((b1, b2, b3), dim=-1)

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()

        self.displace_global = PointTransformerV3(in_channels=4,enc_patch_size=(512,512,512,512,512),dec_patch_size=(512,512,512,512))
        self.trans = (
            nn.Sequential(
                nn.Linear(128*64* 2 + 1024, 4096),
                nn.BatchNorm1d(4096),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(4096, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(64, 6),
            )
        )
        self.displace = (
            nn.Sequential(
                nn.Linear(128*64* 2 + 1024, 4096),
                nn.BatchNorm1d(4096),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(4096, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(64, 3),
            )
        )
        self.conv1 = nn.Sequential(
            nn.Linear(3*256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.stepconv = Local_point()
        self.relate_adj = ToothAdjacentRelationModule(dim=256)
        self.relate_topdowm = ToothTopLowRelationModule(dim=256)
        self.tooth_id_embedding = nn.Embedding(28, 1024)

        self.tooth_id_embedding_trans = nn.Embedding(28, 128)


    def forward(self, x):
        bs, c, n = x.shape
        device = x.device
        l0_points = x.permute(0, 2, 1).contiguous().view(bs * n, c)
        offset = torch.zeros([bs], device = device, dtype= torch.int64)

        x_local = x.transpose(2, 1).reshape(bs * 28, -1, 3)  # .transpose(2, 1)
        bs1, num_point, n1 = x_local.shape

        for i in range(bs):
            offset[i] = n * (i + 1)

        ids_trans = torch.arange(28).to(device).expand(bs, 28)
        id_encoding_trans = self.tooth_id_embedding_trans(ids_trans).view(bs * 28, num_point, 1)
        point_embedd = torch.cat([x_local, id_encoding_trans], dim=-1).view(bs*28*num_point, -1)

        data_dict = {'coord': l0_points,
                     'offset': offset, "grid_size": 0.01,
                     'feat': point_embedd}
        # feat = self.displace(data_dict)
        data_dict = self.displace_global(data_dict)['feat']

        data_dict = data_dict.view(bs * 28, -1)

        data_tooth = self.stepconv(x_local).view(bs, 28, num_point, -1)
        data_adj = self.relate_adj(data_tooth)
        data_topdowm = self.relate_topdowm(data_tooth)
        data_local = torch.cat([data_tooth, data_adj,data_topdowm], dim=-1).view(bs*28*num_point, -1)
        data_local = self.conv1(data_local)
        data_local = data_local.view(bs * 28, -1)

        ids = torch.arange(28).to(device).expand(bs, 28)
        id_encoding = self.tooth_id_embedding(ids).view(bs * 28, -1)


        x = self.trans(torch.cat([data_dict, data_local, id_encoding], dim=1))
        y = self.displace(torch.cat([data_dict, data_local, id_encoding], dim=1))
        return x, y

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss




class contact_loss(torch.nn.Module):
    def __init__(self):
        super(contact_loss, self).__init__()
    @torch.no_grad()
    def computer_pair(self,P, Q, mask):
        pair_mask = mask[:, :-1] & mask[:, 1:]

        dist = torch.cdist(P, Q)
        min_dist = dist.min(dim=-1)[0]

        # 取最小 k 个点
        min_dist_sorted, a = torch.sort(min_dist)

        dist1 = torch.cdist(Q, P)
        min_dist1 = dist1.min(dim=-1)[0]
        # 取最小 k 个点
        min_dist_sorted1, a1 = torch.sort(min_dist1)

        return a, a1, pair_mask, min_dist_sorted, min_dist_sorted1

    def contact_projection_loss(self, pred_pts_i, pred_pts_j, target_pts_i, target_pts_j, mask, min_dist1, min_dist2, margin=0.1):
        """
        pred_pts_i: [B, 20, 3] 预测的接触面点
        target_pts_i: [B, 20, 3] 目标的接触面点（真值）
        """
        # 1. 计算真值的方向轴线 (从 i 指向 j)
        min_max = (min_dist1 < 1) & (min_dist2 < 1)

        mask_float = min_max.unsqueeze(-1).float()
        count = torch.sum(mask_float, dim=1)

        count_mask = count != 0

        c_target_i = torch.sum(target_pts_i * mask_float, dim=1) / count.clamp(min=1e-8)
        c_target_j = torch.sum(target_pts_j * mask_float, dim=1) / count.clamp(min=1e-8)
        c_pred_i = torch.sum(pred_pts_i * mask_float, dim=1) / count.clamp(min=1e-8)
        c_pred_j = torch.sum(pred_pts_j * mask_float, dim=1) / count.clamp(min=1e-8)


        # 轴线方向 n
        axis_vec = c_target_j - c_target_i
        n_target = axis_vec / (torch.norm(axis_vec, dim=-1, keepdim=True) + 1e-8)

        # 中心投影距离
        center_dist = torch.sum((c_pred_j - c_pred_i) * n_target, dim=-1)

        # 3. [进阶] 约束接触面上的所有点 (防止边角相交)
        # 计算 j 组里的 20 个点相对于 i 中心的位移，投影到轴线上都必须大于 margin
        # pts_j_rel: [B, 20, 3]
        pts_j_rel = pred_pts_j - c_pred_i.unsqueeze(1)
        # point_dists: [B, 20]
        point_dists = torch.sum(pts_j_rel * n_target.unsqueeze(1), dim=-1)

        # 4. 损失计算
        # 约束1：中心距离要接近真值距离（引力）
        target_dist_val = torch.norm(axis_vec, dim=-1)
        loss_align = torch.pow(center_dist - target_dist_val, 2)

        # 约束2：所有点必须在平面之后（防穿模斥力）
        # 只要任何一个点的投影距离小于 margin，就产生重罚
        loss_collision = torch.relu(margin - point_dists).pow(2).mean(dim=-1)

        loss_flat = torch.var(point_dists, dim=-1)

        return (loss_align + 5.0 * loss_collision + 0.5 * loss_flat)[mask & count_mask.squeeze(-1)].mean()
    def forward(self, A, target, k=20, max_dist = 200):
        tooth_mask = (target.abs().sum(dim=(2, 3)) > 0)
        b = A.shape[0]

        # ---------- 上颌 ----------
        upper = target[:, :14]
        upper_mask = tooth_mask[:, :14]
        P = upper[:, :-1]
        Q = upper[:, 1:]
        index1, index2, pair_mask, min_dist1, min_dist2 = self.computer_pair(P, Q, upper_mask)
        upper_data = A[:, :14]
        index1 = index1[..., :k].unsqueeze(-1).expand(-1, -1, -1, 3)

        min_dist1 = min_dist1[..., :k]#.unsqueeze(-1).expand(-1, -1, -1, 3)
        P_data1 = torch.gather(upper_data[:, :-1], 2, index1).reshape(b * 13, k, 3)
        P_target1 = torch.gather(P, 2, index1).reshape(b * 13, k, 3)

        index2 = index2[..., :k].unsqueeze(-1).expand(-1, -1, -1, 3)
        P_data2 = torch.gather(upper_data[:, 1:], 2, index2).reshape(b * 13, k, 3)
        P_target2 = torch.gather(Q, 2, index2).reshape(b * 13, k, 3)
        min_dist2 = min_dist2[..., :k]

        #loss = (self.contact_projection_loss(P_data1, P_data2, P_target1, P_target2, pair_mask.reshape(b*13), min_dist1.reshape(b * 13, k), min_dist2.reshape(b * 13, k))
        #       + self.contact_projection_loss(P_data2, P_data1, P_target2, P_target1, pair_mask.reshape(b*13), min_dist1.reshape(b * 13, k), min_dist2.reshape(b * 13, k)))
        loss = self.contact_projection_loss(P_data1, P_data2, P_target1, P_target2, pair_mask.reshape(b * 13),
                                             min_dist1.reshape(b * 13, k), min_dist2.reshape(b * 13, k))


        upper = target[:, 14:]
        upper_mask = tooth_mask[:, 14:]
        P = upper[:, :-1]
        Q = upper[:, 1:]
        index1, index2, pair_mask, min_dist1, min_dist2 = self.computer_pair(P, Q, upper_mask)
        upper_data = A[:, 14:]
        index1 = index1[..., :k].unsqueeze(-1).expand(-1, -1, -1, 3)
        P_data1 = torch.gather(upper_data[:, :-1], 2, index1).reshape(b * 13, k, 3)
        P_target1 = torch.gather(P, 2, index1).reshape(b * 13, k, 3)
        min_dist1 = min_dist1[..., :k]

        index2 = index2[..., :k].unsqueeze(-1).expand(-1, -1, -1, 3)
        P_data2 = torch.gather(upper_data[:, 1:], 2, index2).reshape(b * 13, k, 3)
        P_target2 = torch.gather(Q, 2, index2).reshape(b * 13, k, 3)
        min_dist2 = min_dist2[..., :k]

        #loss = (self.contact_projection_loss(P_data1, P_data2, P_target1, P_target2, pair_mask.reshape(b*13), min_dist1.reshape(b * 13, k), min_dist2.reshape(b * 13, k))
         #      + self.contact_projection_loss(P_data2, P_data1, P_target2, P_target1, pair_mask.reshape(b*13), min_dist1.reshape(b * 13, k), min_dist2.reshape(b * 13, k))) + loss
        loss = self.contact_projection_loss(P_data1, P_data2, P_target1, P_target2, pair_mask.reshape(b * 13),
                                             min_dist1.reshape(b * 13, k), min_dist2.reshape(b * 13, k)) + loss

        return loss

def midline_loss(pred, target):
    center_pred = pred.mean(dim=2)
    center_gt = target.mean(dim=2)

    mid_pred = center_pred[:, 6:8].mean(dim=1)
    #mid_gt = center_gt[:, 6:8].mean(dim=1)

    #L_mid = ((mid_pred - mid_gt) ** 2).mean()

    mid_pred1 = center_pred[:, 20:22].mean(dim=1)
    #mid_gt1 = center_gt[:, 20:22].mean(dim=1)

    #L_mid = ((mid_pred1 - mid_gt1) ** 2).mean() + L_mid + 2 * ((mid_pred[:, 0] - mid_pred1[:, 0]) ** 2).mean()
    L_mid = 5 * ((mid_pred[:, 0] - mid_pred1[:, 0]) ** 2).mean()


    return L_mid
class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.contact_loss = contact_loss()
        self.abs_loss = nn.SmoothL1Loss(reduction='mean', beta=0.1)
        self.tre_loss = nn.SmoothL1Loss(reduction='mean', beta=0.1)
        #self.aloss = OBBCollisionLoss()

    def forward(self, m, centroid, points, transform, target, displace, epoch):
        points = points.transpose(2, 1)
        bs = points.shape[0]
        num_point = points.shape[1]
        per_tooth_num = num_point // 28

        point_norm = points.clone().reshape(bs, 28, per_tooth_num, -1)
        l_center_norm = point_norm.mean(dim=2, keepdim=True)
        p_rot = torch.matmul(point_norm - l_center_norm, rotation_6d_to_matrix(transform).reshape(bs, 28, 3, 3))
        pts_mid = (p_rot + l_center_norm + displace.reshape(bs, 28, 3).unsqueeze(2))
        pts_mid = pts_mid * m[..., None, None, None].expand(-1, 28, per_tooth_num, 3)
        pred_pc = pts_mid + centroid.unsqueeze(1).unsqueeze(1).expand(bs, 28, 1, 3)
        pred_pc = pred_pc.float()

        points = points.mul(m[..., None, None].expand(-1, 28 * per_tooth_num, 3))
        points = points.transpose(0, 1)
        centroid = centroid[None, ...].expand(28 * per_tooth_num, -1, -1)
        points = torch.add(points, centroid)
        points = points.transpose(0, 1)
        points = points.float()


        point_cloud = points[:, :, 0:3].reshape(bs, 28, per_tooth_num, -1)  # .permute(0, 1, 3, 2)

        transform_target = target[:, :, 0:3, 0:3]
        displace_target = target[:, :, 0:3, 3]

        target_pc = torch.matmul(point_cloud, transform_target) + displace_target[:, :, None, :].expand(-1, -1,
                                                                                                        per_tooth_num,
                                                                                                        -1)

        tre_loss = self.tre_loss(pred_pc, target_pc)  # self.abs_loss(pred_pc, target_pc)#absloss(pred_pc, target_pc)

        cd_loss = chamfer_sqrt(pred_pc.reshape(bs, num_point, 3),
                               target_pc.reshape(bs, num_point, 3))  # torch.mean(torch.abs(pred_pc - target_pc))


        pred_center = torch.mean(pred_pc, dim=-2)

        target_center = torch.mean(target_pc, dim=-2)

        tre_center = self.abs_loss(pred_center, target_center)  # absloss(pred_center, target_center)

        if (epoch > 10):
            L_contact = self.contact_loss(pred_pc, target_pc) * 0.01#0.01 m  # self.obblose(pred_pc, target_pc) * 0.001#self.contact_loss(A=pred_pc, target=target_pc) * 0.01  # 0.05
            L_mid = midline_loss(pred_pc, target_pc) * 0.01#0.01
            total_loss = tre_loss + (tre_center) + L_contact + L_mid#0.01
        else:
            L_contact = self.contact_loss(pred_pc, target_pc) * 0.01 #0.01 self.obblose(pred_pc, target_pc) * 0.001#self.contact_loss(A=pred_pc, target=target_pc) * 0.01  # 0.05
            L_mid = midline_loss(pred_pc, target_pc) * 0.01#0.01
            total_loss = tre_loss + tre_center  # * 1.5

        return total_loss, mse(pred_pc, target_pc), tre_loss, absloss(pred_pc,
                                                                      target_pc), cd_loss, L_contact, L_mid, tre_center, tre_center

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = max(np.max(np.sqrt(np.sum(pc ** 2, axis=1))), 0.0000001)
    pc = pc / m
    return pc, m, centroid
def sample_non_adjacent(n, k, min_dist=2):
    """
    n: 总范围 [0, n-1]
    k: 取几个值
    min_dist: 最小间距（min_dist=2 表示不相邻）
    """
    indices = []
    candidates = list(range(n))

    for _ in range(k):
        if not candidates:
            break
        # 随机选一个
        idx = np.random.choice(candidates)
        indices.append(idx)

        # 剔除不符合间距限制的值
        candidates = [c for c in candidates if abs(c - idx) >= min_dist]

    return sorted(indices)
def IDR_twotooth(data, label):
    data = data.reshape(28, 128, 3)
    data_mask = np.sum(np.sum(data, axis=1), axis=1)
    data_mask = data_mask != 0
    data_mask_1 = data_mask == 0
    #data_2 = np.matmul(data, label[:, 0:3, 0:3]) + label[:, 0:3, 3].reshape(28, 1, 3)
    #data_temp = data[data_mask]
    #data_target = np.matmul(data, label[:, 0:3, 0:3]) + label[:, 0:3, 3].reshape(28, 1, 3)
    #np.savetxt('submesh_loss/ground_truth.txt', data_target[data_mask].reshape(data_temp.shape[0]*128, 3).astype(np.float32))
    #np.savetxt('submesh_loss/input.txt', data[data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))
    #np.savetxt('submesh_loss/ground_truth_2.txt', data_2[:, :, 0:3][data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))

    num, num_points, c = data.shape

    x = sample_non_adjacent(14, np.random.randint(1, 6), min_dist=2)
    x = np.array(x)
    x[x > 6] += 10
    x[x < 7] += 3
    for i in x:
        if (random.random() < 0.5):
            i1 = i + 1
            if (data_mask_1[i] or data_mask_1[i1]):
                continue
        else:
            i1 = i - 1
            if (data_mask_1[i] or data_mask_1[i1]):
                continue
        center1 = np.mean(data[i, ...], axis=0)
        center2 = np.mean(data[i1, ...], axis=0)
        a1 = np.random.uniform(0, 0.2)
        a2 = np.random.uniform(0, 0.2)
        ve1 = (center2 - center1) * a1
        ve2 = (center2 - center1) * a2

        data[i, ...] = data[i, ...] + ve1
        displace1 = label[i, 0:3, 3] - np.matmul(ve1, label[i, 0:3, 0:3])
        label[i, 0:3, 3] = displace1

        data[i1, ...] = data[i1, ...] - ve2
        displace2 = label[i1, 0:3, 3] + np.matmul(ve2, label[i1, 0:3, 0:3])
        label[i1, 0:3, 3] = displace2

        #data_target_au = np.matmul(data, label[:, 0:3, 0:3]) + label[:, 0:3, 3].reshape(28, 1, 3)
        #np.savetxt('submesh_loss/ground_truth_au.txt',
        #           data_target_au[data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))
        #np.savetxt('submesh_loss/input_au.txt', data[data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))
    return data.reshape(num * num_points, 3), label
def IDR_singletooth(data, label):
    data = data.reshape(28, 128, 3)
    data_mask = np.sum(np.sum(data, axis=1), axis=1)
    data_mask = data_mask != 0
    data_mask_1 = data_mask == 0
    #data_2 = np.matmul(data, label[:, 0:3, 0:3]) + label[:, 0:3, 3].reshape(28, 1, 3)
    #data_temp = data[data_mask]
    #data_target = np.matmul(data, label[:, 0:3, 0:3]) + label[:, 0:3, 3].reshape(28, 1, 3)
    #np.savetxt('submesh_loss/ground_truth.txt', data_target[data_mask].reshape(data_temp.shape[0]*128, 3).astype(np.float32))
    #np.savetxt('submesh_loss/input.txt', data[data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))
    #np.savetxt('submesh_loss/ground_truth_2.txt', data_2[:, :, 0:3][data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))

    num, num_points, c = data.shape
    data_au = np.zeros_like(data)
    label_au = np.zeros_like(label)

    for i in range(data.shape[0]):
        if (random.random() < 0.2):
            displace = np.zeros(3)
            displace[0] = np.random.uniform(-1, 1)
            displace[1] = np.random.uniform(-1.5, 1.5)
            displace[2] = np.random.uniform(-0.7, 0.7)

            # rot = np.random.rand(3)# * 30
            tip = np.random.uniform(-5, 5)
            torque = np.random.uniform(-8, 8)
            rotation = np.random.uniform(-20, 20)

            #displace[0] = np.random.uniform(-0.5, 0.5)
            #displace[1] = np.random.uniform(-0.5, 0.5)
            #displace[2] = np.random.uniform(-0.5, 0.5)

            # rot = np.random.rand(3)# * 30
            #tip = np.random.uniform(-5, 5)
            #torque = np.random.uniform(-5, 5)
            #rotation = np.random.uniform(-5, 5)

            r1 = Rotation.from_euler('zyx', [rotation, torque, tip], degrees=True)  # 顺序和角度
            R = r1.as_matrix()

            au_R = R.T

            # 求解逆矩阵
            inverse_R = np.linalg.inv(au_R)

            normal_data = data[i, ...]
            single_labe = label[i, ...]
            centroid = np.mean(normal_data, axis=0)
            normal_data = normal_data - centroid

            A1 = np.matmul(normal_data, au_R) + displace + centroid
            after_R = np.matmul(inverse_R, single_labe[0:3, 0:3])
            after_T = np.matmul(centroid, single_labe[0:3, 0:3]) + single_labe[0:3, 3]
            after_T_temp = np.matmul(centroid, inverse_R) + np.matmul(displace, inverse_R)
            after_T = after_T - np.matmul(after_T_temp, single_labe[0:3, 0:3])

            data_au[i, ...] = A1
            label_au[i, 0:3, 0:3] = after_R
            label_au[i, 0:3, 3] = after_T
        else:
            data_au[i, ...] = data[i, ...]
            label_au[i, ...] = label[i, ...]


    data_au[data_mask_1] = 0
    label_au[data_mask_1] = 0

    #data_target_au = np.matmul(data_au, label_au[:, 0:3, 0:3]) + label_au[:, 0:3, 3].reshape(28, 1, 3)
    #np.savetxt('submesh_loss/ground_truth_au.txt', data_target_au[data_mask].reshape(data_temp.shape[0]*128, 3).astype(np.float32))
    #np.savetxt('submesh_loss/input_au.txt', data_au[data_mask].reshape(data_temp.shape[0]*128, 3).astype(np.float32))
    return data_au.reshape(num * num_points, 3), label_au

def IDR_singlerotate(data, label):
    data = data.reshape(28, 128, 3)
    data_mask = np.sum(np.sum(data, axis=1), axis=1)
    data_mask = data_mask != 0
    data_mask_1 = data_mask == 0

    num, num_points, c = data.shape
    data_au = np.zeros_like(data)
    label_au = np.zeros_like(label)

    for i in range(data.shape[0]):
        if (random.random() < 0.5):
            displace = np.zeros(3)
            if(i in [6, 7]):#上颌中切牙
                r_x = np.random.uniform(-35, 35)
                r_z = np.random.uniform(-25, 25)
                r_y = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('xzy', [r_x, r_z, r_y], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-1.5, 1.5)
                displace[1] = np.random.uniform(-2, 2)
                displace[2] = np.random.uniform(-1.5, 1.5)
            elif(i in [5, 8]):#上颌侧切牙
                r_x = np.random.uniform(-10, 10)
                r_z = np.random.uniform(-45, 45)
                r_y = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('zxy', [r_z, r_x, r_y], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-1.5, 1.5)
                displace[1] = np.random.uniform(-2, 2)
                displace[2] = np.random.uniform(-1.5, 1.5)
            elif (i in [4, 9, 18, 23]):#上下颌尖牙
                r_z = np.random.uniform(-40, 40)
                r_y = np.random.uniform(-25, 25)
                r_x = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('zyx', [r_z, r_y, r_x], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-2, 2)
                displace[1] = np.random.uniform(-2, 2)
                displace[2] = np.random.uniform(-1.5, 1.5)
            elif (i in [0, 1, 12, 13, 14, 15, 26, 27]):#磨牙(cunyi)
                r_y = np.random.uniform(-25, 25)
                r_x = np.random.uniform(-10, 10)
                r_z = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('yzx',[r_y, r_z, r_x], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-0.5, 0.5)
                displace[1] = np.random.uniform(-1, 1)
                displace[2] = np.random.uniform(-0.5, 0.5)
            elif (i in [2, 3, 10, 11, 16, 17, 24, 25]):#前磨牙
                r_x = np.random.uniform(-20, 20)
                r_z = np.random.uniform(-20, 20)
                r_y = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('xzy', [r_x, r_z, r_y], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-1, 1)
                displace[1] = np.random.uniform(-1, 1)
                displace[2] = np.random.uniform(-1, 1)
            else:
                displace[0] = np.random.uniform(-0.5, 0.5)
                displace[1] = np.random.uniform(-0.5, 0.5)
                displace[2] = np.random.uniform(-0.5, 0.5)

                # rot = np.random.rand(3)# * 30
                tip = np.random.uniform(-10, 10)
                torque = np.random.uniform(-10, 10)
                rotation = np.random.uniform(-10, 10)

                r1 = Rotation.from_euler('zyx', [rotation, torque, tip], degrees=True)  # 顺序和角度

            R = r1.as_matrix()

            au_R = R.T

            # 求解逆矩阵
            inverse_R = np.linalg.inv(au_R)

            normal_data = data[i, ...]
            single_labe = label[i, ...]
            centroid = np.mean(normal_data, axis=0)
            normal_data = normal_data - centroid

            A1 = np.matmul(normal_data, au_R) + displace + centroid
            after_R = np.matmul(inverse_R, single_labe[0:3, 0:3])
            after_T = np.matmul(centroid, single_labe[0:3, 0:3]) + single_labe[0:3, 3]
            after_T_temp = np.matmul(centroid, inverse_R) + np.matmul(displace, inverse_R)
            after_T = after_T - np.matmul(after_T_temp, single_labe[0:3, 0:3])

            data_au[i, ...] = A1
            label_au[i, 0:3, 0:3] = after_R
            label_au[i, 0:3, 3] = after_T
        else:
            data_au[i, ...] = data[i, ...]
            label_au[i, ...] = label[i, ...]


    data_au[data_mask_1] = 0
    label_au[data_mask_1] = 0

    #data_target_au = np.matmul(data_au, label_au[:, 0:3, 0:3]) + label_au[:, 0:3, 3].reshape(28, 1, 3)
    #np.savetxt('submesh_loss/ground_truth_au.txt', data_target_au[data_mask].reshape(data_temp.shape[0]*128, 3).astype(np.float32))
    #np.savetxt('submesh_loss/input_au.txt', data_au[data_mask].reshape(data_temp.shape[0]*128, 3).astype(np.float32))
    return data_au.reshape(num * num_points, 3), label_au


def IDR_teeth(data, label):
    data = data.reshape(28, 128, 3)

    data_mask = np.sum(np.sum(data, axis=1), axis=1)
    data_mask = data_mask != 0
    data_mask_1 = data_mask == 0

    #data_2 = np.matmul(data, label[:, 0:3, 0:3]) + label[:, 0:3, 3].reshape(28, 1, 3)

    data_temp = data[data_mask]

    #data_target = np.matmul(data, label[:, 0:3, 0:3]) + label[:, 0:3, 3].reshape(28, 1, 3)
    #np.savetxt('submesh_loss/ground_truth.txt',
    #           data_target[data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))
    #np.savetxt('submesh_loss/input.txt', data[data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))
    #np.savetxt('submesh_loss/ground_truth_2.txt',
    #           data_2[:, :, 0:3][data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))

    centroid = np.mean(data[data_mask].reshape(data_temp.shape[0] * 128, 3), axis=0)
    data = data - centroid

    data = data.reshape(28, 128, 3)
    num, num_points, c = data.shape


    displace = np.zeros(3)
    displace[0] = np.random.uniform(-1.0, 1.0)
    displace[1] = np.random.uniform(-1.5, 1.5)
    displace[2] = np.random.uniform(-0.5, 0.5)
    #rot = np.random.rand(3)# * 30
    rx = np.random.uniform(-10, 10)
    ry = np.random.uniform(-10, 10)
    rz = np.random.uniform(-30, 30)

    r1 = Rotation.from_euler('zyx', [rz, ry, rx], degrees=True)  # 顺序和角度
    R = r1.as_matrix()

    au_R = R.T

    # 求解逆矩阵
    inverse_R = np.linalg.inv(au_R)

    data_au = np.zeros_like(data)
    label_au = np.zeros_like(label)

    for i in range(data.shape[0]):
        normal_data = data[i, ...]
        single_labe = label[i, ...]

        A1 = np.matmul(normal_data, au_R) + displace + centroid
        after_R = np.matmul(inverse_R, single_labe[0:3, 0:3])
        after_R = np.matmul(after_R, au_R)

        after_T = np.matmul(centroid, single_labe[0:3, 0:3]) + single_labe[0:3, 3]
        after_T_temp = np.matmul(centroid, inverse_R) + np.matmul(displace, inverse_R)
        after_T = after_T - np.matmul(after_T_temp, single_labe[0:3, 0:3])
        after_T = np.matmul(after_T, au_R) + centroid + displace - np.matmul(centroid, au_R)

        data_au[i, ...] = A1
        label_au[i, 0:3, 0:3] = after_R
        label_au[i, 0:3, 3] = after_T
    data_au[data_mask_1] = 0
    label_au[data_mask_1] = 0
    #data_target_au = np.matmul(data_au, label_au[:, 0:3, 0:3]) + label_au[:, 0:3, 3].reshape(28, 1, 3)
    #np.savetxt('submesh_loss/ground_truth_au.txt',
    #           data_target_au[data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))
    #np.savetxt('submesh_loss/input_au.txt', data_au[data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))
    return data_au.reshape(num * num_points, 3), label_au
def IDR_autooth(data, label):
    data = data.reshape(28, 128, 3)
    data_mask = np.sum(np.sum(data, axis=1), axis=1)
    data_mask = data_mask != 0
    data_mask_1 = data_mask == 0
    data_2 = np.matmul(data, label[:, 0:3, 0:3]) + label[:, 0:3, 3].reshape(28, 1, 3)
    #data_temp = data[data_mask]
    #np.savetxt('submesh_loss/ground_truth.txt', data_2[data_mask].reshape(data_temp.shape[0]*128, 3).astype(np.float32))
    #np.savetxt('submesh_loss/input.txt', data[data_mask].reshape(data_temp.shape[0] * 128, 3).astype(np.float32))


    num, num_points, c = data_2.shape
    data_au = np.zeros_like(data_2)
    label_au = np.zeros_like(label)

    for i in range(data.shape[0]):
        if (random.random() < 1.0):
            displace = np.zeros(3)
            if (i in [6, 7]):  # 上颌中切牙
                r_x = np.random.uniform(-35, 35)
                r_z = np.random.uniform(-25, 25)
                r_y = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('xzy', [r_x, r_z, r_y], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-1.5, 1.5)
                displace[1] = np.random.uniform(-2, 2)
                displace[2] = np.random.uniform(-1.5, 1.5)
            elif (i in [5, 8]):  # 上颌侧切牙
                r_x = np.random.uniform(-10, 10)
                r_z = np.random.uniform(-45, 45)
                r_y = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('zxy', [r_z, r_x, r_y], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-1.5, 1.5)
                displace[1] = np.random.uniform(-2, 2)
                displace[2] = np.random.uniform(-1.5, 1.5)
            elif (i in [4, 9, 18, 23]):  # 上下颌尖牙
                r_z = np.random.uniform(-40, 40)
                r_y = np.random.uniform(-25, 25)
                r_x = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('zyx', [r_z, r_y, r_x], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-2, 2)
                displace[1] = np.random.uniform(-2, 2)
                displace[2] = np.random.uniform(-1.5, 1.5)
            elif (i in [0, 1, 12, 13, 14, 15, 26, 27]):  # 磨牙(cunyi)
                r_y = np.random.uniform(-25, 25)
                r_x = np.random.uniform(-10, 10)
                r_z = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('yzx', [r_y, r_z, r_x], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-0.5, 0.5)
                displace[1] = np.random.uniform(-1, 1)
                displace[2] = np.random.uniform(-0.5, 0.5)
            elif (i in [2, 3, 10, 11, 16, 17, 24, 25]):  # 前磨牙
                r_x = np.random.uniform(-20, 20)
                r_z = np.random.uniform(-20, 20)
                r_y = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('xzy', [r_x, r_z, r_y], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-1, 1)
                displace[1] = np.random.uniform(-1, 1)
                displace[2] = np.random.uniform(-1, 1)
            else:
                displace[0] = np.random.uniform(-0.5, 0.5)
                displace[1] = np.random.uniform(-0.5, 0.5)
                displace[2] = np.random.uniform(-0.5, 0.5)

                # rot = np.random.rand(3)# * 30
                tip = np.random.uniform(-10, 10)
                torque = np.random.uniform(-10, 10)
                rotation = np.random.uniform(-10, 10)

                r1 = Rotation.from_euler('zyx', [rotation, torque, tip], degrees=True)  # 顺序和角度

            '''
            if (i in [6, 7]):  # 上颌中切牙
                r_x = np.random.uniform(-35, 35)
                r_z = np.random.uniform(-25, 25)
                r1 = Rotation.from_euler('xz', [r_x, r_z], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-1.5, 1.5)
                displace[1] = np.random.uniform(-2, 2)
                displace[2] = np.random.uniform(-1.5, 1.5)
            elif (i in [5, 8]):  # 上颌侧切牙
                r_x = np.random.uniform(-10, 10)
                r_z = np.random.uniform(-45, 45)
                r1 = Rotation.from_euler('zx', [r_z, r_x], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-1.5, 1.5)
                displace[1] = np.random.uniform(-2, 2)
                displace[2] = np.random.uniform(-1.5, 1.5)
            elif (i in [4, 9, 18, 23]):  # 上下颌尖牙
                r_z = np.random.uniform(-40, 40)
                r_y = np.random.uniform(-25, 25)
                r_x = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('zyx', [r_z, r_y, r_x], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-2, 2)
                displace[1] = np.random.uniform(-2, 2)
                displace[2] = np.random.uniform(-1.5, 1.5)
            elif (i in [0, 1, 12, 13, 14, 15, 26, 27]):  # 磨牙
                r_y = np.random.uniform(-25, 25)
                r1 = Rotation.from_euler('y', r_y, degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-0.5, 0.5)
                displace[1] = np.random.uniform(-1, 1)
                displace[2] = np.random.uniform(-0.5, 0.5)
            elif (i in [2, 3, 10, 11, 16, 17, 24, 25]):  # 前磨牙
                r_x = np.random.uniform(-20, 20)
                r_z = np.random.uniform(-20, 20)
                r_y = np.random.uniform(-10, 10)
                r1 = Rotation.from_euler('xzy', [r_x, r_z, r_y], degrees=True)  # 顺序和角度
                displace[0] = np.random.uniform(-1, 1)
                displace[1] = np.random.uniform(-1, 1)
                displace[2] = np.random.uniform(-1, 1)
            else:
                displace[0] = np.random.uniform(-0.5, 0.5)
                displace[1] = np.random.uniform(-0.5, 0.5)
                displace[2] = np.random.uniform(-0.5, 0.5)

                # rot = np.random.rand(3)# * 30
                tip = np.random.uniform(-10, 10)
                torque = np.random.uniform(-10, 10)
                rotation = np.random.uniform(-10, 10)

                r1 = Rotation.from_euler('zyx', [rotation, torque, tip], degrees=True)  # 顺序和角度
            '''
            R = r1.as_matrix()

            au_R = R.T

            # 求解逆矩阵
            inverse_R = np.linalg.inv(au_R)

            normal_data = data_2[i, ...]
            centroid = np.mean(normal_data, axis=0)
            normal_data = normal_data - centroid

            A1 = np.matmul(normal_data, au_R) + displace + centroid
            after_T = centroid - np.matmul(centroid, inverse_R) - np.matmul(displace, inverse_R)

            data_au[i, ...] = A1
            label_au[i, 0:3, 0:3] = inverse_R
            label_au[i, 0:3, 3] = after_T
        else:
            data_au[i, ...] = data[i, ...]
            label_au[i, ...] = label[i, ...]

    data_au[data_mask_1] = 0
    label_au[data_mask_1] = 0

    #data_target_au = np.matmul(data_au, label_au[:, 0:3, 0:3]) + label_au[:, 0:3, 3].reshape(28, 1, 3)
    #np.savetxt('submesh_loss/ground_truth_au.txt', data_target_au[data_mask].reshape(data_temp.shape[0]*128, 3).astype(np.float32))
    #np.savetxt('submesh_loss/input_au.txt', data_au[data_mask].reshape(data_temp.shape[0]*128, 3).astype(np.float32))
    return data_au.reshape(num * num_points, 3), label_au




class MalDataLoader(Dataset):
    def __init__(self, split):
        data_path = split + '.npy'
        solution_path = split[:-4] + 'label.npy'
        if not os.path.exists(data_path) or not os.path.exists(solution_path):
            print('error: data not found')
        else:
            print('Load processed data from %s...' % split)
            self.list_of_points = np.load(data_path)
            self.list_of_labels = np.load(solution_path)
        print(self.list_of_labels.shape, self.list_of_points.shape)

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]
        point_set = point_set.reshape(28 * 128, 3)

        if (random.random() < 0.5):
            point_set, label = IDR_autooth(point_set, label)
        else:
            if (random.random() < 0.2):
                point_set, label = IDR_twotooth(point_set, label)
            if (random.random() < 0.5):
                point_set, label = IDR_singletooth(point_set, label)

        point_set[:, 0:3], m, centroid = pc_normalize(point_set[:, 0:3])
        return point_set, label, m, centroid

    def __getitem__(self, index):
        return self._get_item(index)
class ValDataLoader(Dataset):
    def __init__(self, split):
        data_path = split + '.npy'
        solution_path = split[:-4] + 'label.npy'
        if not os.path.exists(data_path) or not os.path.exists(solution_path):
            print('error: data not found')
        else:
            print('Load processed data from %s...' % split)
            self.list_of_points = np.load(data_path)
            self.list_of_labels = np.load(solution_path)
        print(self.list_of_labels.shape, self.list_of_points.shape)

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]
        point_set[:, 0:3], m, centroid = pc_normalize(point_set[:, 0:3])
        return point_set, label, m, centroid

    def __getitem__(self, index):
        return self._get_item(index)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=5, type=int, help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--k', type=int, default=0, help='k for k-fold')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def normalize(pts):
        # pts: [B, 28, 128, 3]
        # 计算每颗牙齿自己的中心 [B, 28, 1, 3]
        center = pts.mean(dim=2, keepdim=True)
        # 计算缩放因子（通常以所有点到中心的平均距离为准）
        dist = torch.norm(pts - center, dim=-1, keepdim=True)
        scale = dist.max(dim=2, keepdim=True)[0] + 1e-8

        normalized_pts = (pts - center) / scale
        return normalized_pts, center, scale
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    seed = args.seed
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('malocclusion')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    train_dataset = MalDataLoader(
        '/home/buaaa302/ISICDM-ATRC-data/starting_kit/tsing/train_data')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)

    val_dataset = ValDataLoader(
        '/home/buaaa302/ISICDM-ATRC-data/starting_kit/tsing/test_data')
    valDataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10,
                                                drop_last=True)

    '''MODEL LOADING'''
    model = get_model(k=args.num_category, normal_channel=args.use_normals)
    criterion = get_loss()

    if not args.use_cpu:
        model = model.cuda()
        criterion = criterion.cuda()

    log_string('Training from scratch...')
    start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)#StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0

    '''TRANING'''
    best_tre_mean = 100
    best_tre_std = 100
    best_mse_mean = 100
    best_mse_std = 100
    best_cd_mean = 100
    best_cd_std = 100
    best_muti_mean = 1
    best_muti_std = 1
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        model = model.train()

        total_loss = 0


        for batch_id, (points, target,m, centroid) in enumerate(trainDataLoader, 0):
            optimizer.zero_grad()
            m = m.cuda()
            centroid = centroid.cuda()
            points = points.data.numpy()
            points = torch.Tensor(points)
            points = points.reshape(16, 28, 128, 3)
            points_perm = torch.randperm(points.shape[-2]).to(points.device)
            points = points[:, :, points_perm, :].reshape(16, 28 * 128, 3)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda().float()

            pred, trans_feat = model(points)
            loss, _, tre1, _, tre2, tre3, tre4, tre5, tre6 = criterion(m, centroid, points, pred, target, trans_feat, epoch)

            log_string(
                '[Epoch %d/%d] loss = %.3f loss1 = %.3f loss2 = %.3f loss3 = %.3f loss4 = %.3f loss5 = %.3f loss6 = %.3f' %
                (   epoch,
                    args.epoch,
                    loss.item(),
                    tre1.item(),
                    tre2.item(),
                    tre3.item(),
                    tre4.item(),
                    tre5.item(),
                    tre6.item()
                 ))


            loss.backward()
            optimizer.step()
            global_step += 1
            total_loss += loss
        #scheduler.step()
        scheduler.step()
        log_string('Train loss: %f' % (total_loss / (len(trainDataLoader) * 1.0)))

        with torch.no_grad():
            model.eval()
            TRE_list = []
            Muti_list = []
            CD_list = []
            MSE_list = []
            for batch_val_id, (points, target,m, centroid) in enumerate(valDataLoader, 0):
                points = points.data.numpy()
                m = m.cuda()
                centroid = centroid.cuda()
                # points = provider.random_point_dropout(points)
                # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)
                points = points.transpose(2, 1)

                if not args.use_cpu:
                    points, target = points.cuda(), target.cuda().float()

                pred, trans_feat = model(points)
                _, loss_MSE, loss_muti, loss_TRE, loss_CD, _, _,_,_ = criterion(m, centroid, points, pred, target,
                                                                            trans_feat, epoch)
                TRE_list.append(loss_TRE.item())
                Muti_list.append(loss_muti.item())
                CD_list.append(loss_CD.item())
                MSE_list.append(loss_MSE.item())
            TRE_mean = np.mean(TRE_list)
            TRE_std = np.std(TRE_list)
            Muti_mean = np.mean(Muti_list)
            Muti_std = np.std(Muti_list)
            CD_mean = np.mean(CD_list)
            CD_std = np.std(CD_list)
            MSE_mean = np.mean(MSE_list)
            MSE_std = np.std(MSE_list)

            log_string(
                '[Epoch %d/%d] TRE = %.3f + %.3f CD = %.3f + %.3f MSE = %.3f + %.3f Muti = %.3f + %.3f bestTRE = %.3f + %.3f bestCD = %.3f + %.3f bestMSE = %.3f + %.3f bestMuti = %.3f + %.3f' %
                (epoch,
                 args.epoch,
                 TRE_mean,
                 TRE_std,
                 CD_mean,
                 CD_std,
                 MSE_mean,
                 MSE_std,
                 Muti_mean,
                 Muti_std,
                 best_tre_mean,
                 best_tre_std,
                 best_cd_mean,
                 best_cd_std,
                 best_mse_mean,
                 best_mse_std,
                 best_muti_mean,
                 best_muti_std
                 ))
            if (epoch + 1) % 50 == 0:
                file_name = 'ckpt' + str(epoch + 1) + '.pth'
                output_path = os.path.join(checkpoints_dir, file_name)
                torch.save({
                    'model': model.state_dict()
                }, output_path)

                log_string('Saved checkpoint to %s ...' % output_path)
            if TRE_mean < best_tre_mean:
                file_name = 'ckpt-best.pth'
                output_path = os.path.join(checkpoints_dir, file_name)
                torch.save({
                    'model': model.state_dict()
                }, output_path)

                log_string('Saved checkpoint to %s ...' % output_path)
                if TRE_mean < best_tre_mean:
                    best_tre_std = TRE_std
                    best_tre_mean = TRE_mean
                    best_muti_mean = Muti_mean
                    best_muti_std = Muti_std
                    best_cd_std = CD_std
                    best_cd_mean = CD_mean
                    best_mse_mean = MSE_mean
                    best_mse_std = MSE_std


    logger.info('End of training...')




if __name__ == '__main__':
    args = parse_args()
    main(args)
