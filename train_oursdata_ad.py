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


def mse(pred, target):
    return torch.mean(torch.sqrt(torch.sum((pred - target) ** 2, dim=-1)))
def absloss(pred, target):
    return torch.mean(torch.abs(pred - target))
from scipy.spatial.transform import Rotation
class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()

        #self.displace = PointTransformerV3(in_channels=3)
        #self.displace = PointTransformerV3(in_channels=3,enc_patch_size=(512,512,512,512,512),dec_patch_size=(512,512,512,512))
        self.displace_global = PointTransformerV3(in_channels=3,enc_patch_size=(768,768,768,768,768),dec_patch_size=(768,768,768,768))
        self.displace_local = PointTransformerV3(in_channels=3,enc_patch_size=(64, 64, 64, 64, 64),dec_patch_size=(64, 64, 64, 64))#3.96
        #self.displace_local = PointTransformerV3(in_channels=3, enc_patch_size=(48,48,48,48,48),
        #                                         dec_patch_size=(48,48,48,48))
        #self.displace_local = PointTransformerV3(in_channels=3, enc_patch_size=(96,96,96,96,96),
        #                                         dec_patch_size=(96,96,96,96))
        self.trans = (
            nn.Sequential(
                nn.Linear(128*64*2, 4096),
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
                nn.Linear(64, 9),
            )
        )
        self.displace = (
            nn.Sequential(
                nn.Linear(128*64*2, 4096),
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

    def forward(self, x):
        bs, c, n = x.shape
        device = x.device
        l0_points = x.permute(0, 2, 1).contiguous().view(bs * n, c)
        offset = torch.zeros([bs], device = device, dtype= torch.int64)
        for i in range(bs):
            offset[i] = n * (i + 1)
        data_dict = {'coord': l0_points,
                     'offset': offset, "grid_size": 0.01,
                     'feat': l0_points}
        # feat = self.displace(data_dict)
        data_dict = self.displace_global(data_dict)['feat']
        data_dict = data_dict.view(bs*28, -1)

        x_local = x.transpose(2, 1).reshape(bs * 28, -1, 3).transpose(2, 1)
        bs1, _, n1 = x_local.shape
        x_local = x_local.permute(0, 2, 1).contiguous().view(bs * n, c)

        offset1 = torch.zeros([bs1], device=device, dtype=torch.int64)
        for i in range(bs1):
            offset1[i] = n1 * (i + 1)
        data_dict1 = {'coord': x_local,
                     'offset': offset1, "grid_size": 0.01,
                     'feat': x_local}
        # feat = self.displace(data_dict)
        data_dict1 = self.displace_local(data_dict1)['feat']
        data_dict1 = data_dict1.view(bs * 28, -1)



        # feat = data_dict['feat']
        x = self.trans(torch.cat([data_dict, data_dict1], dim=1))
        y = self.displace(torch.cat([data_dict, data_dict1], dim=1))

        return x, y
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, m, centroid, points, transform, target, displace):
        points = points.transpose(2, 1)
        bs = points.shape[0]
        num_point = points.shape[1]
        per_tooth_num = num_point // 28

        points = points.mul(m[...,None,None].expand(-1,28*per_tooth_num,3))
        points = points.transpose(0, 1)
        centroid = centroid[None,...].expand(28*per_tooth_num, -1, -1)
        points = torch.add(points, centroid)
        points = points.transpose(0, 1)
        points = points.float()

        '''
        point_cloud_cat = torch.cat([points[:, :, 0:3], torch.ones([bs, num_point, 1]).cuda()], dim=2)
        point_cloud_cat = point_cloud_cat.reshape(bs, 24, per_tooth_num, -1)

        pred = pred.reshape(bs, 24, 4, 4)
        pred_pc = torch.matmul(point_cloud_cat, pred)[:, :, :, 0:3]
        pred_pc = pred_pc.reshape(bs, num_point, -1)

        target_pc = torch.matmul(point_cloud_cat, target)[:, :, :, 0:3]
        target_pc = target_pc.reshape(bs, num_point, -1)

        TRE = torch.mean(torch.abs(pred_pc - target_pc))
        '''
        #point_cloud = torch.cat([points[:, :, 0:3], torch.ones([bs, num_point, 1]).cuda()], dim=2)
        #point_cloud = points[:, :, 0:3].reshape(bs, 28, per_tooth_num, -1)#.permute(0, 1, 3, 2)

        point_cloud = torch.cat([points[:, :, 0:3], torch.ones([bs, num_point, 1]).cuda()], dim=2)
        point_cloud = point_cloud.reshape(bs, 28, per_tooth_num, -1).permute(0, 1, 3, 2)

        target = target.reshape(bs, 28, 4, 4)
        pred = target.clone()
        pred[:, :, 0:3, 0:3] = transform.reshape(bs, 28, 3, 3)
        pred[:, :, 0:3, 3] = displace.reshape(bs, 28, 3)

        pred_pc = torch.matmul(pred, point_cloud).permute(0, 1, 3, 2)[:, :, :, 0:3]
        target_pc = torch.matmul(target, point_cloud).permute(0, 1, 3, 2)[:, :, :, 0:3]

        tre_loss = absloss(pred_pc, target_pc)

        cd_loss = chamfer_sqrt(pred_pc.reshape(bs, num_point, 3),
                            target_pc.reshape(bs, num_point, 3))  # torch.mean(torch.abs(pred_pc - target_pc))

        true_transformer = torch.zeros([bs, 28, 3, 3], device=target.device)
        true_transformer[:, :, 0, 0] = 1
        true_transformer[:, :, 1, 1] = 1
        true_transformer[:, :, 2, 2] = 1

        raw = transform.reshape(bs, 28, 3, 3)
        transform_T = raw.transpose(-1,-2)
        #print(transform_T.shape)
        #print(true_transformer.shape)
        #print(torch.matmul(transform_T, raw).shape)
        regular = torch.mean(abs((torch.matmul(transform_T, raw) - true_transformer)))

        #angle = np.arccos((np.trace(R) - 1) / 2)

        pred_center = torch.mean(pred_pc, dim=-2)
        target_center = torch.mean(target_pc, dim=-2)
        #tre = (pred_center - target_center).pow(2)
        #tre = torch.sum(tre, dim=-1)
        tre_center = absloss(pred_center, target_center)


        #emd_loss = emd(pred_pc.reshape(bs * 28, 128, 3), target_pc.reshape(bs * 28, 128, 3))
        #TRE = TRE + 0.01 * emd_loss

        #mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = tre_loss + regular# + cd_loss# + regular + cd_loss# + mat_diff_loss * self.mat_diff_loss_scale


        pre_one = pred.reshape(bs * 28, -1)
        target_one = target.reshape(bs * 28, -1)
        target_sum = torch.sum(target_one, dim=1)
        pre_one = pre_one[target_sum != 0]
        target_one = target_one[target_sum != 0]
        cos = torch.cosine_similarity(pre_one, target_one) / 2.0 + 0.5

        #pred1 = pred_pc.reshape(bs * 28, -1)
        #target1 = target_pc.reshape(bs * 28, -1)
        #pred1 = pred1[target_sum != 0]
        #target1 = target1[target_sum != 0]
        #TRE_eva = absloss(pred_pc, target_pc)

        return total_loss, mse(pred_pc, target_pc), torch.mean(abs(cos)), tre_loss, cd_loss, tre_center, regular


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = max(np.max(np.sqrt(np.sum(pc ** 2, axis=1))), 0.0000001)
    pc = pc / m
    return pc, m, centroid


def IDR_all(data, label):

    data = data.reshape(28, 128, 3)
    num, num_points, c = data.shape

    au_R = np.zeros([4, 4])
    displace = np.random.rand(3) * 0.02
    #rot = np.random.rand(3)# * 30
    r1 = Rotation.from_euler('x', random.random() * 30, degrees=False)  # 顺序和角度
    R = r1.as_matrix()

    au_R[0:3, 0:3] = R
    au_R[0:3, 3] = displace
    au_R[3, 3] = 1

    # 求解逆矩阵
    inverse_R = np.linalg.inv(au_R)

    for i in range(data.shape[0]):
        normal_data = data[i, ...]
        centroid = np.mean(normal_data, axis=0)
        centroid_mat = np.ones([4, 1])
        centroid_mat[0:3, 0] = centroid[:]
        normal_data = normal_data - centroid

        single_data = np.concatenate([normal_data, np.ones([num_points, 1])], axis=-1).transpose((1, 0))
        single_labe = label[i, ...]

        A1 = np.matmul(au_R, single_data).transpose((1, 0))
        A1[:, 0:3] = A1[:, 0:3] + centroid


        after_R = np.matmul(single_labe, inverse_R)
        T = np.matmul(single_labe, centroid_mat) - np.matmul(after_R, centroid_mat)
        after_R[0:3, 3] = after_R[0:3, 3] + T[0:3, 0]

        data[i, ...] = A1[:, 0:3]
        label[i, ...] = after_R

    return data.reshape(num * num_points, 3), label
def IDR_1(data, label):
    centroid = np.mean(data, axis=0)
    centroid_mat = np.ones([4, 1])
    centroid_mat[0:3, 0] = centroid[:]
    data = data - centroid

    data = data.reshape(28, 128, 3)
    num, num_points, c = data.shape

    au_R = np.zeros([4, 4])
    displace = np.random.rand(3) * 0.02
    rot = np.random.rand(3) * 0.02
    r1 = Rotation.from_euler('xyz', rot, degrees=False)  # 顺序和角度
    R = r1.as_matrix()

    au_R[0:3, 0:3] = R
    au_R[0:3, 3] = displace
    au_R[3, 3] = 1

    # 求解逆矩阵
    inverse_R = np.linalg.inv(au_R)

    for i in range(data.shape[0]):
        normal_data = data[i, ...]
        single_data = np.concatenate([normal_data, np.ones([num_points, 1])], axis=-1).transpose((1, 0))
        A1 = np.matmul(au_R, single_data).transpose((1, 0))
        A1[:, 0:3] = A1[:, 0:3] + centroid

        data[i, ...] = A1[:, 0:3]
    return data.reshape(num * num_points, 3), label

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
        if(random.random() < 0.2):
            point_set, label = IDR_all(point_set, label)
        if(random.random() < 0.2):
            point_set, label = IDR_1(point_set, label)

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
        point_set = point_set.reshape(28 * 128, 3)
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
        '/home/buaaa302/ISICDM-ATRC-data/starting_kit/tsing/train_bisaidata')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)

    val_dataset = ValDataLoader(
        '/home/buaaa302/ISICDM-ATRC-data/starting_kit/tsing/test_bisaidata')
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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
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

        scheduler.step()
        for batch_id, (points, target, m, centroid) in enumerate(trainDataLoader, 0):
            optimizer.zero_grad()
            m = m.cuda()
            centroid = centroid.cuda()
            points = points.data.numpy()
            #if (random.random() < 0.2):
                #points = provider.random_point_dropout(points)
            if (random.random() < 0.2):
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            if (random.random() < 0.2):
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3], shift_range=0.02)
            #if (random.random() < 0.2):
            #    points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])
            #xinde
            #points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])
            #points[:, :, 0:3] = provider.rotate_perturbation_point_cloud(points[:, :, 0:3])
            #points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])

            points = torch.Tensor(points)
            points = points.reshape(16, 28, 128, 3)
            points_perm = torch.randperm(points.shape[-2]).to(points.device)
            points = points[:, :, points_perm, :].reshape(16, 28*128, 3)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda().float()

            pred, trans_feat = model(points)
            loss, _, _, tre1, tre2, tre3, tre4 = criterion(m, centroid, points, pred, target, trans_feat)

            log_string(
                '[Epoch %d/%d] loss = %.3f loss1 = %.3f loss2 = %.3f loss3 = %.3f loss4 = %.3f' %
                (   epoch,
                    args.epoch,
                    loss.item(),
                    tre1.item(),
                    tre2.item(),
                    tre3.item(),
                    tre4.item()
                 ))


            loss.backward()
            optimizer.step()
            global_step += 1

        log_string('Train loss: %f' % loss)

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
                _, loss_MSE, loss_muti, loss_TRE, loss_CD, _, _ = criterion(m, centroid, points, pred, target,
                                                                            trans_feat)
                TRE_list.append(loss_TRE.item())
                Muti_list.append(loss_muti.item())
                CD_list.append(loss_CD.item())
                MSE_list.append(loss_MSE.item())
                print('[Epoch %d/%d] TRE = %.3f CD = %.3f MSE = %.3f Muti = %.3f ' %
                        (epoch,
                         args.epoch,
                         loss_TRE.item(),
                         loss_CD.item(),
                         loss_MSE.item(),
                         loss_muti.item(),
                         ))

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
