import time
import math
import re
import sys
import os
import argparse

import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils import model_zoo
import torchvision
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
# matplotlib.use('TkAgg')

from model import SixDRepNet, SixDRepNet2
import utils
import datasets
from loss import GeodesicLoss


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=80, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=256, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.0001, type=float)
    parser.add_argument('--scheduler', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='Pose_300W_LP', type=str) #Pose_300W_LP
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--other_information', dest='other_information', help='Other information for marking.',
        default='MSELoss', type=str)

    args = parser.parse_args()
    return args

# 将快照中的参数加载到模型中，这样可以方便地从某个快照处重新开始训练
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    # 将快照中的参数更新到model_dict中
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':

    args = parse_args()
    # cudnn.enabled 设置为 True 意味着在处理图像数据时，
    # 会使用 CuDNN 进行加速，以提高模型的处理速度。
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    summary_name = '{}_{}_bs{}_{}_{}'.format(
        'SixDRepNet', int(time.time()), args.batch_size, args.dataset, args.other_information)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))

    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='RepVGG-B1g2-train.pth',
                        deploy=False,
                        pretrained=True)
 
    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')

    # 图像归一化，对图像数据进行标准化的一种方法
    # 不同的数据集图像的均值和标准差不一定一样，
    # 所以我在尝试不同的数据集的时候可能要先计算一下这两个值
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    # 对图像进行一系列处理，最终输出224x224大小的图像
    transformations = transforms.Compose([transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
                                          transforms.ToTensor(),
                                          normalize])

    # 加载的数据库最终会通过transformations进行预处理
    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)

    # 创建一个加载器，这里的num_workers指的是线程数
    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    # 尝试将模型和数据加载到GPU上，如果此处的gpu为False则在CPU上跑
    model.cuda(gpu)
    # FIXME：我自己做的第一个尝试，用余弦损失函数来做
    # 这种尝试失败了
    # crit = torch.nn.CosineEmbeddingLoss().cuda(gpu)
    crit = torch.nn.MSELoss().cuda(gpu) #GeodesicLoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)


    if not args.snapshot == '':
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])

    #milestones = np.arange(num_epochs)
    milestones = [10, 20]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    print('Starting training.')
    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0
        for i, (images, gt_mat, _, _) in enumerate(train_loader):
            iter += 1
            # 将数据转换为张量并转移到GPU
            images = torch.Tensor(images).cuda(gpu)

            # Forward pass，用backbone提取出特征
            # 这里的model会自动调用forward函数对images进行处理
            pred_mat = model(images)

            # Calc loss，计算损失函数
            loss = crit(gt_mat.cuda(gpu), pred_mat)

            # 以下三行代码是反向传播步骤中的一个整体
            # 在反向传播之前，需要先将梯度清零，以防受到前一次迭代的影响。
            optimizer.zero_grad()
            # PyTorch 的反向传播函数，它将根据当前的损失函数计算梯度。
            loss.backward()
            # 使用梯度更新模型参数。具体而言，
            # 这个函数将使用优化器的方法（例如随机梯度下降，Adam 等），
            # 根据当前的梯度更新模型的参数。
            optimizer.step()

            loss_sum += loss.item()

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: '
                      '%.6f' % (
                          epoch+1,
                          num_epochs,
                          i+1,
                          len(pose_dataset)//batch_size,
                          loss.item(),
                      )
                      )
        
        # 根据前面的配置决定是否调整学习率
        if b_scheduler:
            scheduler.step()

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...',
                # 下面的是原有代码，我改成更下面的这种直接保存模型的了
                # 改进后得到的结果准确度更高
                #   torch.save({
                #       'epoch': epoch,
                #       'model_state_dict': model.state_dict(),
                #       'optimizer_state_dict': optimizer.state_dict(),
                #   }, 'output/snapshots/' + summary_name + '/' + args.output_string +
                #       '_epoch_' + str(epoch+1) + '.tar')
                  torch.save(model
                  , 'output/snapshots/' + summary_name + '/' + args.output_string +
                      '_epoch_' + str(epoch+1) + '.pth')
                  )
