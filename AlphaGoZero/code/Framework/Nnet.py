'''
    @Author: fxm
    @Date: Dec 27, 2020.
    @Title: Nnet class.
'''

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm


'''
    残差块结构
    深度学习残差网络的基本结构
'''
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

'''
    深度神经网络模型类：网络对象
    只包含网络结构，网络训练、预测在Net.py中
'''
class NNet(nn.Module):
    
    '''
        初始化网络
        网络参数：
        board_x、board_y:棋盘大小
        action_num:动作最多数量
        args：参数选择，通过训练网络传入
    '''
    def __init__(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # 网络卷积层、残差块设置
        super(NNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.resblock1 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock2 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock3 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock4 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock5 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock6 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock7 = ResidualBlock(args.num_channels, args.num_channels, stride=1)

        # 网络BN层设置
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)

        # 网络全连接层及其BN层设置
        self.fc1 = nn.Linear(args.num_channels * (self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    
    '''前向传播过程，代码后的注释为输入的大小（输出后的大小）'''
    def forward(self, s):
                                                                    # batch_size * board_x * board_y
        s = s.view(-1, 1, self.board_x, self.board_y)               # batch_size * 1 * board_x * board_y
        s = F.relu(self.bn1(self.conv1(s)))                         # batch_size * num_channels * board_x * board_y 
        s = self.resblock1(s)
        s = self.resblock2(s)
        s = self.resblock3(s) 
        s = self.resblock4(s) 
        s = self.resblock5(s)
        s = self.resblock6(s)
        s = self.resblock7(s)
        s = F.relu(self.bn2(self.conv2(s)))                         # batch_size * num_channels * (board_x-2) * (board_y-2)
        s = F.relu(self.bn3(self.conv3(s)))                         # batch_size * num_channels * (board_x-4) * (board_y-4)
        s = s.view(-1, self.args.num_channels*(self.board_x-4) * (self.board_y-4))

        # 使用dropout层来增强网络效果
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p= self.args.dropout, training= self.training)  # batch_size * 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p= self.args.dropout, training= self.training)  # batch_size * 512

        # 损失,分别对应action概率向量(策略)和分数(值)
        p = self.fc3(s)                                                                          # batch_size * action_size
        v = self.fc4(s)                                                                          # batch_size * 1

        return F.log_softmax(p, dim=1), torch.tanh(v)
    
    '''计算损失'''
    def lossP(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def lossV(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]