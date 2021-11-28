'''
    @Author: fxm
    @Date: Dec 27, 2020.
    @Title: Net class.
'''

import os
import time
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from .Nnet import NNet

'''
    参数设置
'''
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

'''
    NET类：网络对象
    不包含网络的结构，只包含训练、预测、加载、保存
    网络等操作
'''
class NET():
    '''
        初始化部分
        参数设置：
        net：网络结构对象
        board_x、board_y：棋盘大小
        action_size：动作大小
    '''
    def __init__(self, game):
        self.net = NNet(game, args)
        if args.cuda:
            self.net = self.net.cuda()
        self.board_x, self.board_y = game.getBoardSize()

        if args.cuda:
            self.net.cuda()

    '''
        训练过程
        通过传入的棋盘、action概率向量和值
        policy net和value net通过参数共享的方式
        对结果进行不断的修正，降低损失
        得到最好的网络模型
        这个网络是AC网络的综合模型
    '''
    def train(self, data):
        # 使用Adam梯度下降方法
        optimzer = optim.Adam(self.net.parameters(), lr=args.lr)

        total_num = 0
        total_loss_p = 0
        total_loss_v = 0
        for epoch in range(args.epochs):
            print('EPOCH::', str(epoch+1))
            # 使用BN层和dropout层
            self.net.train()

            batch_num = int(len(data) / args.batch_size)
            t = tqdm(range(batch_num), desc='Training Net')

            for _ in t:
                # 随机选择batch_size数目个样例进行训练
                ids = np.random.randint(len(data), size= args.batch_size)
                # 类似解包过程，将data的每一项分离开来
                boards, ps, vs = list(zip(*[data[i] for i in ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                ps = torch.FloatTensor(np.array(ps))
                vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if args.cuda:
                    boards, ps, vs = boards.contiguous().cuda(), ps.contiguous().cuda(), vs.contiguous().cuda()
                
                # 计算网络输出
                out_p, out_v = self.net(boards)
                loss_p = self.net.lossP(ps, out_p)
                loss_v = self.net.lossV(vs, out_v)

                # 两种损失对应的是均方误差损失和交叉熵损失
                loss = loss_p + loss_v

                # 输出损失部分
                total_loss_p += loss_p * boards.size(0)
                total_loss_v += loss_v * boards.size(0)
                total_num += boards.size(0)
                t.set_postfix(Loss_p= total_loss_p/total_num, Loss_v= total_loss_v/total_num)

                # 反向传播部分
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()
    
    '''
        预测过程
        通过传入的棋盘(即状态)
        来预测其对应的action概率向量和值
    '''
    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: 
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.net.eval()
        with torch.no_grad():
            p, v = self.net(board)

        return torch.exp(p).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
    '''保存训练好的网络模型'''
    def saveCheckpoint(self, folder= 'checkpoint', filename= 'checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("模型目录不存在! 创建目录 {}".format(folder))
            os.mkdir(folder)
        else:
            'sdsdsdsdsdsd'
        torch.save({
            'state_dict': self.net.state_dict(),
        }, filepath)

    '''加载训练好的网络模型'''
    def loadCheckpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("没有模型在路径： {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location= map_location)
        self.net.load_state_dict(checkpoint['state_dict'])