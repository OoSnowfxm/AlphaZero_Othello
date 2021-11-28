'''
    @Author: fxm
    @Date: Dec 27, 2020.
    @Title: Learn class.
'''

import logging
import os
import numpy as np
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from tqdm import tqdm
from SelfPlay import SelfPlay
from Framework.Mcts import MCTS
from Framework.Net import dotdict

log = logging.getLogger(__name__)

args = dotdict({
    'numIters': 100,              # 神经网络训练迭代次数
    'numEps': 100,                # 自学习过程每次比赛场数
    'tempThreshold': 15,        # 搜索树的阈值
    'updateThreshold': 0.5,     # 如果新模型赢了超过updateThreshold比例的比赛，就接受新模型
    'maxlenOfQueue': 200000,    # 训练数据上限
    'numMCTSSims': 25,          # 搜索树采样数量
    'arenaCompare': 2,         # 新旧模型对比需要的比赛次数
    'cpuct': 1,                 # 搜索树参数
    'checkpoint': './Model/',   # 模型保存路径
    'load_model': False,        # 是否加载模型
    'numItersFortable': 20,
})

class Learn():
    '''
        初始化
        参数设置:
        game：游戏对象
        net：网络对象
        pnet：竞争对手的网络对象
        mcts：蒙特卡洛树对象
        args：其他参数
        table：保存的训练数据，每次训练都使用这里面的数据
    '''
    def __init__(self, game, net, args):
        self.game = game
        self.net = net
        self.pnet = self.net.__class__(self.game)  
        self.args = args
        self.mcts = MCTS(self.game, self.net, self.args)
        self.table = []  
        self.firstrain = True
    '''
        process过程算法
        将蒙特卡洛树搜索得到的
        棋盘(即状态)、得到的action概率向量、值
        传入神经网络进行学习
    '''
    def process(self):
        # 结果保存表
        table = []
        board = self.game.initBoard()
        # 由于是学习过程，谁先手无所谓
        self.curcolor = 1
        step = 0

        # 只要比赛不结束就一直进行
        while True:
            step += 1
            # 当前玩家所使用棋盘对象
            canonicalBoard = self.game.getCanonicalForm(board, self.curcolor)
            # 如果step大于阈值，则不再计算准确的概率向量，而是将概率最大的设为1，其他为0
            temp = int(step < self.args.tempThreshold)

            # 获得当前的概率向量
            prob = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, prob)
            for b, p in sym:
                table.append([b, self.curcolor, p, None])
            
            # 执行动作后获得下一状态，并交换掌棋权，然后继续执行上述过程
            action = np.random.choice(len(prob), p=prob)
            board, self.curcolor = self.game.getNextState(board, self.curcolor, action)
            ended = self.game.getGameEnded(board, self.curcolor)

            # 如果结束就返回结果，是一个三元组
            # 最终的结果包括了搜索轨迹中所有的状态对应的
            # 状态本身，动作action概率向量，以及最有希望得到的值
            if ended != -2:
                return [(x[0], x[2], ended * ((-1) ** (x[1] != self.curcolor))) for x in table]
    
    '''
        学习过程
        AC模块采用神经网络学习
        process模块采用蒙特卡洛树搜索过程
    '''
    def learn(self):
        for i in range(1, self.args.numIters+1):
            log.info(f'Starting Iter #{i} ...')
            deq = deque([], maxlen=self.args.maxlenOfQueue)

            # 在自学习过程中，执行process算法过程
            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(self.game, self.net, self.args)  
                deq += self.process()

            # 将训练结果保存到table中
            self.table.append(deq)

            if len(self.table) > self.args.numItersFortable:
                log.warning(f"Removing the oldest entry in data. len(table) = {len(self.table)}")
                self.table.pop(0)

            # 训练前需要将数据打乱
            data = []
            for e in self.table:
                data.extend(e)
            shuffle(data)


            # 训练新的网络前需要保存旧的网络
            self.net.saveCheckpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.loadCheckpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.net.train(data)
            mcts = MCTS(self.game, self.net, self.args)

            # 如果是第一次训练，那么不需要比较网络
            if self.firstrain == True:
                self.net.saveCheckpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                self.net.saveCheckpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                self.firstrain = False
                continue

            log.info('与旧的网络对比中：')
            arena = SelfPlay(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                        lambda x: np.argmax(mcts.getActionProb(x, temp=0)), self.game)
            pwins, wins, draws = arena.playGames(self.args.arenaCompare)

            log.info('胜/负 : %d / %d ; 平局 : %d' % (wins, pwins, draws))
            if pwins + wins == 0 or float(wins) / (pwins + wins) <= self.args.updateThreshold:
                log.info('不接受新的模型')
                self.net.loadCheckpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('接受新的模型')
                self.net.saveCheckpoint(folder=self.args.checkpoint, filename='best.pth.tar')