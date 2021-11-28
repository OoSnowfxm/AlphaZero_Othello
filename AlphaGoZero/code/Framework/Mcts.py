'''
    @Author: fxm
    @Date: Dec 27, 2020.
    @Title: Mcts class.
'''

import logging
import math
import numpy as np

Eps = 1e-8
log = logging.getLogger(__name__)

# 蒙特卡洛树搜索对象
class MCTS():
    '''
        初始化过程
        参数设置：
        game:游戏对象
        net:网络对象
        args:参数
        N(s,a):记录边的访问次数
　　　　 S(s):  记录该状态的访问次数，有S(s) = sum(N(s,i))
　　　　 Q(s,a) :平均行动价值
　　　　 P(s,a) :选择该条边的先验概率
        Ended(s):存储状态s是否对应了游戏结束
        Valid(s):存储状态s对应的所有的可行动作
    '''
    def __init__(self, game, net, args):
        self.game = game
        self.net = net
        self.args = args
        self.Q = {}          
        self.N = {}      
        self.S = {}        
        self.P = {}          
        self.Ended = {}            
        self.Valid = {}            
    
    '''
        获得当前棋盘得到的动作的概率向量
        在temp为0的时候，说明网络深度已经很深，
        这时候采用将最大概率设为1来计算概率向量
    '''
    def getActionProb(self, canonicalBoard, temp=1):
        for _ in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        # 获得棋盘的字符串解释
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.N[(s, a)] if (s, a) in self.N else 0 \
            for a in range(self.game.getActionSize())]

        # 如果temp = 0，我们期望获取准确的动作，也就是测试阶段和深度过深的训练阶段
        if temp == 0:
            idx = np.array(np.argwhere(counts == np.max(counts))).flatten()
            idx = np.random.choice(idx)
            probs = [0] * len(counts)
            probs[idx] = 1
            return probs

        # 如果temp不为0，我们期望获取动作的概率向量，也就是深度不深的训练阶段
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    '''
        蒙特卡洛搜索树过程
        接收当前玩家看到的棋盘为参数
        主要作用为更新Q表的值
    '''
    def search(self, canonicalBoard):
        # 获得当前棋盘的字符串解释，注意是玩家所看到的棋盘
        s = self.game.stringRepresentation(canonicalBoard)
        # 如果当前状态不在结束判别列表内，就加入到列表中
        if s not in self.Ended:
            self.Ended[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Ended[s] != -2:
            return -self.Ended[s]
        
        # 如果策略列表中没有，就使用深度网络预测的值
        if s not in self.P:
            self.P[s], v = self.net.predict(canonicalBoard)
            valids = self.game.getValid(canonicalBoard, 1)
            self.P[s] = self.P[s] * valids      
            sump = np.sum(self.P[s])
            # 将结果修正到0——1之间
            if sump > 0:
                self.P[s] /= sump  
            # 如果神经网络预测的结果有问题，那么直接使用valid作为当前状态下的策略
            # 这个时候每一个合法的动作都拥有相同的概率
            else:  
                log.error("All valid moves were masked, doing a workaround.")
                self.P[s] = self.P[s] + valids
                self.P[s] /= np.sum(self.P[s])

            self.Valid[s] = valids
            self.S[s] = 0
            return -v

        # 在当前状态下根据Q表选择最佳的动作
        valids = self.Valid[s]
        best = -float('inf')
        best_action = -1

        # 从所有合法动作中选择出UCT值最大的一个作为当前状态的下一个动作
        for a in range(self.game.getActionSize()):
            if valids[a]:
                # 如果Q中已经有这一项
                if (s, a) in self.Q:
                    u = self.Q[(s, a)] + self.args.cpuct * self.P[s][a] * \
                        math.sqrt(self.S[s]) / (1 + self.N[(s, a)])
                # 如果没有
                else:
                    u = self.args.cpuct * self.P[s][a] * math.sqrt(self.S[s] + Eps)  
                # 更新当前的最优动作
                if u > best:
                    best = u
                    best_action = a

        # 获取下一个动作
        a = best_action
        next_state, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_state = self.game.getCanonicalForm(next_state, next_player)

        # 递归实现搜索过程，本质上是一个回溯过程
        v = self.search(next_state)

        # 这些是蒙特卡洛树搜索的反向传播过程，也是递归的回溯部分
        # 更新Q，原来的加上新的值
        if (s, a) in self.Q:
            self.Q[(s, a)] = (self.N[(s, a)] * self.Q[(s, a)] + v * 1) / (self.N[(s, a)] + 1)
            self.N[(s, a)] += 1
        else:
            self.Q[(s, a)] = v
            self.N[(s, a)] = 1

        self.S[s] += 1
        return -v
