'''
    @Author: fxm
    @Date: Dec 27, 2020.
    @Title: Game class.
'''
from __future__ import print_function
from .Board import Board
import numpy as np


'''
    Game类：游戏对象
    包含了一些和游戏相关的函数实现
    也是对棋盘类的进一步描述
'''
class Game(object):

    '''初始化'''
    def __init__(self, size):
        self.size = size  
    
    '''初始化棋盘'''
    def initBoard(self):
        b = Board(self.size)
        return np.array(b.matrix)
    
    '''获得棋盘大小'''
    def getBoardSize(self):
        return (self.size, self.size)
    
    '''获取动作action类型的数据的大小'''
    def getActionSize(self):
        return self.size * self.size + 1

    '''
        获取下一个状态
        状态的描述使用（棋盘，当前执棋方）表示
        在模型设计中，不同的执棋方看到的棋盘是完全相反的
        方便蒙特卡洛树搜索的设计
    '''
    def getNextState(self, board, color, action):
        if action == self.size * self.size:
            return (board, -color)
        b = Board(self.size)
        b.matrix = np.copy(board)
        # action是0——65大小的数据，move是元组，通过变换建立起两者的关系
        move = (int(action/ self.size), action % self.size)
        b.executeMove(move, color)
        return (b.matrix, -color)
    
    '''
        获取一个valid数组
        其index为动作action，范围在0——64内
        其含义为该动作是否是有效的
    '''
    def getValid(self, board, color):
        # 大小为Action Size
        valids = [0] * self.getActionSize()
        b = Board(self.size)
        b.matrix = np.copy(board)
        legalMoves =  b.getLegalMoves(color)
        
        # 如果没有可执行的动作，说明不能执行动作
        # valid的最后一项没有意义,但是要求它永远是有效的
        # 即没有合法动作就对应这一项
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        
        # 对于可执行动作中的每一项，
        # 找到在valid中对应的位置
        for x, y in legalMoves:
            valids[self.size * x + y] = 1 
        return np.array(valids)
    
    '''
        判断当前是否可以执行动作
        如果不能继续执行动作，需要交换掌棋权，由对方连下，返回True
        反之返回False
    '''
    def getNoAction(self, board, color):
        valids = self.getValid(board, color)
        for i in range(len(valids)):
            if valids[i] == 1 and i != self.size ** 2:
                return False
        return True

    '''
        判断游戏是否结束
        返回结束一方是否获胜
        获胜返回1，落败返回-1
        未结束返回-2，平局返回0
    '''
    def getGameEnded(self, board, color):
        b = Board(self.size)
        b.matrix = np.copy(board)
        
        # 如果双方有一方可以走，游戏就没有结束，返回0
        if b.hasLegalMoves(color):
            return -2
        if b.hasLegalMoves(-color):
            return -2
        # 如果color的颜色更多，那么返回1，否则返回-1
        if b.count(color) > 0:
            return 1
        elif b.count(color) == 0:
            return 0
        else:
            return -1
    
    '''
        获得当前玩家所看到的棋盘状态
        玩家总是把自己的棋子看成1，对方的棋子看作-1
        不代表棋盘的真实状态
    '''
    def getCanonicalForm(self, board, color):
        return color * board
    
    '''
        对当前棋盘做对称性变换
        对称性强的几个棋盘所包含的信息可以直接获取的
        不需要通过自博弈多次学习
    '''
    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.size * self.size + 1)  
        pi_board = np.reshape(pi[:-1], (self.size, self.size))
        List = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                List += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return List
    
    '''
        棋盘的字符串解释
        在深度学习训练中，传入棋盘会造成不必要的维度增长
        使用字符串代表不同的棋盘状态
    '''
    def stringRepresentation(self, board):
        return board.tostring()
    
    '''
        获得玩家在当前棋盘的得分情况
        使用的统计得分函数为黑白棋子个数差
    '''
    def getScore(self, board, color):
        b = Board(self.size)
        b.matrix = np.copy(board)
        return b.count(color)