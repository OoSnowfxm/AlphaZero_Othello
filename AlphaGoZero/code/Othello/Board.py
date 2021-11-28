'''
    @Author: fxm
    @Date: Dec 27, 2020.
    @Title: Board class.
'''
import numpy as np

'''
    Board类，棋盘对象
    其中包含对棋盘的初始化，棋盘中不同颜色的棋子树统计
    以及一些落子相关的操作
'''
class Board():
    '''棋盘的八个方向，黑白棋八个方向上的有效落子都要考虑'''
    directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    '''
        初始化棋盘
        size：棋盘大小
        matrix：棋盘矩阵，用以存储棋盘中的数据
        初始时棋盘中已经有四个棋子
    '''
    def __init__(self, size):
        self.size = size
        self.matrix = [None] * self.size
        for i in range(self.size):
            self.matrix[i] = [0] * self.size
        
        # 棋盘的最中间有四个初始棋子
        middle = self.size // 2
        self.matrix[middle-1][middle] = -1
        self.matrix[middle-1][middle-1] = 1
        self.matrix[middle][middle-1] = -1
        self.matrix[middle][middle] = 1

    '''
        棋盘上不同颜色的棋子的个数统计
        接收参数color
        返回该color的棋子个数
    '''
    def count(self, color):
        cnt = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.matrix[x][y] == color:
                    cnt += 1
                if self.matrix[x][y] == -color:
                    cnt -= 1
        
        return cnt

    '''#################################################################################'''
    '''下面是与动作相关的函数'''

    '''
        静态方法函数，可以不使用对象而使用类名调用
        一个迭代器类型的函数
        返回以当前动作(动作也是一个位置)和方向可以得到的所有位置
    '''
    @staticmethod
    def increment_move(move, direction, n):
        move = list(map(sum, zip(move, direction)))
        while all(map(lambda x: 0 <= x < n, move)): 
            yield move
            move=list(map(sum,zip(move,direction)))

    '''
        寻找以当前位置和方向为基线的落子位置
        以当前位置为起始位置，从该方向搜索，
        找到第一个或找不到一个合法位置
        返回该位置或者None
    '''
    def searchMove(self, piece, direction):
        x, y = piece
        color = self.matrix[x][y]
        flips = []

        for x, y in Board.increment_move(piece, direction, self.size):
            # 如果位置为空，那么中间有子就可以翻转
            if self.matrix[x][y] == 0:
                if flips:
                    return (x, y)
                else:
                    return None
            # 如果位置为相同颜色棋子，那么该方向都不能落子
            elif self.matrix[x][y] == color:
                return None
            # 如果位置为异色棋子，那么可以继续搜索
            elif self.matrix[x][y] == -color:
                flips.append((x, y))

    '''
        对于某个棋子，得到与它相关的可以移动的位置
        接收参数piece
        返回列表，存储了与该棋子相关的可以移动的位置
    '''
    def getMovesOfPiece(self, piece):
        (x,y) = piece
        color = self.matrix[x][y]
        if color == 0:
            return None

        moves = []
        # 在所有方向上搜索move
        for direction in self.directions:
            move = self.searchMove(piece, direction)
            if move:
                moves.append(move)
        return moves

    '''
        获取当前颜色的棋子的合法移动列表
        接收参数color
        返回该color的棋子可以移动的位置的列表
        数据 move 类型是元组，对应棋盘上的位置
    '''
    def getLegalMoves(self, color):
        moves = set() 
        # 对棋盘上所有该颜色棋子进行上一步的操作
        for x in range(self.size):
            for y in range(self.size):
                if self.matrix[x][y] == color:
                    move = self.getMovesOfPiece((x,y))
                    moves.update(move)
        return list(moves)

    '''
        判断当前颜色的棋子是否有可以移动的位置
        接收参数color
        返回布尔类型值，是否能够移动
    '''
    def hasLegalMoves(self, color):
        moves = self.getLegalMoves(color)
        if len(moves) > 0:
            return True
        return False

    '''
        获取需要翻转的棋子
        接收一个移动位置和方向为参数
        返回该移动位置在该方向上可以翻转的所有棋子的列表
    '''
    def getFlips(self, move, direction, color):
        flips = [move]      # 初始化，只做初始化用，真正翻转也不会翻转这个位置

        # 以该移动位置为起点，并有方向的去搜索需要翻转的棋子
        for x, y in Board.increment_move(move, direction, self.size):
            # 如果是空位，那么不能翻转
            if self.matrix[x][y] == 0:
                return []
            # 如果是异色子，继续搜索
            if self.matrix[x][y] == -color:
                flips.append((x, y))
            # 如果是同色子，且中间有异色子，返回列表
            elif self.matrix[x][y] == color and len(flips) > 0:
                return flips

        return []

    '''
        执行当前移动
        接收移动位置和当前玩家为参数，注意颜色就是玩家，玩家就是颜色
        将对应位置设置为合适的颜色
    '''
    def executeMove(self, move, color):
        # 对所有方向上的翻转列表进行翻转
        flips = [flip for direction in self.directions
                      for flip in self.getFlips(move, direction, color)]
        assert len(list(flips)) > 0
        for x, y in flips:
            self.matrix[x][y] = color
         
    '''#################################################################################'''