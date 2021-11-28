'''
    @Author: fxm
    @Date: Dec 27, 2020.
    @Title: SelfPlay class.
'''

import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

'''
    SelfPlay类
    自博弈过程
'''
class SelfPlay():
    '''
        初始化
        参数设置：
        net1, net2: 两个智能体玩家，接收棋盘为输入并输出动作
        game：游戏对象
        display：布尔值，是否输出棋盘
    '''
    def __init__(self, net1, net2, game):
        self.net1 = net1
        self.net2 = net2
        self.game = game
    
    '''
        单局游戏的自博弈过程
    '''
    def playGame(self):
        # 玩家列表
        nets = [self.net2, None, self.net1]
        # 谁先手无所谓
        curcolor = 1
        board = self.game.initBoard()
        cnt = 1
        # 只要游戏不结束，两个网络就一直博弈
        while self.game.getGameEnded(board, curcolor) == -2:
            cnt += 1
            # 这么做的目的是，当网络1在玩游戏时，下一动作为列表中第2项玩家1做出
            # 当网络2在玩游戏时，下一动作为列表中第0项玩家2做出
            action = nets[curcolor + 1](self.game.getCanonicalForm(board, curcolor))
            valids = self.game.getValid(self.game.getCanonicalForm(board, curcolor), 1)

            # 如果动作不在合法动作列表内，返回错误
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            
            # 更改状态，交换执棋者
            board, curcolor = self.game.getNextState(board, curcolor, action)    
        # 输出最后赢棋的人
        return curcolor * self.game.getGameEnded(board, curcolor)
    
    '''
        多局游戏的自博弈过程，
        其中两个网络分别先手一半的游戏
    '''
    def playGames(self, num):
        num = int(num / 2)
        win1 = 0
        win2 = 0
        draws = 0
        # 玩家1先手
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame()
            if gameResult == 1:
                win1 += 1
            elif gameResult == -1:
                win2 += 1
            else:
                draws += 1

        # 交换先手
        self.net1, self.net2 = self.net2, self.net1

        # 玩家2先手
        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame()
            if gameResult == -1:
                win1 += 1
            elif gameResult == 1:
                win2 += 1
            else:
                draws += 1

        return win1, win2, draws