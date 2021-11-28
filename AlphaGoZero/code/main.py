'''
    @Author: fxm
    @Date: Dec 27, 2020.
    @Title: main FILE.
'''

from Framework.Mcts import MCTS
from Othello.Game import Game
from Framework.Net import NET, dotdict
from UI.UI import UI, pygame
import numpy as np


if __name__ == "__main__":
    g = Game(8)
    n = NET(g)
    n.loadCheckpoint('Model/','best1.pth.tar')
    args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts = MCTS(g, n, args)
    ai = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
    GAME = UI(g)

    pygame.init() #初始化游戏
    screen = pygame.display.set_mode((GAME.screen_w, GAME.screen_h)) 
    pygame.display.set_caption('黑白棋') 
    GAME.display(screen, ai, False)