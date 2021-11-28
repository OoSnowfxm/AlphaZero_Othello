'''
    @Author: fxm
    @Date: Dec 27, 2020.
    @Title: test FILE.
'''

from Framework.Mcts import MCTS
from Othello.Game import Game
from Framework.Net import NET, dotdict
from UI.UI import UI, pygame
from SelfPlay import SelfPlay
import numpy as np

if __name__ == "__main__":
    g = Game(8)
    n1 = NET(g)
    n1.loadCheckpoint('Model/','best1.pth.tar')
    n2 = NET(g)
    n2.loadCheckpoint('Model/','best.pth.tar')
    args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args)
    mcts2 = MCTS(g, n2, args)
    ai1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    ai2 = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    arena = SelfPlay(ai1, ai2, g)
    win1, win2, draws = arena.playGames(10)
    print(win1, win2, draws)