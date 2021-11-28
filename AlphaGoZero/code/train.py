'''
    @Author: fxm
    @Date: Dec 27, 2020.
    @Title: train FILE.
'''

import logging
import coloredlogs
from Othello.Game import Game
from Othello.Board import Board
from Framework.Net import NET 
from Learn import Learn, args

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.
LOAD_FILE = False

'''
    训练过程
    通过不断的学习提升棋力
'''
def TRAIN():
    log.info('Loading %s...', Game.__name__)
    g = Game(8)
    log.info('Loading %s...', NET.__name__)
    net = NET(g)
    log.info('Loading the Learn Process...')
    c = Learn(g, net, args)

    if LOAD_FILE:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process')
    c.learn()    

if __name__ == "__main__":
    TRAIN()