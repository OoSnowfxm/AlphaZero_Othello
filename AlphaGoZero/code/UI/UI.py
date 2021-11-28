'''
    @Author: fxm
    @Date: Dec 27, 2020.
    @Title: UI class.
'''
import sys
sys.path.append('..')
import pygame
import logging
from pygame.locals import *
import pygame.gfxdraw
from collections import namedtuple
from Framework.Net import dotdict
from Othello.Board import Board

log = logging.getLogger(__name__)

'''
    参数设置
'''
SIZE = 30                                                                       # 棋盘每个点之间的间隔
Line_Points = 9                                                                 # 棋盘每行/每列点数
Outer_Width = 20                                                                # 棋盘外宽度
Border_Width = 4                                                                # 边框宽度
Inside_Width = 4                                                                # 边框跟实际的棋盘之间的间隔
Border_Length = SIZE * (Line_Points - 1) + Inside_Width * 2 + Border_Width      # 边框线的长度
Start_X = Outer_Width + int(Border_Width / 2) + Inside_Width                    # 起始点X坐标
Start_Y = Outer_Width + int(Border_Width / 2) + Inside_Width                    # 起始点Y坐标
SCREEN_HEIGHT = SIZE * (Line_Points - 1) + Outer_Width \
    * 2 + Border_Width + Inside_Width * 2                                       # 游戏屏幕的高
SCREEN_WIDTH = SCREEN_HEIGHT + 200                                              # 游戏屏幕的宽
Stone_Radius = SIZE // 2                                                        # 棋子半径
Checkerboard_Color = (0xE3, 0x92, 0x65)                                         # 棋盘颜色
BLACK_COLOR = (0, 0, 0)                                                         # 黑色
WHITE_COLOR = (255, 255, 255)                                                   # 白色
RED_COLOR = (245, 222, 179)                                                     # 淡黄色
BLUE_COLOR = (30, 30, 200)                                                      # 蓝色
RIGHT_INFO_POS_X = SCREEN_HEIGHT + Stone_Radius * 2 + 10

'''
    UI类，游戏界面设置对象
'''
class UI():
    '''
        初始化
        参数设置：
        game：游戏对象
        screen_h：游戏屏幕高
        screen_w：游戏屏幕宽
    '''
    def __init__(self, game):
        self.game = game
        self.screen_h = SCREEN_HEIGHT
        self.screen_w = SCREEN_WIDTH

    '''输出一段文字信息'''
    def printText(self, screen, font, x, y, text, fcolor=(255, 255, 255)):
        imgText = font.render(text, True, fcolor)
        screen.blit(imgText, (x, y))


    '''画棋盘'''
    def drawCheckerboard(self, screen):
        # 填充棋盘背景色
        screen.fill(Checkerboard_Color)
        # 画棋盘网格线外的边框
        pygame.draw.rect(screen, BLACK_COLOR, (Outer_Width, Outer_Width, \
            Border_Length, Border_Length), Border_Width)
        # 画网格线
        for i in range(Line_Points): #竖线
            pygame.draw.line(screen, BLACK_COLOR, (Start_Y, Start_Y + SIZE * i), \
                (Start_Y + SIZE * (Line_Points - 1), Start_Y + SIZE * i), 1)
        for j in range(Line_Points): #横线
            pygame.draw.line(screen, BLACK_COLOR, (Start_X + SIZE * j, Start_X), \
                (Start_X + SIZE * j, Start_X + SIZE * (Line_Points - 1)), 1)

    '''画棋子'''
    def drawChessman(self, screen, point, stone_color):
        pygame.gfxdraw.aacircle(screen, Start_X + SIZE * point[0] + SIZE // 2, \
            Start_Y + SIZE * point[1] + SIZE // 2, Stone_Radius, stone_color)
        pygame.gfxdraw.filled_circle(screen, Start_X + SIZE * point[0] + SIZE // 2, \
            Start_Y + SIZE * point[1] + SIZE // 2, Stone_Radius, stone_color)

    '''画一个单独的不在棋盘内的棋子'''
    def drawChessmanPos(self, screen, pos, stone_color):
        pygame.gfxdraw.aacircle(screen, pos[0], pos[1], Stone_Radius, stone_color)
        pygame.gfxdraw.filled_circle(screen, pos[0], pos[1], Stone_Radius, stone_color)
    
    '''画提示可以走的棋子位置'''
    def drawMoves(self, screen, point, stone_color):
        pygame.gfxdraw.aacircle(screen, Start_X + SIZE * point[0] + SIZE // 2, \
            Start_Y + SIZE * point[1] + SIZE // 2, Stone_Radius // 3, stone_color)
        pygame.gfxdraw.filled_circle(screen, Start_X + SIZE * point[0] + SIZE // 2, \
            Start_Y + SIZE * point[1] + SIZE // 2, Stone_Radius // 3, stone_color)
    
    '''画右侧信息显示'''
    def drawRightInfo(self, screen, font, moves, curplayer, is_human_first):
        self.drawChessmanPos(screen, (SCREEN_HEIGHT + Stone_Radius, Start_X + Stone_Radius), BLACK_COLOR)
        self.drawChessmanPos(screen, (SCREEN_HEIGHT + Stone_Radius, Start_X + Stone_Radius * 4), WHITE_COLOR)
        if is_human_first == True:
            self.printText(screen, font, RIGHT_INFO_POS_X, Start_X + 3, '玩家', BLUE_COLOR)
            self.printText(screen, font, RIGHT_INFO_POS_X, Start_X + Stone_Radius * 3 + 3, '电脑', BLUE_COLOR)
        else:
            self.printText(screen, font, RIGHT_INFO_POS_X, Start_X + 3, '电脑', BLUE_COLOR)
            self.printText(screen, font, RIGHT_INFO_POS_X, Start_X + Stone_Radius * 3 + 3, '玩家', BLUE_COLOR)
        if curplayer == -1:
            self.printText(screen, font, SCREEN_HEIGHT, SCREEN_HEIGHT//2, f'当前出棋：黑棋', BLUE_COLOR)
        else:
            self.printText(screen, font, SCREEN_HEIGHT, SCREEN_HEIGHT//2, f'当前出棋：白棋', BLUE_COLOR)
        
        

    '''根据鼠标点击位置，返回游戏区坐标'''
    def getClickpoint(self, click_pos):
        pos_x = click_pos[0] - Start_X
        pos_y = click_pos[1] - Start_Y
        # 如果鼠标点击范围不在游戏区内，就返回None
        if pos_x < -Inside_Width or pos_y < -Inside_Width:
            return None
        
        x = int(pos_x  / SIZE) 
        y = int(pos_y  / SIZE)
        # 如果鼠标点击范围超过棋盘另一侧长度，也返回None
        if x >= Line_Points or y >= Line_Points:
            return None

        return (x, y)
    
    '''画出棋盘的所有信息'''
    def drawAll(self, screen, board, end, moves, curplayer, is_human_first):
        # 画棋盘
        font1 = pygame.font.SysFont('SimHei', 72)
        font2 = pygame.font.SysFont('SimHei', 24)
        fwidth, fheight = font1.size('黑方获胜')
        self.drawCheckerboard(screen)

        self.drawRightInfo(screen, font2, moves, curplayer, is_human_first)
        # 画棋盘上已有的棋子
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 1:
                    self.drawChessman(screen, (j,i), WHITE_COLOR)
                elif board[i][j] == -1:
                    self.drawChessman(screen, (j,i), BLACK_COLOR)
        
        if (is_human_first == True and curplayer == -1) or \
            (is_human_first == False and curplayer == 1):
            for move in moves:
                x = move[0]
                y = move[1]
                self.drawMoves(screen, (y,x), RED_COLOR)

        if end != -2:
            if end == 1:
                self.printText(screen, font1, (SCREEN_WIDTH - fwidth)//2, (SCREEN_HEIGHT - fheight)//2, '白子获胜', RED_COLOR)
            elif end == -1:
                self.printText(screen, font1, (SCREEN_WIDTH - fwidth)//2, (SCREEN_HEIGHT - fheight)//2, '黑子获胜', RED_COLOR)
            elif end == 0:
                self.printText(screen, font1, (SCREEN_WIDTH - fwidth)//2, (SCREEN_HEIGHT - fheight)//2, '平局', RED_COLOR)
        pygame.display.flip()   

    '''人类走棋'''
    def humanplay(self, board):
        valid = self.game.getValid(board, 1)
        while True:
            for event in pygame.event.get(): 
                if event.type == MOUSEBUTTONDOWN:               # 鼠标有输入，则为落子 
                    pressed_array = pygame.mouse.get_pressed()
                    if pressed_array[0]:
                        mouse_pos = pygame.mouse.get_pos()
                        point = self.getClickpoint(mouse_pos)
                        y,x = point[0], point[1]
                        if ((0 <= x) and (x < self.game.size) and (0 <= y) and (y < self.game.size)) or \
                            ((x == self.game.size) and (y == 0)):
                            a = self.game.size * x + y if x != -1 else self.game.size ** 2
                            if valid[a]:
                                return a
                            else:continue
                        else:continue

    '''游戏仿真主过程'''
    def display(self, screen, ai, is_human_first):
        if is_human_first:
            players = ['human', None, 'ai']
        else:
            players = ['ai', None, 'human']

        curplayer = -1
        board = self.game.initBoard()
        
        while True:
            b = Board(self.game.size)
            b.matrix = board
            moves = b.getLegalMoves(curplayer)
            end = self.game.getGameEnded(board, curplayer)
            self.drawAll(screen, board, curplayer * end, moves, curplayer, is_human_first)
            if end != -2:
                continue
            
            # 如果当前无路可走，就让对方连走两步
            if self.game.getNoAction(board, curplayer) == True:
                # 更改状态，交换执棋者
                board, curplayer = self.game.getNextState(board, curplayer, self.game.size **2)
                continue
                    
            if players[curplayer+1] == 'human':
                action = self.humanplay(self.game.getCanonicalForm(board, curplayer))
            elif players[curplayer+1] == 'ai':
                action = ai(self.game.getCanonicalForm(board, curplayer))
            
            valids = self.game.getValid(self.game.getCanonicalForm(board, curplayer), 1)

            # 如果动作不在合法动作列表内，返回错误
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            # 更改状态，交换执棋者
            board, curplayer = self.game.getNextState(board, curplayer, action)