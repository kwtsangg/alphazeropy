#!/usr/bin/env python
__file__       = "gui_pygame.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2019"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2019 Feb 28"

Description=""" A simple gui to play the board game.
"""

#================================================================
# Module
#================================================================
import os, sys
import numpy as np
import pygame
from pygame.locals import *

#================================================================
# Main
#================================================================
# Basic parameters
Color_dict = {}
Color_dict["white"] = (255,255,255)
Color_dict["black"] = (0,0,0)
Color_dict["brown"] = (160,82,45)

class Board_gui:
  def __init__(self,
          Ncol = 7,
          Nrow = 6,
          FPS = 30,
          SCREENX = 512,
          SCREENY = 512,
          XGAP = [0,0],
          YGAP = [100,0],
          Color_screen = Color_dict["brown"],
          Color_line = Color_dict["black"],
          ):
    self.Ncol = Ncol
    self.Nrow = Nrow
    self.FPS = FPS
    self.SCREENX = SCREENX
    self.SCREENY = SCREENY
    self.XGAP = XGAP
    self.YGAP = YGAP
    self.Color_screen = Color_screen
    self.Color_line = Color_line

    pygame.init()
    pygame.display.set_caption('My AlphaZeroPy Platform')
    self.FPSCLOCK = pygame.time.Clock()
    self.reset()

    self.draw_line()
    while True:
      for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
          pygame.quit()
          sys.exit()
        #elif event.type==VIDEORESIZE:
        #  self.SCREENX = event.dict['size'][0] - self.XGAP[0] - self.XGAP[1]
        #  self.SCREENY = event.dict['size'][1] - self.YGAP[0] - self.YGAP[1]
        #  self.reset()
        #  self.draw_line()
        elif event.type == pygame.MOUSEBUTTONUP:
          pos = pygame.mouse.get_pos()
          self.move(pos, Color_dict["white"])
      pygame.display.update()
      self.FPSCLOCK.tick(FPS)

  def reset(self):
    full_screenx = self.SCREENX + self.XGAP[0] + self.XGAP[1]
    full_screeny = self.SCREENY + self.YGAP[0] + self.YGAP[1]
    #self.SCREEN = pygame.display.set_mode((full_screenx, full_screeny),HWSURFACE|DOUBLEBUF|RESIZABLE)
    self.SCREEN = pygame.display.set_mode((full_screenx, full_screeny))
    self.SCREEN.fill(self.Color_screen)
    self.BOARDX = [self.SCREENX*0.1 + self.XGAP[0], self.SCREENX*0.9 + self.XGAP[0]]
    self.BOARDY = [self.SCREENY*0.1 + self.YGAP[0], self.SCREENY*0.9 + self.YGAP[0]]
    self.colLines = np.linspace(self.BOARDX[0], self.BOARDX[1], self.Ncol+1)
    self.rowLines = np.linspace(self.BOARDY[0], self.BOARDY[1], self.Nrow+1)
    self.MidptColLines = 0.5*(self.colLines[1:] + self.colLines[:-1])
    self.MidptRowLines = 0.5*(self.rowLines[1:] + self.rowLines[:-1])
    dx = self.colLines[1] - self.colLines[0]
    dy = self.rowLines[1] - self.rowLines[0]
    self.stone_radius = int(min(dx,dy)*0.4)
 
  def draw_line(self):
    for c in self.colLines:
      pygame.draw.line(self.SCREEN, self.Color_line, (c,self.BOARDY[0]), (c,self.BOARDY[1]), 5)
    for r in self.rowLines:
      pygame.draw.line(self.SCREEN, self.Color_line, (self.BOARDX[0],r), (self.BOARDX[1],r), 5)

  def draw_stone(self, state):
    pos_black = np.argwhere(state == 1)
    pos_white = np.argwhere(state == -1)
    for bpos in pos_black:
      self.move(bpos, Color_dict["black"])
    for wpos in pos_white:
      self.move(bpos, Color_dict["white"])

  def move(self, pos, color):
    if self.BOARDX[1] > pos[0] > self.BOARDX[0] and self.BOARDY[1] > pos[1] > self.BOARDY[0]:
      grid_x = int(self.MidptColLines[np.argmin(np.abs(self.MidptColLines-pos[0]))])
      grid_y = int(self.MidptRowLines[np.argmin(np.abs(self.MidptRowLines-pos[1]))])
      pygame.draw.circle(self.SCREEN,color,(grid_x,grid_y), self.stone_radius)

def game_selection():
  pass

if __name__ == '__main__':
  Board_gui()

