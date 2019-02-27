import os, sys
import numpy as np
import pygame
from pygame.locals import *
from gui import *

FPS = 30
XGAP = [0,0]
YGAP = [100,0]
SCREENX = 512 + XGAP[0] + XGAP[1]
SCREENY = 512 + YGAP[0] + YGAP[1]
FRAMEX = SCREENX - XGAP[0] - XGAP[1]
FRAMEY = SCREENY - YGAP[0] - YGAP[1]
SW = [FRAMEX*0.1 + XGAP[0], FRAMEX*0.9 + XGAP[0]]
SH = [FRAMEY*0.1 + YGAP[0], FRAMEY*0.9 + YGAP[0]]
Color_screen = 	(129, 216, 208) # Tiffany
Color_line = (0,0,0)

def main():
  global SCREEN, FPSCLOCK, SCREENX, SCREENY, SW, SH
  pygame.init()
  FPSCLOCK = pygame.time.Clock()
  # First time setup
  SCREEN = pygame.display.set_mode((SCREENX, SCREENY),HWSURFACE|DOUBLEBUF|RESIZABLE)
  SCREEN.fill(Color_screen)
  pygame.display.set_caption('AlphaZero Platform')
  boardstyle = "grid"
  draw_board(7,6,boardstyle)
  while True:
    for event in pygame.event.get():
      if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
        pygame.quit()
        sys.exit()
      elif event.type==VIDEORESIZE:
        SCREEN = pygame.display.set_mode(event.dict['size'],HWSURFACE|DOUBLEBUF|RESIZABLE)
        SCREEN.fill(Color_screen)
        SCREENX = event.dict['size'][0]
        SCREENY = event.dict['size'][1]
        FRAMEX = SCREENX - XGAP[0] - XGAP[1]
        FRAMEY = SCREENY - YGAP[0] - YGAP[1]
        SW = [FRAMEX*0.1 + XGAP[0], FRAMEX*0.9 + XGAP[0]]
        SH = [FRAMEY*0.1 + YGAP[0], FRAMEY*0.9 + YGAP[0]]
        draw_board(7,6,boardstyle)
    pygame.display.update()
    FPSCLOCK.tick(FPS)

def draw_board(Ncol, Nrow, style="grid", color_line=(0,0,0)):
  if style=="space":
    Ncol += 1
    Nrow += 1
  colLines = np.linspace(SW[0], SW[1], Ncol)
  rowLines = np.linspace(SH[0], SH[1], Nrow)
  
  for c in colLines:
    pygame.draw.line(SCREEN,Color_line, (c,SH[0]), (c,SH[1]), 5)
  for r in rowLines:
    pygame.draw.line(SCREEN,Color_line, (SW[0],r), (SW[1],r), 5)
  dx = colLines[1] - colLines[0]
  dy = rowLines[1] - rowLines[0]
  return dx, dy

def game_selection():
  pass

if __name__ == '__main__':
    main()

