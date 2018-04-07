#!/usr/bin/env python
__file__       = "gui.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2018"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2018 Apr 06"

Description=""" To make a gui for playing gomoku.
"""

#================================================================
# Module
#================================================================
import pygame
from sys import exit

#================================================================
# Global parameters
#================================================================
color = {}
color["brown"] = (255,222,173)
color["black"] = (0, 0, 0)
color["white"] = (255, 255, 255)

#================================================================
# Main
#================================================================
def main():
  while True:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        exit()
      if event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 1:
          print("hi")

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Wendy')
    screen = pygame.display.set_mode([800,800], 0, 32)
    screen.fill(color["brown"])
    pygame.Rect(45, 45, 720, 720)
    pygame.display.update()
    main()



