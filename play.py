#!/usr/bin/env python
__file__       = "play.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2018"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2018 Apr 04"

Description=""" A platform to play games.
"""

#================================================================
# Module
#================================================================
import numpy as np
import sys
import argparse, textwrap

from server import Server, Human
from alphazero import AlphaZero
from mcts_id import MCTS_player # or from mcts_cyclic_ref import MCTS_player

#================================================================
# Main
#================================================================
class platform:
  def __init__(self, args):
    # board params
    self.game             = args.game
    if self.game is None:
      raise ValueError("Please let me know which game you want to play by --game")

    self.board_height     = args.board_height
    self.board_width      = args.board_width
    self.n_in_row         = args.n_in_row
    if self.game in ["gomoku", "connectfour"]:
      if self.n_in_row is None:
        raise ValueError("Please let me know the winning criteria by --n-in-row")

    # player1, AI/human brain params
    self.p1_temp        = args.p1_temp
    self.p1_n_rollout   = args.p1_n_rollout
    self.p1_c_puct      = args.p1_c_puct

    self.p1_brain_path  = args.p1_brain
    self.p1_name        = args.p1_name
    if self.p1_brain_path is None:
      self.p1           = Human(name=self.p1_name)
    else:
      self.p1_brain     = AlphaZero()
      self.p1_brain.load_class(self.p1_brain_path)
      self.p1           = MCTS_player(self.p1_brain.predict, c_puct = self.p1_c_puct, n_rollout = self.p1_n_rollout, temp = self.p1_temp, name="AlphaZero "+self.p1_name)
      print("Overwriting board size according to trained model (player 1) ...")
      self.board_height = self.p1_brain.board_height
      self.board_width  = self.p1_brain.board_width

    # player2, AI/human brain params
    self.p2_temp       = args.p2_temp
    self.p2_n_rollout  = args.p2_n_rollout
    self.p2_c_puct     = args.p2_c_puct

    self.p2_brain_path = args.p2_brain
    self.p2_name       = args.p2_name
    if self.p2_brain_path is None:
      self.p2          = Human(name=self.p2_name)
    else:
      self.p2_brain    = AlphaZero()
      self.p2_brain.load_class(self.p2_brain_path)
      self.p2          = MCTS_player(self.p2_brain.predict, c_puct = self.p2_c_puct, n_rollout = self.p2_n_rollout, temp = self.p2_temp, name="AlphaZero "+self.p2_name)
      print("Overwriting board size according to trained model (player 2) ...")
      self.board_height = self.p2_brain.board_height
      self.board_width  = self.p2_brain.board_width

    # Initialize board
    sys.path.insert(0, '%s/' % self.game)
    from board import Board
    self.Board  = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
    self.server = Server(self.Board)

  def start_game(self):
    self.server.start_game(player1=self.p1, player2=self.p2)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=textwrap.dedent(Description), prog=__file__, formatter_class=argparse.RawTextHelpFormatter)
  # player params
  parser.add_argument("--p1-brain",                        action="store",  type=str,   help="player1, directory path that store AI brain (empty means human player)")
  parser.add_argument("--p1-name",       default="Alice",  action="store",  type=str,   help="player1, name")
  parser.add_argument("--p2-brain",                        action="store",  type=str,   help="player2, directory path that store AI brain (empty means human player)")
  parser.add_argument("--p2-name",       default="Bob",    action="store",  type=str,   help="player2, name")
  # board params
  parser.add_argument("--game",                            action="store",  type=str,   help="gomoku, connectfour, reversi")
  parser.add_argument("--board-height",  default=6 ,       action="store",  type=int,   help="height of the board")
  parser.add_argument("--board-width",   default=6 ,       action="store",  type=int,   help="width of the board")
  parser.add_argument("--n-in-row",                        action="store",  type=int,   help="needed if game is gomoku or connectfour")
  # AI brain params
  parser.add_argument("--p1-temp",       default=1e-3,     action="store",  type=float, help="player1, temperature to control how greedy of selecting next action")
  parser.add_argument("--p1-n-rollout",  default=400,      action="store",  type=int,   help="player1, number of simulations for each move")
  parser.add_argument("--p1-c-puct",     default=5,        action="store",  type=int,   help="player1, coefficient of controlling the extent of exploration versus exploitation")
  parser.add_argument("--p2-temp",       default=1e-3,     action="store",  type=float, help="player2, temperature to control how greedy of selecting next action")
  parser.add_argument("--p2-n-rollout",  default=400,      action="store",  type=int,   help="player2, number of simulations for each move")
  parser.add_argument("--p2-c-puct",     default=5,        action="store",  type=int,   help="player2, coefficient of controlling the extent of exploration versus exploitation")
  # other
  parser.add_argument("--version", action="version", version='%(prog)s ' + __version__)
  args = parser.parse_args()

  a = platform(args)
  a.start_game()

