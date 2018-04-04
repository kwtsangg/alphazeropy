import numpy as np
import sys
sys.path.insert(0, '../')
from AlphaZero_Gomoku import AlphaZero_Gomoku
from mcts_id import MCTS_player
import game

board_height     = 3
board_width      = 3
n_in_row         = 3
n_feature_plane  = 3
n_rollout        = 500
AI_brain1  = AlphaZero_Gomoku(board_height=board_height, board_width=board_width, n_in_row=n_in_row, n_feature_plane=n_feature_plane)
AI_brain2  = AlphaZero_Gomoku(board_height=board_height, board_width=board_width, n_in_row=n_in_row, n_feature_plane=n_feature_plane)
AI_player1 = MCTS_player(AI_brain1.predict, name='AlphaZero_Gomoku1', n_rollout=n_rollout, is_self_play=False, c_puct=1., temp=1.)
AI_player2 = MCTS_player(AI_brain2.predict, name='AlphaZero_Gomoku2', n_rollout=n_rollout, is_self_play=False, c_puct=1., temp=1.)

board = game.Board(width=board_width, height=board_height, n_in_row=n_in_row)
server = game.Server(board)
output = server.start_self_play(AI_player1, AI_player2)
for ioutput in output[1]:
  print ioutput
