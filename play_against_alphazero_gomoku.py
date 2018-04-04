import game
from AlphaZero_Gomoku import AlphaZero_Gomoku
from mcts_id import MCTS_player

board_height     = 6
board_width      = 6
n_in_row         = 4
n_feature_plane  = 3
is_pass_disabled = True
n_rollout        = 2000

AI_brain1  = AlphaZero_Gomoku(board_height=board_height, board_width=board_width, n_in_row=n_in_row, n_feature_plane=n_feature_plane)
AI_brain1.load_class("trained_model/201804030251_model_6_6_n_4_res_blocks_5/")
#AI_brain2  = AlphaZero_Gomoku(board_height=board_height, board_width=board_width, n_in_row=n_in_row, n_feature_plane=n_feature_plane)
#AI_brain2.load_class("trained_model/201804030251_model_6_6_n_4_res_blocks_5/")

AI_player1 = MCTS_player(AI_brain1.predict, name='AlphaZero_Gomoku1', n_rollout=n_rollout, is_self_play=False, c_puct=5., temp=1e-3)
#AI_player2 = MCTS_player(AI_brain2.predict, name='AlphaZero_Gomoku2', n_rollout=n_rollout, is_self_play=False, c_puct=5., temp=1e-3)

board  = game.Board(width=board_width, height=board_height, n_in_row=n_in_row, is_pass_disabled=is_pass_disabled)
server = game.Server(board)
#server.start_game(player2=AI_player1, player1=AI_player2)
server.start_game(player1=AI_player1)
#server.start_game()
