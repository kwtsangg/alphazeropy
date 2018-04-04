import numpy as np
import sys
sys.path.insert(0, '../')
from AlphaZero_Gomoku import AlphaZero_Gomoku

width           = 3
height          = 3
n_in_row        = 3
n_feature_plane = 3

myCNN = AlphaZero_Gomoku(board_height=height, board_width=width, n_in_row=n_in_row, n_feature_plane=n_feature_plane)
myCNN.load_class("trained_model/201802241746_model_3_3_n_3_res_blocks_4/")

a = np.array([  [[1,1,1],[1,1,1],[1,1,1]], [[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]   ])
out = myCNN.predict(np.array([ a ]), raw_output=False)
print a[0]
print a[1]
print out[0][0]
print out[0][2]
print ""

a = np.array([  [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]], [[1,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]   ])
out = myCNN.predict(np.array([ a ]), raw_output=False)
print a[0]
print a[1]
print out[0][0]
print out[0][2]
print ""
