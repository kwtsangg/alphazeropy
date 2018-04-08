#!/usr/bin/env python
from __future__ import print_function # to prevent Py2 from interpreting it as a tuple
__file__       = "train.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2018"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2018-Feb-15"

Description="""
  To train my AI.
  Example:
      python train.py --game gomoku --n-in-row 5 --board-height 9 --board-width 9 --reflection-symmetry
      python train.py --game connectfour --n-in-row 4 --board-height 6 --board-width 7 --reflection-symmetry --rotation-symmetry 0
      python train.py --game reversi --board-height 8 --board-width 8 --reflection-symmetry
"""

#===============================================================================
#  Module
#===============================================================================
import numpy as np
import sys
import argparse, textwrap

from server import Server
from alphazero import AlphaZero
from mcts_id import MCTS_player # or from mcts_cyclic_ref import MCTS_player

#===============================================================================
#  Main
#===============================================================================

class train_pipeline:
  def __init__(self, args):
    # board params
    self.game             = args.game
    if self.game is None:
      raise ValueError("Please let me know which game you want to train by --game")

    self.board_height     = args.board_height
    self.board_width      = args.board_width
    self.n_in_row         = args.n_in_row
    if self.game in ["gomoku", "connectfour"]:
      if self.n_in_row is None:
        raise ValueError("Please let me know the winning criteria by --n-in-row")
      self.savename       = "%s_n_in_row_%i" % (self.game, self.n_in_row)
    else:
      self.savename       = self.game

    sys.path.insert(0, '%s/' % self.game)
    from board import Board
    self.Board            = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
    self.server           = Server(self.Board)

    # AI brain params
    self.load_path         = args.load_path
    self.save_path         = args.save_path
    self.n_feature_plane   = args.n_feature_plane
    self.n_filter          = args.n_filter
    self.kernel_size_conv  = args.kernel_size_conv
    self.kernel_size_res   = args.kernel_size_res
    self.n_res_blocks      = args.n_res_blocks
    self.l2_regularization = args.l2_regularization
    self.bn_axis           = args.bn_axis

    self.AI_brain          = AlphaZero(
              save_path         = self.save_path,
              board_height      = self.board_height,
              board_width       = self.board_width,
              n_feature_plane   = self.n_feature_plane,
              n_filter          = self.n_filter,
              kernel_size_conv  = self.kernel_size_conv,
              kernel_size_res   = self.kernel_size_res,
              n_res_blocks      = self.n_res_blocks,
              l2_regularization = self.l2_regularization,
              bn_axis           = self.bn_axis
            )
    if self.load_path is not None:
      self.AI_brain.load_class(self.load_path)

    # training params 
    self.optimizer             = args.optimizer
    self.learning_rate         = args.learning_rate
    self.learning_momentum     = args.learning_momentum
    self.temp                  = args.temp
    self.temp_trans_after      = args.temp_trans_after
    self.n_rollout             = args.n_rollout
    self.c_puct                = args.c_puct
    self.batch_size            = args.batch_size
    self.epochs                = args.epochs

    self.AI_player = MCTS_player(self.AI_brain.predict, c_puct = self.c_puct, n_rollout = self.n_rollout, is_self_play = True, temp = self.temp)

    # other training params
    self.play_batch_size             = args.play_batch_size
    self.game_batch_size             = args.game_batch_size
    self.rotation_symmetry           = args.rotation_symmetry
    if not 0 in self.rotation_symmetry:
      self.rotation_symmetry.append(0)
    self.reflection_symmetry         = args.reflection_symmetry

  #===============================#
  # Train pipeline 
  #===============================#

  def get_game_data(self, play_batch_size=1):
    state_result_list  = []
    policy_result_list = []
    value_result_list  = []
    for i in range(play_batch_size):
      game_data_output = self.server.start_self_play(self.AI_player, is_shown=True, temp_trans_after=self.temp_trans_after)[1]
      state_list, policy_list, value_list = self.get_dihedral_game_data(game_data_output)
      state_result_list.extend(state_list)
      policy_result_list.extend(policy_list)
      value_result_list.extend(value_list)
    return state_result_list, policy_result_list, value_result_list

  def get_dihedral_game_data(self, game_data_output):
    """
      Because there are symmetries of the game board, this funcion genereates new game data set by rotations and reflections.
    """
    state_result_list  = []
    policy_result_list = []
    value_result_list  = []
    for one_training_set in game_data_output:
      # state has a shape of (n_feature_plane, height, weight)
      # policy has a shape of (height * weight + 1, )
      # value has a shape of (1,)
      state  = one_training_set[0]
      policy = one_training_set[1]
      value  = one_training_set[2]

      n_feature_plane, height, width = state.shape

      for i in self.rotation_symmetry:
        # rotate the state to generate new game data set
        for j in range(n_feature_plane):
          state[j] = np.rot90(state[j], i)
        state_result_list.append( state )
        tmp_policy = np.rot90(policy[:-1].reshape(height, width), i).reshape(-1,)
        policy_result_list.append( np.append(tmp_policy, policy[-1]) )
        value_result_list.append( value )

        # reflect the state horizontally before rotation
        if self.reflection_symmetry:
          for j in range(n_feature_plane):
            state[j] = np.rot90(np.fliplr(state[j]), i)
          state_result_list.append( state )
          tmp_policy = np.rot90(np.fliplr(policy[:-1].reshape(height, width)), i).reshape(-1,)
          policy_result_list.append( np.append(tmp_policy, policy[-1]) )
          value_result_list.append( value )
    return state_result_list, policy_result_list, value_result_list

  def train(self):
    try:
      train_x, train_y_policy, train_y_value = [], [], []
      for i in range(self.game_batch_size):
        print("%i/%i" % (i, self.game_batch_size))
        state_result_list, policy_result_list, value_result_list = self.get_game_data(self.play_batch_size)
        train_x.extend(state_result_list)
        train_y_policy.extend(policy_result_list)
        train_y_value.extend(value_result_list)

        if len(train_x) > self.batch_size:
          print("Training ...")
          self.AI_brain.train(np.array(train_x), [np.array(train_y_policy), np.array(train_y_value)], optimizer=self.optimizer, learning_rate=self.learning_rate, learning_momentum=self.learning_momentum, epochs=self.epochs, batch_size=self.batch_size)
          train_x, train_y_policy, train_y_value = [], [], []

      self.AI_brain.save_class(name=self.savename)
    except KeyboardInterrupt:
      print("Saving model ...")
      self.AI_brain.save_class(name=self.savename)
      pass

if __name__ == "__main__":
  class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass
  parser = argparse.ArgumentParser(description=textwrap.dedent(Description), prog=__file__, formatter_class=CustomFormatter)
  # board params
  parser.add_argument("--game",                                   action="store",            type=str,   help="gomoku, connectfour, reversi")
  parser.add_argument("--board-height",        default=6 ,        action="store",            type=int,   help="height of the board")
  parser.add_argument("--board-width",         default=6 ,        action="store",            type=int,   help="width of the board")
  parser.add_argument("--n-in-row",                               action="store",            type=int,   help="needed if game is gomoku or connectfour")
  # AI brain params
  parser.add_argument("--save-path",                              action="store",            type=str,   help="directory path that trained model will be saved in")
  parser.add_argument("--load-path",                              action="store",            type=str,   help="directory path containing trained model")
  parser.add_argument("--n-feature-plane",     default=3,         action="store",            type=int,   help="number of feature plane")
  parser.add_argument("--n-filter",            default=32,        action="store",            type=int,   help="number of filters used in conv2D")
  parser.add_argument("--kernel-size-conv",    default=(3,3),     action="store",            type=tuple, help="kernel size of first convolution layer")
  parser.add_argument("--kernel-size-res",     default=(3,3),     action="store",            type=tuple, help="kernel size of residual blocks")
  parser.add_argument("--n-res-blocks",        default=5,         action="store",            type=int,   help="number of residual blocks")
  parser.add_argument("--l2-regularization",   default=1e-4,      action="store",            type=float, help="a parameter controlling the level of L2 weight regularizatio to prevent overfitting")
  parser.add_argument("--bn-axis",             default=-1,        action="store",            type=int,   help="batch normalization axis. For 'tf', it is 3. For 'th', it is 1.")
  # training params 
  parser.add_argument("--optimizer",           default="adam",    action="store",            type=str,   help="method of gradient descent (adam or sgd)")
  parser.add_argument("--learning-rate",       default=1e-3,      action="store",            type=float, help="learning rate")
  parser.add_argument("--learning-momentum",   default=0.9,       action="store",            type=float, help="learning momentum of sgd")
  parser.add_argument("--temp",                default=1.,        action="store",            type=float, help="temperature to control how greedy of selecting next action")
  parser.add_argument("--temp-trans-after",    default=20,        action="store",            type=int,   help="after this number of moves, the temperature becomes 1e-3.")
  parser.add_argument("--n-rollout",           default=400,       action="store",            type=int,   help="number of simulations for each move")
  parser.add_argument("--c-puct",              default=5.,        action="store",            type=float, help="coefficient of controlling the extent of exploration versus exploitation")
  parser.add_argument("--batch-size",          default=512,       action="store",            type=int,   help="mini-batch size for training")
  parser.add_argument("--epochs",              default=50,        action="store",            type=int,   help="number of training steps for each gradient descent update")
  # other training params
  parser.add_argument("--play-batch-size",     default=1,         action="store",            type=int,   help="number of games generated in each calling")
  parser.add_argument("--game-batch-size",     default=5000,      action="store",            type=int,   help="total number of games generated")
  parser.add_argument("--rotation-symmetry",   default=[0,1,2,3], action="store", nargs="+", type=int,   help="rotational symmetry (anti-clockwise), 0 = no rotation, 1 = 90 deg, 2 = 180 deg, 3 = 270 deg")
  parser.add_argument("--reflection-symmetry", default=False,     action="store_true",                   help="make use of reflection symmetry to generate more game data ")
  # other
  parser.add_argument("--version", action="version", version='%(prog)s ' + __version__)
  args = parser.parse_args()

  a = train_pipeline(args)
  a.train()
  

