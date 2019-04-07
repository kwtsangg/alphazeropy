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
    To create an untrained-MCTS brain:
      python train.py --game gomoku --n-in-row 5 --board-height 9 --board-width 9
      python train.py --game connectfour --n-in-row 4 --board-height 6 --board-width 7
      python train.py --game reversi --board-height 8 --board-width 8

    To generate gameplay:
      python train.py --game connectfour --n-in-row 4 --board-height 6 --board-width 7 --n-filter 32 --n-rollout 400 --c-puct 5 --generate-game-data-only

    To train on gameplay:
      python train.py --game connectfour --n-in-row 4 --board-height 6 --board-width 7 --n-filter 32 --batch-size 1024 --load-latest-model --epochs 200 --learning-rate 1e-2 --train-on-game-data-only --train-on-last-n-sets 5000
"""

#===============================================================================
#  Module
#===============================================================================
import numpy as np
import os, sys
import argparse, textwrap
import time
from datetime import datetime

#===============================================================================
#  Main
#===============================================================================

class train_pipeline:
  def __init__(self, args):
    # other
    self.engine = args.engine

    # board params
    self.game             = args.game
    if self.game is None:
      raise ValueError("Please let me know at least which game you want to train by --game")

    self.board_height     = args.board_height
    self.board_width      = args.board_width
    self.n_in_row         = args.n_in_row

    # AI brain params
    self.load_path              = args.load_path
    self.save_path              = args.save_path
    self.load_latest_model_flag = args.load_latest_model
    self.n_filter               = args.n_filter
    self.kernel_size_conv       = args.kernel_size_conv
    self.kernel_size_res        = args.kernel_size_res
    self.n_res_blocks           = args.n_res_blocks
    self.l2_regularization      = args.l2_regularization
    self.bn_axis                = args.bn_axis
    self.model_no               = 0

    from alphazero import AlphaZero
    self.AI_brain = None
    if self.load_latest_model_flag:
      self.AI_brain = AlphaZero()
      print("Loading latest trained model (It may overwrite --load-path) ...")
      latest_model_dir = self.load_latest_model()
      if self.model_no == 0:
        print("An untrained brain will be created.")
        self.AI_brain = None
      else:
        print("latest model (%i) in %s is loaded." % (self.model_no, latest_model_dir))
        print("Overwritting board size according to trained model ...")
        self.board_height = self.AI_brain.board_height
        self.board_width  = self.AI_brain.board_width
    elif self.load_path:
      self.AI_brain = AlphaZero()
      print("Loading trained model %s ..." % self.load_path)
      self.AI_brain.load_class(self.load_path, engine=self.engine)
      print("Overwritting board size according to trained model ...")
      self.board_height = self.AI_brain.board_height
      self.board_width  = self.AI_brain.board_width
      self.model_no     = dir_path.split("/")
      if len(self.model_no[-1]) == 0:
        self.model_no   = model_no[-2]
      else:
        self.model_no   = model_no[-1]

    # Initialize Board (and/or self.AI_brain)
    sys.path.insert(0, '%s/' % self.game)
    from board import Board
    from server import Server
    self.Board            = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
    self.server           = Server(self.Board)
    self.n_feature_plane  = self.Board.n_feature_plane

    # Save the Board params in case they are None
    if self.board_height is None:
      self.board_height = self.Board.height
      print("Using default board height of %i ..." % self.board_height)
    if self.board_width is None:
      self.board_width = self.Board.width
      print("Using default board width of %i ..." % self.board_width)
    if self.game in ["gomoku", "connectfour"]:
      if self.n_in_row is None:
        self.n_in_row = self.Board.n_in_row
        print("Using default n_in_row of %i as winning criteria ..." % self.n_in_row)
      self.savename = "%s_n_in_row_%i" % (self.game, self.n_in_row)
    else:
      self.savename = "%s" % (self.game)

    # Creating an untrained brain
    if not self.AI_brain:
      print("Creating and saving a new untrained brain of AlphaZero ...")
      self.AI_brain          = AlphaZero(
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
      self.model_no, model_dir = self.AI_brain.save_class(name=self.savename, path=self.save_path)
      np.savetxt("%s/elo.txt" % (model_dir), [0.], header="An untrained-MCTS brain (which elo is defined to be 0)")
      np.savetxt("%s/game.txt" % (model_dir), [0], header="Number of gameplay")

    # AI hyper-params
    self.temp                  = args.temp
    self.n_rollout             = args.n_rollout
    self.epsilon               = args.epsilon
    self.dir_param             = args.dir_param
    self.s_thinking            = args.s_thinking
    self.use_thinking          = args.use_thinking
    self.c_puct                = args.c_puct

    # training params 
    self.learning_rate         = args.learning_rate
    self.learning_rate_f       = args.learning_rate_f
    self.batch_size            = args.batch_size
    self.epochs                = args.epochs

    from mcts_cyclic_ref import MCTS_player
    self.AI_player = MCTS_player(self.AI_brain.predict, c_puct = self.c_puct, n_rollout = self.n_rollout, epsilon = self.epsilon, dirichlet_param = self.dir_param, temp = self.temp, s_thinking=self.s_thinking, use_thinking=self.use_thinking)

    # modes and modes options
    self.game_data_dir           = "%s/%s_game_data" % (self.save_path, self.savename)
    self.generate_game_data_only = args.generate_game_data_only
    self.train_on_game_data_only = args.train_on_game_data_only

    self.max_game_gen            = args.max_game_gen
    self.fraction_game_survive   = args.fraction_game_survive
    self.train_on_last_n_sets    = args.train_on_last_n_sets
    self.train_every_mins        = args.train_every_mins
    self.train_rounds            = args.train_rounds

    if self.generate_game_data_only:
      # if the dir doesnt exist, mkdir it
      if not os.path.isdir(self.game_data_dir):
        print("Making a directory to store game data (%s) ..." % self.game_data_dir)
        os.system("mkdir -p %s" % self.game_data_dir)

    if self.train_on_game_data_only:
      if not os.path.isdir(self.game_data_dir):
        raise ValueError("The game data should be put in a subdirectory of --save-path with name %s_game_data" % self.savename)

  #===============================#
  # Gameplay generating functions
  #===============================#

  def load_latest_model(self):
    print("Checking on latest model ...")
    latest_model_no  = self.model_no
    latest_model_dir = ""
    try: # Python2
      model_dir_list = os.walk(self.save_path).next()[1]
    except: # Python3
      model_dir_list = next(os.walk(self.save_path))[1]
    for model_dir in model_dir_list:
      try:
        model_no = int(model_dir.split("_")[0])
        if model_no > latest_model_no:
          latest_model_no  = model_no
          latest_model_dir = model_dir
      except:
        pass
    if latest_model_no == self.model_no:
      print("No latest model. Keep the current model.")
    else:
      print("Loading latest model '%s/%s' ..." % (self.save_path, latest_model_dir))
      self.AI_brain.load_class("%s/%s" % (self.save_path, latest_model_dir), engine=self.engine)
      self.model_no = latest_model_no
    return "%s/%s" % (self.save_path, latest_model_dir)

  def get_game_data(self, max_game_gen=1):
    state_result_list  = []
    policy_result_list = []
    value_result_list  = []
    for i in range(max_game_gen):
      game_data_output = self.server.start_self_play(self.AI_player, is_shown=True)[1]
      state_list, policy_list, value_list = self.get_dihedral_game_data(game_data_output)
      state_result_list.extend(state_list)
      policy_result_list.extend(policy_list)
      value_result_list.extend(value_list)
    return state_result_list, policy_result_list, value_result_list

  def get_game_data_parallel(self):
    for i in range(self.max_game_gen):
      print("%i/%i" % (i, self.max_game_gen))
      # Writing in this way is to solve a rare bug.
      # If saving and loading of a new model happen at the same time, 'resources temporarily unavailable' appears.
      # If so, load it once more.
      try:
        latest_model_dir = self.load_latest_model()
      except:
        latest_model_dir = self.load_latest_model()
      print("Generating game data ...")
      game_data_output = self.server.start_self_play(self.AI_player, is_shown=True)[1]
      state_list, policy_list, value_list = self.get_dihedral_game_data(game_data_output)
      np.save("%s/%s_%s_%s_model_%s.npy" % (self.game_data_dir, self.savename, datetime.today().strftime('%Y%m%d%H%M%S'), os.getpid(), self.model_no), list(zip(state_list, policy_list, value_list)))

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

      for i in self.Board.rotation_symmetry:
        # rotate the state to generate new game data set
        for j in range(n_feature_plane):
          state[j] = np.rot90(state[j], i)
        state_result_list.append( state )
        tmp_policy = np.rot90(policy[:-1].reshape(height, width), i).reshape(-1,)
        policy_result_list.append( np.append(tmp_policy, policy[-1]) )
        value_result_list.append( value )

        # reflect the state horizontally before rotation
        if 1. in self.Board.reflection_symmetry:
          for j in range(n_feature_plane):
            state[j] = np.rot90(np.fliplr(state[j]), i)
          state_result_list.append( state )
          tmp_policy = np.rot90(np.fliplr(policy[:-1].reshape(height, width)), i).reshape(-1,)
          policy_result_list.append( np.append(tmp_policy, policy[-1]) )
          value_result_list.append( value )
    return state_result_list, policy_result_list, value_result_list

  #================================================================
  # Training function
  #================================================================

  def train(self):
    """
      Generating gamedata until certain amount, and then train on those gamedata.
    """
    train_x, train_y_policy, train_y_value = [], [], []
    game_length = []

    try:
      i = 0
      for gamedata in sorted(os.listdir(self.game_data_dir))[::-1][:self.train_on_last_n_sets]:
        if gamedata.endswith(".npy"):
          i += 1
          print("%5i importing %s" % (i, gamedata))
          state_result_list, policy_result_list, value_result_list = list(zip(* np.load("%s/%s" % (self.game_data_dir, gamedata)) ))
          train_x.extend(state_result_list)
          train_y_policy.extend(policy_result_list)
          train_y_value.extend(value_result_list)
          game_length.append(len(state_result_list))
    except:
      print("Fail to import existing game data. Skipping ...")

    for i in range(self.max_game_gen):
      print("%i/%i" % (i, self.max_game_gen))
      state_result_list, policy_result_list, value_result_list = self.get_game_data(1)
      train_x.extend(state_result_list)
      train_y_policy.extend(policy_result_list)
      train_y_value.extend(value_result_list)
      game_length.append(len(state_result_list))
      np.save("%s/%s_%s_%s_model_%s.npy" % (self.game_data_dir, self.savename, datetime.today().strftime('%Y%m%d%H%M%S'), os.getpid(), self.model_no), list(zip(state_result_list, policy_result_list, value_result_list)))
      if len(game_length) > self.train_on_last_n_sets and len(train_x) > self.batch_size:
        print("Training ...")
        self.AI_brain.train(np.array(train_x), [np.array(train_y_policy), np.array(train_y_value)], learning_rate=self.learning_rate, learning_rate_f=self.learning_rate_f, epochs=self.epochs, batch_size=self.batch_size)
        game_length = game_length[-int(len(game_length)*self.fraction_game_survive):]
        sum_game_length = sum(game_length)
        train_x = train_x[-sum_game_length:]
        train_y_policy = train_y_policy[-sum_game_length:]
        train_y_value = train_y_value[-sum_game_length:]
        self.model_no, savepath = self.AI_brain.save_class(name=self.savename, path=self.save_path)
    self.AI_brain.save_class(name=self.savename, path=self.save_path)

  def train_on_dir(self):
    for i_rounds in range(self.train_rounds):
      i = 0
      train_x, train_y_policy, train_y_value = [], [], []
      number_of_gameplay = len(os.listdir(self.game_data_dir))
      for gamedata in sorted(os.listdir(self.game_data_dir))[::-1][:self.train_on_last_n_sets]:
        if gamedata.endswith(".npy"):
          i += 1
          print("%5i importing %s" % (i, gamedata))
          state_result_list, policy_result_list, value_result_list = list(zip(* np.load("%s/%s" % (self.game_data_dir, gamedata)) ))
          train_x.extend(state_result_list)
          train_y_policy.extend(policy_result_list)
          train_y_value.extend(value_result_list)

      if len(train_x) > self.batch_size:
        print("Training ...")
        self.AI_brain.train(np.array(train_x), [np.array(train_y_policy), np.array(train_y_value)], learning_rate=self.learning_rate, learning_rate_f=self.learning_rate_f, epochs=self.epochs, batch_size=self.batch_size)
        train_x, train_y_policy, train_y_value = [], [], []
        print("Saving the trained model ...")
        self.model_no, savepath = self.AI_brain.save_class(name=self.savename, path=self.save_path)
        np.savetxt("%s/game.txt" % (savepath), [number_of_gameplay], header="Number of gameplay")
      else:
        print("The model is not trained. Probably because of lack of game data. sample %i, batch size %i" % (len(train_x), self.batch_size))
      if i_rounds < self.train_rounds-1:
        print("%s the next training will start after %s mins" % (datetime.today().strftime('%Y%m%d%H%M%S'), self.train_every_mins))
        time.sleep(self.train_every_mins*60.)

#================================================================
# Footer
#================================================================

if __name__ == "__main__":
  class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass
  parser = argparse.ArgumentParser(description=textwrap.dedent(Description), prog=__file__, formatter_class=CustomFormatter)
  # board params
  parser.add_argument("--game",                                     action="store",            type=str,   help="gomoku, connectfour, reversi")
  parser.add_argument("--board-height",                             action="store",            type=int,   help="height of the board")
  parser.add_argument("--board-width",                              action="store",            type=int,   help="width of the board")
  parser.add_argument("--n-in-row",                                 action="store",            type=int,   help="needed if game is gomoku or connectfour")
  # AI brain params
  parser.add_argument("--save-path", default="%s/{}_training_model" % os.getcwd(), action="store", type=str, help="directory path that trained model will be saved in")
  parser.add_argument("--load-path",                                action="store",            type=str,   help="directory path of trained model")
  parser.add_argument("--load-latest-model",   default=False,       action="store_true",                   help="load latest trained model from the directory of --save-path instead of creating a new model")
  parser.add_argument("--n-filter",            default=32,          action="store",            type=int,   help="number of filters used in conv2D")
  parser.add_argument("--kernel-size-conv",    default=(3,3),       action="store",            type=tuple, help="kernel size of first convolution layer")
  parser.add_argument("--kernel-size-res",     default=(3,3),       action="store",            type=tuple, help="kernel size of residual blocks")
  parser.add_argument("--n-res-blocks",        default=5,           action="store",            type=int,   help="number of residual blocks")
  parser.add_argument("--l2-regularization",   default=1e-4,        action="store",            type=float, help="a parameter controlling the level of L2 weight regularizatio to prevent overfitting")
  parser.add_argument("--bn-axis",             default=1,           action="store",            type=int,   help="batch normalization axis. For 'channel first' = 1, 'channel last' = -1.")
  # AI params
  parser.add_argument("--temp",                default=1.,          action="store",            type=float, help="temperature to control how greedy of selecting next action")
  parser.add_argument("--n-rollout",           default=400,         action="store",            type=int,   help="number of simulations for each move")
  parser.add_argument("--epsilon",             default=0.25,        action="store",            type=float, help="fraction of noise in prior probability")
  parser.add_argument("--dir-param",           default=0.1,         action="store",            type=float, help="extent of encouraging exploration")
  parser.add_argument("--s-thinking",          default=1,           action="store",            type=int,   help="time allowed to think for each move (in seconds)")
  parser.add_argument("--use-thinking",        default=False,       action="store_true",                   help="use thinking time instead of n_rollouts")
  parser.add_argument("--c-puct",              default=5.,          action="store",            type=float, help="coefficient of controlling the extent of exploration versus exploitation")
  # training params 
  parser.add_argument("--learning-rate",       default=1e-3,        action="store",            type=float, help="learning rate")
  parser.add_argument("--learning-rate-f",                          action="store",            type=float, help="final learning rate. If specify, exponential decay of learning rate is used.")
  parser.add_argument("--batch-size",          default=512,         action="store",            type=int,   help="mini-batch size for training")
  parser.add_argument("--epochs",              default=50,          action="store",            type=int,   help="number of training steps for each gradient descent update")
  # modes
  parser.add_argument("--train-online",        default=False,       action="store_true",                   help="generate game data and train on those recursively")
  parser.add_argument("--generate-game-data-only", default=False,   action="store_true",                   help="generate game data only without training")
  parser.add_argument("--train-on-game-data-only", default=False,   action="store_true",                   help="train model by game data from directory (only training, no generation)")
  # modes options
  parser.add_argument("--fraction-game-survive", default=0.8,     action="store",            type=float, help="fraction of game survive for next training (used in --train-online)")
  parser.add_argument("--max-game-gen",          default=5000,    action="store",            type=int,   help="maximum number of games generated (used in --train-online and --generate-game-data-only)")
  parser.add_argument("--train-on-last-n-sets",  default=500,     action="store",            type=int,   help="train on the last n recent game (used in --train-online and --train-on-game-data-only)")
  parser.add_argument("--train-every-mins",      default=10.,     action="store",            type=float, help="period (in mins) of performing training (used in --train-on-game-data-only)")
  parser.add_argument("--train-rounds",          default=1000,    action="store",            type=int,   help="number of rounds of training (used in --train-on-game-data-only)")
  # other
  parser.add_argument("--engine",     action="store", default="gpu", type=str,   help="'cpu' for hiding gpu, 'tpu' for colab training.")
  parser.add_argument("--gpu-memory", action="store",                type=float, help="fraction of gpu memory to be used (It may fail if it s less than 1Gb. And the true usage may be higher.)")
  parser.add_argument("--version", action="version", version='%(prog)s ' + __version__)
  args = parser.parse_args()
  args.save_path = args.save_path.format(args.game)

  # Limiting gpu memory
  if args.engine == "gpu" and args.gpu_memory:
    assert args.gpu_memory > 0 and args.gpu_memory <= 1.
    import tensorflow as tf
    opts = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory*0.85) # 0.85 makes the estimation of gpu usage closer to the real usage.
    sess = tf.Session(config=tf.ConfigProto(gpu_options=opts))
  elif args.engine == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
  elif args.engine == "tpu":
    args.train_rounds = 1

  # Main
  a = train_pipeline(args)
  if args.train_online:
    a.train()
  elif args.generate_game_data_only:
    a.get_game_data_parallel()
  elif args.train_on_game_data_only:
    a.train_on_dir()
  
