#!/usr/bin/env python
from __future__ import print_function # to prevent Py2 from interpreting it as a tuple
__file__       = "play.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2018"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2018 Apr 04"

Description=""" A platform to play or evaluate games.
"""

#================================================================
# Module
#================================================================
import numpy as np
import os, sys
import argparse, textwrap
from datetime import datetime

from server import Server, Human

#================================================================
# Functions
#================================================================
def inverse_logistic(b, P, c_elo=1./400.):
  """
    get the elo of player b given the elo of player a and the probability of (a defeats b)
  """
  return int(b-np.log(1./P-1.)/c_elo)

def find_model(evaluate_model_path, with_elo=True, with_latest=True):
  # with_latest\with_elo  |     True        |    False
  #                 True  | Latest_with_elo | Latest_without_elo
  #                 False |   strongest     |     ---
  current_model_elo = -1.
  current_model_no  = 0
  current_model_dir = ""
  try: # Python2
    model_dir_list = os.walk(evaluate_model_path).next()[1]
  except: # Python3
    model_dir_list = next(os.walk(evaluate_model_path))[1]
  for model_dir in model_dir_list:
    if with_latest:
      try:
        model_no = int(model_dir.split("_")[0])
        if model_no > current_model_no:
          if with_elo:
            try:
              model_elo = np.loadtxt("%s/%s/elo.txt" % (evaluate_model_path, model_dir))
              if model_elo > current_model_elo:
                current_model_elo = model_elo
              else:
                continue
            except:
              continue
          current_model_no  = int(model_dir.split("_")[0])
          current_model_dir = model_dir
      except:
        pass
    else:
      try:
        model_elo = np.loadtxt("%s/%s/elo.txt" % (evaluate_model_path, model_dir))
        if model_elo > current_model_elo:
          current_model_elo = model_elo
          current_model_no  = int(model_dir.split("_")[0])
          current_model_dir = model_dir
      except:
        pass

  if current_model_no == 0:
    if with_latest:
      if with_elo:
        raise ValueError("Cannot find the latest possible player with elo")
      else:
        raise ValueError("Cannot find the latest possible player without elo")
    else:
      raise ValueError("Cannot find the strongest player with elo")
  return current_model_elo, current_model_no, "%s/%s" % (evaluate_model_path, current_model_dir)

#================================================================
# Main
#================================================================
class platform:
  def __init__(self, args):
    # evaluate params
    self.evaluate            = args.evaluate
    self.evaluate_game       = int(args.evaluate_game/2)*2
    self.evaluate_model_path = args.evaluate_model_path

    # board params
    self.game             = args.game
    if self.game is None:
      raise ValueError("Please let me know which game you want to play by --game")

    self.board_height     = args.board_height
    self.board_width      = args.board_width
    self.n_in_row         = args.n_in_row

    # player1, AI/human brain params
    self.p1_temp         = args.p1_temp
    self.p1_n_rollout    = args.p1_n_rollout
    self.p1_epsilon      = args.p1_epsilon
    self.p1_dir_param    = args.p1_dir_param
    self.p1_s_thinking   = args.p1_s_thinking
    self.p1_use_thinking = args.p1_use_thinking
    self.p1_c_puct       = args.p1_c_puct

    self.p1_brain_path  = args.p1_brain
    self.p1_name        = args.p1_name

    if self.p1_brain_path is None:
      self.p1 = Human(name=self.p1_name)
    else:
      from alphazero import AlphaZero
      from mcts_cyclic_ref import MCTS_player
      self.p1_brain = AlphaZero()
      self.p1_brain.load_class(self.p1_brain_path, False)
      self.p1_name = self.p1_brain_path.split("/")[-1]
      if not self.p1_name:
        self.p1_name = self.p1_brain_path.split("/")[-2]
      self.p1 = MCTS_player(self.p1_brain.predict, c_puct = self.p1_c_puct, n_rollout = self.p1_n_rollout, epsilon = self.p1_epsilon, dirichlet_param = self.p1_dir_param, temp = self.p1_temp, name = "AlphaZero "+self.p1_name, s_thinking = self.p1_s_thinking, use_thinking = self.p1_use_thinking)
      print("Overwriting board size according to trained model (player 1) ...")
      self.board_height = self.p1_brain.board_height
      self.board_width  = self.p1_brain.board_width

    # player2, AI/human brain params
    self.p2_temp         = args.p2_temp
    self.p2_n_rollout    = args.p2_n_rollout
    self.p2_epsilon      = args.p2_epsilon
    self.p2_dir_param    = args.p2_dir_param
    self.p2_s_thinking   = args.p2_s_thinking
    self.p2_use_thinking = args.p2_use_thinking
    self.p2_c_puct       = args.p2_c_puct

    self.p2_brain_path  = args.p2_brain
    self.p2_name        = args.p2_name

    if self.p2_brain_path is None:
      self.p2 = Human(name=self.p2_name)
    else:
      from alphazero import AlphaZero
      from mcts_cyclic_ref import MCTS_player
      self.p2_brain     = AlphaZero()
      self.p2_brain.load_class(self.p2_brain_path, False)
      self.p2_name = self.p2_brain_path.split("/")[-1]
      if not self.p2_name:
        self.p2_name = self.p2_brain_path.split("/")[-2]
      self.p2 = MCTS_player(self.p2_brain.predict, c_puct = self.p2_c_puct, n_rollout = self.p2_n_rollout, epsilon = self.p2_epsilon, dirichlet_param = self.p2_dir_param, temp = self.p2_temp, name = "AlphaZero "+self.p2_name, s_thinking = self.p2_s_thinking, use_thinking = self.p2_use_thinking)
      print("Overwriting board size according to trained model (player 2) ...")
      self.board_height = self.p2_brain.board_height
      self.board_width  = self.p2_brain.board_width

    # Initialize board
    sys.path.insert(0, '%s/' % self.game)
    from board import Board
    self.Board  = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
    self.server = Server(self.Board)

    # Save the Board params in case they are None (This part maybe not be necessary because they wont be used anymore)
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

    # Other
    self.analysis = args.analysis

  # Main modes
  def start_game(self):
    self.server.start_game(player1=self.p1, player2=self.p2, is_gui=True, is_analysis=self.analysis)

  def start_evaluation(self):
    print("Evaluating ... ")
    # model_score[0] : [win, lose, draw] as player 1
    # model_score[1] : [win, lose, draw] as player 2
    try:
      print("Loading the past score ...")
      model_score = np.loadtxt("%s/evaluate.txt" % self.p1_brain_path).T
    except:
      model_score = np.zeros(6).reshape(2,3)

    # Save the opponent
    np.savetxt("%s/evaluate_opponent.txt" % self.p1_brain_path, [self.p2_brain_path], fmt="%s")

    # print summary
    def print_summary():
      print("\nSummary: %i/%i" % (np.sum(model_score), self.evaluate_game))
      print("      play as p1, play as p2")
      print("win   %11i %11i" % (model_score[0][0], model_score[1][0]))
      print("lose  %11i %11i" % (model_score[0][1], model_score[1][1]))
      print("draw  %11i %11i" % (model_score[0][2], model_score[1][2]))
      print("total %11i %11i" % (sum(model_score[0]), sum(model_score[1])))
    print_summary()

    half_total_game = int(self.evaluate_game // 2)
    # play as player 1
    while np.sum(model_score[0]) < half_total_game:
      print("\nGenerating gameplay %i/%i ..." % (np.sum(model_score), self.evaluate_game))
      winner = self.server.start_game(player1=self.p1, player2=self.p2, is_gui=False, is_analysis=self.analysis)
      if winner == 1:
        model_score[0][0] += 1.
      elif winner == 0:
        model_score[0][2] += 1.
      else:
        model_score[0][1] += 1.

      np.savetxt("%s/evaluate.txt" % self.p1_brain_path, model_score.T, header="play as p1, play as p2")
      print_summary()

    # play as player 2
    while np.sum(model_score[1]) < half_total_game:
      print("\nGenerating gameplay %i/%i ..." % (np.sum(model_score), self.evaluate_game))
      winner = self.server.start_game(player1=self.p2, player2=self.p1, is_gui=False, is_analysis=self.analysis)
      if winner == 2:
        model_score[1][0] += 1.
      elif winner == 0:
        model_score[1][2] += 1.
      else:
        model_score[1][1] += 1.

      np.savetxt("%s/evaluate.txt" % self.p1_brain_path, model_score.T, header="play as p1, play as p2")
      print_summary()

    prob = (np.sum(model_score.T[0]) + 0.5*model_score.T[2])/float(self.evaluate_game)
    if prob == 1.:
      print("The evaluation fails because its opponent is too weak.")
      return 0

    elo_player2 = np.loadtxt("%s/elo.txt" % self.p2_brain_path)
    elo_player1 = inverse_logistic(elo_player2, prob)
    print("The win rate of the model candidate is %.2f %%, which corresponds to elo of %i" % (prob*100., elo_player1))
    comment="%s The elo of the model (%s, n_rollout %i, s_thinking %i s, use_thinking %i) is evaluated against the model (%s, n_rollout %i, s_thinking %i s, use_thinking %i) which has an elo of %i" % (datetime.today().strftime('%Y%m%d%H%M'), self.p1_brain_path, self.p1_n_rollout, self.p1_s_thinking, self.p1_use_thinking, self.p2_brain_path, self.p2_n_rollout, self.p2_s_thinking, self.p2_use_thinking, elo_player2)
    print(comment)
    np.savetxt("%s/elo.txt" % self.p1_brain_path, [elo_player1], fmt="%i", header=comment)
    return elo_player1

if __name__ == "__main__":
  class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass
  parser = argparse.ArgumentParser(description=textwrap.dedent(Description), prog=__file__, formatter_class=CustomFormatter)
  # player params
  parser.add_argument("--p1-brain",                          action="store",       type=str,   help="player1, directory path that store AI brain (empty means human player)")
  parser.add_argument("--p1-name",         default="Alice",  action="store",       type=str,   help="player1, name")
  parser.add_argument("--p2-brain",                          action="store",       type=str,   help="player2, directory path that store AI brain (empty means human player)")
  parser.add_argument("--p2-name",         default="Bob",    action="store",       type=str,   help="player2, name")
  # board params
  parser.add_argument("--game",                              action="store",       type=str,   help="gomoku, connectfour, reversi")
  parser.add_argument("--board-height",                      action="store",       type=int,   help="height of the board")
  parser.add_argument("--board-width",                       action="store",       type=int,   help="width of the board")
  parser.add_argument("--n-in-row",                          action="store",       type=int,   help="needed if game is gomoku or connectfour")
  # AI brain params
  parser.add_argument("--p1-temp",         default=0.,       action="store",       type=float, help="player1, temperature to control how greedy of selecting next action")
  parser.add_argument("--p1-n-rollout",    default=400,      action="store",       type=int,   help="player1, number of simulations for each move")
  parser.add_argument("--p1-epsilon",      default=0.25,     action="store",       type=float, help="player1, fraction of noise in prior probability")
  parser.add_argument("--p1-dir-param",    default=0.1,      action="store",       type=float, help="player1, extent of encouraging exploration")
  parser.add_argument("--p1-s-thinking",   default=1,        action="store",       type=int,   help="player1, time allowed to think for each move (in seconds)")
  parser.add_argument("--p1-use-thinking", default=False,    action="store_true",              help="player1, use thinking time instead of n_rollouts")
  parser.add_argument("--p1-c-puct",       default=5.,       action="store",       type=float, help="player1, coefficient of controlling the extent of exploration versus exploitation")
  parser.add_argument("--p2-temp",         default=0.,       action="store",       type=float, help="player2, temperature to control how greedy of selecting next action")
  parser.add_argument("--p2-n-rollout",    default=400,      action="store",       type=int,   help="player2, number of simulations for each move")
  parser.add_argument("--p2-epsilon",      default=0.25,     action="store",       type=float, help="player2, fraction of noise in prior probability")
  parser.add_argument("--p2-dir-param",    default=0.1,      action="store",       type=float, help="player2, extent of encouraging exploration")
  parser.add_argument("--p2-s-thinking",   default=1,        action="store",       type=int,   help="player2, time allowed to think for each move (in seconds)")
  parser.add_argument("--p2-use-thinking", default=False,    action="store_true",              help="player2, use thinking time instead of n_rollouts")
  parser.add_argument("--p2-c-puct",       default=5.,       action="store",       type=float, help="player2, coefficient of controlling the extent of exploration versus exploitation")
  # evaluator
  parser.add_argument("--evaluate",        default=False,    action="store_true",              help="get the elo of model (--p1-brain) against model (--p2-brain)")
  parser.add_argument("--evaluate-game",   default=100,      action="store",       type=int,   help="number of games used in getting elo")
  parser.add_argument("--evaluate-model-path", default="%s/{}_training_model" % os.getcwd(), action="store", type=str, help="directory where models are saved")
  # other
  parser.add_argument("--analysis",        default=False,    action="store_true",              help="if MCTS_player, show the value of the chosen move")
  parser.add_argument("--version", action="version", version='%(prog)s ' + __version__)
  args = parser.parse_args()
  args.evaluate_model_path = args.evaluate_model_path.format(args.game)

  # If evaluate
  if args.evaluate:
    if args.p1_brain:
      # Check whether it has been evaluated
      try:
        np.loadtxt("%s/elo.txt" % args.p1_brain)
        raise ValueError("The model (--p1-brain %s) has been evaluated" % args.p1_brain)
      except:
        pass
    else:
      print("Looking for the latest possible player without elo in %s as the challenger" % args.evaluate_model_path)
      p1_elo, p1_model_no, args.p1_brain = find_model(args.evaluate_model_path, with_elo = False, with_latest = True)
      print("Model %s is loaded into p1 brain as the challenger" % p1_model_no)


    # Check whether it has an opponent
    try:
      args.p2_brain = str(np.loadtxt("%s/evaluate_opponent.txt" % args.p1_brain, dtype=str))
      print("A past opponent (%s) is found. Overwritting the p2 brain ..." % args.p2_brain)
    except:
      if args.p2_brain:
        try:
          np.loadtxt("%s/elo.txt" % args.p2_brain)
        except:
          raise ValueError("the model (--p2-brain) has not been evaluated, so it cannot be used to evaluate other model.")
      else:
        print("Looking for the strongest possible player with elo in %s as the opponent" % args.evaluate_model_path)
        p2_elo, p2_model_no, args.p2_brain = find_model(args.evaluate_model_path, with_elo = True, with_latest = False)
        print("Model %s with elo %i is loaded into p2 brain as the opponent" % (p2_model_no, p2_elo))
  else:
    if args.p1_brain == "latest":
      print("Looking for the latest possible player in %s" % args.evaluate_model_path)
      p1_elo, p1_model_no, args.p1_brain = find_model(args.evaluate_model_path, with_elo = False, with_latest = True)
      print("Model %s is loaded into p1 brain" % p1_model_no)
    elif args.p1_brain == "strongest":
      print("Looking for the strongest player in %s" % args.evaluate_model_path)
      p1_elo, p1_model_no, args.p1_brain = find_model(args.evaluate_model_path, with_elo = True, with_latest = False)
      print("Model %s is loaded into p1 brain" % p1_model_no)
    if args.p2_brain == "latest":
      print("Looking for the latest possible player in %s" % args.evaluate_model_path)
      p2_elo, p2_model_no, args.p2_brain = find_model(args.evaluate_model_path, with_elo = False, with_latest = True)
      print("Model %s is loaded into p2 brain" % p2_model_no)
    elif args.p2_brain == "strongest":
      print("Looking for the strongest player in %s" % args.evaluate_model_path)
      p2_elo, p2_model_no, args.p2_brain = find_model(args.evaluate_model_path, with_elo = True, with_latest = False)
      print("Model %s is loaded into p2 brain" % p2_model_no)

  a = platform(args)
  if args.evaluate:
    a.start_evaluation()
  else:
    a.start_game()

