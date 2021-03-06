#!/usr/bin/env python
from __future__ import print_function # to prevent Py2 from interpreting it as a tuple
__file__       = "server.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2018"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2018 Apr 04"

Description=""" game server
"""

#================================================================
# Module
#================================================================
import sys
import numpy as np

#================================================================
# Main
#================================================================
class Human:
  def __init__(self, name=""):
    self.name   = name
    self.nature = "human"

  def get_move(self, Board, **kwargs):
    """
      The function will ask for a move, check whether this move is valid, and then return a valid move
    """
    PASS = "PASS" # So that the user can enter PASS without quotation during 'input'
    legal_action = Board.get_legal_action()
    while True:
      try:
        proposed_move = input(" move is : ")
        try:
          proposed_move = tuple(map(int, proposed_move.split(",")))
        except:
          pass
        if proposed_move in legal_action:
          return proposed_move
        else:
          print("invalid move")
      except KeyboardInterrupt:
        sys.exit("\nUser is terminating program ...")
      except:
        print("invalid move")

  def update_opponent_move(self, *args):
    None

class RandomMove:
  def __init__(self, name=""):
    self.name   = name
    self.nature = "random"

  def get_move(self, Board, **kwargs):
    legal_action = Board.get_legal_action()
    index = np.random.choice(np.arange(len(legal_action)))
    return legal_action[index]

  def update_opponent_move(self, *args):
    None

class Server:
  def __init__(self, Board):
    self.Board = Board

  def start_game(self, player1 = Human(), player2 = Human(), is_gui = True, is_analysis = False):
    """
      Player 1 starts the game.
    """
    # reset
    self.Board.reset()
    try:
      player1.reset()
    except:
      pass
    try:
      player2.reset()
    except:
      pass

    # info
    player        = {1: player1, -1: player2}
    player_number = {1:1, -1:2, 0:0}
    selected_move = None

    if is_gui:
      import gui_pygame as gui
      dualgrid = False
      if self.Board.game in ["go", "gomoku"]:
        dualgrid = True
      Board_gui = gui.Board_gui(self.Board.height, self.Board.width, dualgrid=dualgrid)
      Board_gui.draw_stones(self.Board.state)
    else:
      print("")
      print("Player 1 is %s" % player1.name)
      print("Player 2 is %s" % player2.name)
      print("")
      self.Board.print_state(selected_move)

    while not self.Board.winner[0]:
      if is_gui:
        Board_gui.draw_names(player1.name, player2.name, current_player=self.Board.current_player, winner=self.Board.winner, score=self.Board.score)
      else:
        print("========================================")
        print("Player %i %s ('%s') to move" % (player_number[self.Board.current_player], player[self.Board.current_player].name, self.Board.token[self.Board.current_player]))

      if is_analysis and player[self.Board.current_player].nature == "mcts":
        selected_move, return_probs, selected_move_prob, return_Q, selected_move_value = player[self.Board.current_player].get_move(self.Board, is_return_probs=True, is_analysis=True)
      elif is_gui and player[self.Board.current_player].nature == "human":
        selected_move = Board_gui.asking_for_move(self.Board.get_legal_action())
      else:                           
        selected_move = player[self.Board.current_player].get_move(self.Board)

      self.Board.move(selected_move)
      player[self.Board.current_player].update_opponent_move(selected_move, self.Board.get_current_player_feature_box_id())
      self.Board.check_winner()
      if is_gui:
        Board_gui.draw_stones(self.Board.state)
      else:
        self.Board.print_state(selected_move)

    if is_gui:
      Board_gui.draw_names(player1.name, player2.name, current_player=self.Board.current_player, winner=self.Board.winner, score=self.Board.score)
      Board_gui.freeze()
    else:
      if self.Board.score:
        print("Player1 Score %.1f" % self.Board.score[1])
        print("Player2 Score %.1f" % self.Board.score[-1])
      if self.Board.winner[1] == 0:
        print("It is a draw game !")
      else:
        print("Player %i %s ('%s') wins this game !" % (player_number[self.Board.winner[1]], player[self.Board.winner[1]].name, self.Board.token[self.Board.winner[1]]))

    return player_number[self.Board.winner[1]]

  def start_self_play(self, AI_player, is_shown = True):
    """
      Start a self-play game using 1 MCTS player.
      Use the same search tree for both player(s).
      Store the self-play data: winner, zip(feature, mcts_probs, z)
    """
    # reset
    self.Board.reset()
    AI_player.reset()

    # info
    player        = {1: AI_player, -1: AI_player}
    player_number = {1:1, -1:2, 0:0}
    selected_move = None

    feature_input, policy, turn_player = [], [], []
    if is_shown:
      self.Board.print_state(selected_move)
    while not self.Board.winner[0]:
      if is_shown:
        print("========================================")
        print("Player %i %s ('%s') to move" % (player_number[self.Board.current_player], player[self.Board.current_player].name, self.Board.token[self.Board.current_player]))
        selected_move, return_probs, selected_move_prob, return_Q, selected_move_value = player[self.Board.current_player].get_move(self.Board, is_return_probs=True, is_analysis=True)
      else:
        selected_move, return_probs, selected_move_prob, return_Q, selected_move_value = player[self.Board.current_player].get_move(self.Board, is_return_probs=True)

      # store game state
      feature_input.append(self.Board.get_current_player_feature_box())
      policy.append(return_probs)
      turn_player.append(self.Board.current_player)

      # play
      self.Board.move(selected_move)
      self.Board.check_winner()
      if is_shown:
        self.Board.print_state(selected_move)

    winners_z = np.zeros(len(turn_player))
    if self.Board.winner[1] != 0:
      winners_z[np.array(turn_player) == self.Board.winner[1]] = 1
      winners_z[np.array(turn_player) != self.Board.winner[1]] = -1
    if is_shown:
      if self.Board.score:
        print("Player1 Score %.1f" % self.Board.score[1])
        print("Player2 Score %.1f" % self.Board.score[-1])
      if self.Board.winner[1] == 0:
        print("It is a draw game !")
      else:
        print("Player %i %s ('%s') wins this game !" % (player_number[self.Board.winner[1]], player[self.Board.winner[1]].name, self.Board.token[self.Board.winner[1]]))

    return player_number[self.Board.winner[1]], zip(feature_input, policy, winners_z.reshape(-1,1))

