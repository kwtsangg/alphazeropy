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
import numpy as np

#================================================================
# Main
#================================================================
class Human:
  def __init__(self, name=""):
    self.name  = name

  def get_move(self, Board, **kwargs):
    """
      The function will ask for a move, check whether this move is valid, and then return a valid move
    """
    proposed_move = input(" move is : ")
    try:
      proposed_move = tuple(map(int, proposed_move.split(",")))
    except:
      pass
    if not proposed_move in Board.get_legal_action():
      print("invalid move")
      return self.get_move(Board)
    else:
      return proposed_move

  def update_opponent_move(self, *args):
    None

class Server:
  def __init__(self, Board):
    self.Board = Board

  def start_game(self, player1 = Human(), player2 = Human(), is_shown = True):
    """
      Player 1 starts the game.
    """
    self.Board.reset()
    player        = {1: player1, -1: player2}
    player_number = {1:1, -1:2}
    selected_move = None

    if is_shown:
      print("Player 1 is %s" % player1.name)
      print("Player 2 is %s" % player2.name)
      print("")

    while not self.Board.winner[0]:
      if is_shown:
        print("Player %i %s ('%s') to move" % (player_number[self.Board.current_player], player[self.Board.current_player].name, self.Board.token[self.Board.current_player]))
        self.Board.print_state(selected_move)
      selected_move = player[self.Board.current_player].get_move(self.Board)
      self.Board.move(selected_move)
      player[self.Board.current_player].update_opponent_move(selected_move, self.Board.get_current_player_feature_box_id())
      self.Board.check_winner()

    if is_shown:
      self.Board.print_state(selected_move)
      if self.Board.winner[1] == 0:
        print("It is a draw game !")
      else:
        print("Player %i %s ('%s') wins this game !" % (player_number[self.Board.winner[1]], player[self.Board.winner[1]].name, self.Board.token[self.Board.winner[1]]))

    return self.Board.winner[1]

  def start_self_play(self, AI_player, is_shown = True, temp_trans_after = 30):
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
    player_number = {1:1, -1:2}
    selected_move = None

    feature_input, policy, current_players = [], [], []
    if is_shown:
      self.Board.print_state(selected_move)
    while not self.Board.winner[0]:
      if len(self.Board.history) > temp_trans_after:
        selected_move, return_probs, selected_move_prob, return_Q, selected_move_value = player[self.Board.current_player].get_move(self.Board, is_return_probs=True, temp=0.)
      else:
        selected_move, return_probs, selected_move_prob, return_Q, selected_move_value = player[self.Board.current_player].get_move(self.Board, is_return_probs=True)

      # store game state
      feature_input.append(self.Board.get_current_player_feature_box())
      policy.append(return_probs)
      current_players.append(self.Board.current_player)

      # play
      self.Board.move(selected_move)
      self.Board.check_winner()
      if is_shown:
        print("The value at the move is")
        print(return_Q[:-1].reshape(self.Board.height, self.Board.width))
        print("The resultant policy is")
        print(return_probs[:-1].reshape(self.Board.height, self.Board.width))
        print("The value of the chosen move is       ", selected_move_value)
        print("The probabilty of chosen this move is ", selected_move_prob)
        self.Board.print_state(selected_move)

    winners_z = np.zeros(len(current_players))
    if self.Board.winner[1] != 0:
      winners_z[np.array(current_players) == self.Board.winner[1]] = 1
      winners_z[np.array(current_players) != self.Board.winner[1]] = -1
    if is_shown:
      if self.Board.winner[1] == 0:
        print("It is a draw game !")
      else:
        print("Player %i %s ('%s') wins this game !" % (player_number[self.Board.winner[1]], player[self.Board.winner[1]].name, self.Board.token[self.Board.winner[1]]))

    return self.Board.winner[1], zip(feature_input, policy, winners_z.reshape(-1,1))

