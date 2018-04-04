#!/usr/bin/env python
from __future__ import print_function # to prevent Py2 from interpreting it as a tuple
__file__       = "game.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2018"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2018-Feb-13"

Description=""" To make a general gomoku game.
"""

#===============================================================================
#  Module
#===============================================================================
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#===============================================================================
#  Main
#===============================================================================
class Board:
  """
    To make a gomoku game board for two players.
  """
  def __init__(self,
          width  = 9,
          height = 9,
          **kwargs
          ):
    """
       state: 0 means empty, 1 means black/first player with token 'X', -1 means white/second player with token 'O'
    """
    self.width            = int(width)
    self.height           = int(height)
    self.n_in_row         = 5 if kwargs.get('n_in_row') is None else int(kwargs.get('n_in_row'))
    self.history          = []
    self.winner           = [False, 0]
    self.token            = {1:"X", -1:"O", 0:"."}
    self.current_player   = 1
    self.state            = np.zeros(self.height*self.width).reshape(self.height, self.width)

    # sanity check
    if self.width < self.n_in_row or self.height < self.n_in_row:
      raise Exception('board width and height cannot be less than %d' % self.n_in_row)

  def get_legal_action(self):
    """
      Output:
        a list of legal move. ie. [ [2,3], [4,5], ... ]
    """
    result = list(np.argwhere(self.state==0))
    return result

  def get_current_player_feature_box(self, action = None):
    """
      Implicit assumption: n_feature_plane = 3
      Output:
        a feature box for training CNN having a shape of (n_feature_plane, height, width). It returns [A,B,C].
          A. current  player stone with 1 and others 0
          B. opponent player stone with 1 and others 0
          C. available action with 1 and others 0
          # D. constant layer to show the advantage/disadvantage, eg. komi, of the turn player.
    """
    if action is not None:
      self.move(action)

    tmp_state = self.state*self.current_player
    tmp_state[tmp_state==-1] = 0
    A = tmp_state

    tmp_state = self.state*self.current_player
    tmp_state[tmp_state==1]  = 0
    tmp_state[tmp_state==-1] = 1
    B = tmp_state

    tmp_state = np.abs(self.state)
    C = 1-tmp_state

    if action is not None:
      self.undo_move(action)

    return np.array([A,B,C])

  def get_current_player_feature_box_id(self, action = None):
    return hash(self.get_current_player_feature_box(action).tobytes())

  def get_last_player(self):
    return -self.current_player

  def get_current_player(self):
    return self.current_player

  def move(self, action):
    """
      Input:
        an ordered number [x,y], x:[0,height-1], y:[0,width-1]
    """
    # Take move and record it down
    self.state[action[0]][action[1]] = self.current_player
    self.history.append((action[0], action[1]))

    # Switch current player
    self.current_player = 1 if self.current_player == -1 else -1

  def undo_move(self, action):
    self.state[action[0]][action[1]] = 0
    del self.history[-1]
    # Switch current player
    self.current_player = 1 if self.current_player == -1 else -1

  def check_winner(self):
    """
      Using the last move to check whether the game has a winner
    """
    if len(self.history) > 0:
      last_move       = self.history[-1]
      last_player     = self.get_last_player()

      connected_token = last_player*self.n_in_row
      # check for row
      if connected_token in [sum(self.state[last_move[0]][j:j+self.n_in_row]) for j in range(0, self.width-self.n_in_row+1, 1)]:
        self.winner = [True, last_player]
        return self.winner

      # check for column
      if connected_token in [sum(self.state.T[last_move[1]][i:i+self.n_in_row]) for i in range(0, self.height-self.n_in_row+1, 1)]:
        self.winner = [True, last_player]
        return self.winner

      # check for diagonal with slope 1
      diagonal = np.diag(self.state, last_move[1]-last_move[0])
      if connected_token in [sum(diagonal[i:i+self.n_in_row]) for i in range(0, len(diagonal)-self.n_in_row+1, 1)]:
        self.winner = [True, last_player]
        return self.winner

      # check for diagonal with slope -1
      diagonal = np.diag(self.state[:,::-1], self.width-1-last_move[1]-last_move[0])
      if connected_token in [sum(diagonal[i:i+self.n_in_row]) for i in range(0, len(diagonal)-self.n_in_row+1, 1)]:
        self.winner = [True, last_player]
        return self.winner

      # check for draw game
      if len(np.argwhere(self.state==0)) == 0:
        self.winner = [True, 0]
        return self.winner
    return self.winner

  def print_state(self, selected_move = None):
    if selected_move is not None:
      print(self.token[self.get_last_player()], " took a move ", selected_move)
    output = "   "
    for j in range(self.width):
      output += "%2i " % j
    output += "\n"
    print(output)

    for i in range(self.height):
      output = "%2i " % i
      for j in range(self.width):
        if selected_move is not None and i == selected_move[0] and j == selected_move[1]:
          output = output[:-1]
          output += " [%s]" % self.token[self.state[i][j]]
        else:
          output += "%2s " % self.token[self.state[i][j]]
      output += "\n"
      print(output)

  def reset(self):
    self.current_player = 1
    self.state          = np.zeros(self.height*self.width).reshape(self.height, self.width)
    self.history        = []
    self.winner         = [False, 0]

