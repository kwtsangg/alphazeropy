#!/usr/bin/env python
from __future__ import print_function # to prevent Py2 from interpreting it as a tuple
__file__       = "board.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2018"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2018-Apr-03"

Description=""" To make a general reversi game.
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
    To make a connect four game board for two players.
  """
  def __init__(self,
               **kwargs
              ):
    """
       state: 0 means empty, 1 means black/first player with token 'X', -1 means white/second player with token 'O'
    """
    self.width            = 8 if kwargs.get('width') is None else int(kwargs.get('width'))
    self.height           = 8 if kwargs.get('height') is None else int(kwargs.get('height'))
    self.history          = []
    self.winner           = [False, 0]
    self.token            = {1:"X", -1:"O", 0:"."}
    self.score            = {1:2, -1:2}
    self.current_player   = 1
    self.state            = np.zeros(self.height*self.width).reshape(self.height, self.width)
    self.n_feature_plane  = 3
    self.state[int(self.height/2)-1][int(self.width/2)-1] =  1
    self.state[int(self.height/2)-1][int(self.width/2)  ] = -1
    self.state[int(self.height/2)  ][int(self.width/2)-1] = -1
    self.state[int(self.height/2)  ][int(self.width/2)  ] =  1
    self.rotation_symmetry   = [0, 1, 2, 3]
    self.reflection_symmetry = [0, 1]

    # sanity check
    if self.width % 2 != 0 or self.height % 2 != 0:
      raise Exception('board width and height have to be both even number.')

  def get_legal_action(self):
    """
      Output:
        a list of legal move. ie. [ [2,3], [4,5], ... ]
    """
    result = []
    for j in range(self.width):
      for i in range(self.height):
        if self._is_move_legal(i,j):
          result.append((i,j))
    if not result:
      result = ["PASS"]
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
    if action:
      last_state = self.state.copy()
      self.move(action)

    tmp_state = self.state*self.current_player
    tmp_state[tmp_state==-1] = 0
    A = tmp_state

    tmp_state = self.state*self.current_player
    tmp_state[tmp_state==1]  = 0
    tmp_state[tmp_state==-1] = 1
    B = tmp_state

    C = np.zeros(self.height*self.width).reshape(self.height, self.width)
    for legal_action in self.get_legal_action():
      if type(legal_action) != str:
        C[legal_action] = 1

    if action:
      self.state = last_state
      del self.history[-1]
      # Switch current player
      self.current_player *= -1

    return np.array([A,B,C])

  def get_current_player_feature_box_id(self, action = None):
    # The last term is to prevent cyclic tree nodes
    if action is None:
      return hash("%s%i" % (self.get_current_player_feature_box(action).tobytes(), len(self.history)))
    else:
      return hash("%s%i" % (self.get_current_player_feature_box(action).tobytes(), len(self.history)+1))

  def get_last_player(self):
    return -self.current_player

  def get_current_player(self):
    return self.current_player

  def move(self, action):
    """
      ref: http://code.activestate.com/recipes/580698-reversi-othello/
      Input:
        an ordered number [x,y], x:[0,height-1], y:[0,width-1] or PASS
    """
    if type(action) != str:
      self.state[action[0]][action[1]] = self.current_player
      # 8 directions
      dirx = [-1,  0,  1, -1, 1, -1, 0, 1]
      diry = [-1, -1, -1,  0, 0,  1, 1, 1]
      ndim = max(self.height, self.width)
      for d in range(8):
        ctr = 0
        for i in range(ndim):
          x_new = action[0] + dirx[d] * (i+1)
          y_new = action[1] + diry[d] * (i+1)
          if x_new < 0 or x_new >= self.height or y_new < 0 or y_new >= self.width:
            ctr = 0
            break
          elif self.state[x_new][y_new] == self.current_player:
            break
          elif self.state[x_new][y_new] == 0.:
            ctr = 0
            break
          else:
            ctr += 1
        for k in range(ctr):
          x_new = action[0] + dirx[d] * (k+1)
          y_new = action[1] + diry[d] * (k+1)
          self.state[x_new][y_new] = self.current_player
      # Take move and record it down
      self.history.append((action[0], action[1]))
    else:
      self.history.append("PASS")

    # Switch current player
    self.current_player *= -1

  def check_winner(self):
    """
      If both players pass, we can check the score.
    """
    # Calculate scores
    self.score[1]  = len(np.argwhere(self.state == 1))
    self.score[-1] = len(np.argwhere(self.state == -1))

    if self.history:
      if (self.score[1]+self.score[-1]==self.height*self.width) or (type(self.history[-1]) == str and type(self.history[-2]) == str):
        if self.score[1] > self.score[-1]:
          self.winner = [True, 1]
        elif self.score[1] < self.score[-1]:
          self.winner = [True, -1]
        else:
          self.winner = [True, 0]
    return self.winner

  def print_state(self, selected_move = None):
    if selected_move:
      print(self.token[self.get_last_player()], " took a move ", selected_move)
    output = "   "
    for j in range(self.width):
      output += "%2i " % j
    output += "\n"
    print(output)

    for i in range(self.height):
      output = "%2i " % i
      for j in range(self.width):
        if selected_move and i == selected_move[0] and j == selected_move[1]:
          output = output[:-1]
          output += " [%s]" % self.token[self.state[i][j]]
        else:
          output += "%2s " % self.token[self.state[i][j]]
      output += "\n"
      print(output)

  def reset(self):
    self.current_player = 1
    self.history        = []
    self.winner         = [False, 0]
    self.score          = {1:2, -1:2}
    self.state          = np.zeros(self.height*self.width).reshape(self.height, self.width)
    self.state[int(self.height/2)-1][int(self.width/2)-1] =  1
    self.state[int(self.height/2)-1][int(self.width/2)  ] = -1
    self.state[int(self.height/2)  ][int(self.width/2)-1] = -1
    self.state[int(self.height/2)  ][int(self.width/2)  ] =  1

  def _is_move_legal(self, x, y):
    """
      ref: http://code.activestate.com/recipes/580698-reversi-othello/
    """
    if self.state[x][y] != 0:
      return False
    # 8 directions
    dirx = [-1,  0,  1, -1, 1, -1, 0, 1]
    diry = [-1, -1, -1,  0, 0,  1, 1, 1]
    ndim = max(self.height, self.width)
    for d in range(8):
      ctr = 0
      for i in range(ndim):
        x_new = x + dirx[d] * (i+1)
        y_new = y + diry[d] * (i+1)
        if x_new < 0 or x_new >= self.height or y_new < 0 or y_new >= self.width:
          ctr = 0
          break
        if self.state[x_new][y_new] == self.current_player:
          break
        if self.state[x_new][y_new] == 0:
          ctr = 0
          break
        ctr += 1
      if ctr != 0:
        return True
    return False

