#!/usr/bin/env python
from __future__ import print_function # to prevent Py2 from interpreting it as a tuple
__file__       = "board.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2018"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2018-Feb-13"

Description=""" To make a general go game using TrompTaylor rules.
"""

#===============================================================================
#  Module
#===============================================================================
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import copy

#===============================================================================
#  Main
#===============================================================================
class Stone:
  def __init__(self, color):
    self.stones    = set() # a set of position tuple of stones
    self.liberties = set() # a set of position tuple of liberties
    self.color     = color

  def add(self, position):
    self.stones.add(position)

  def update_liberties(self, Board):
    """
      It is a full update of liberties
    """
    self.liberties = set()
    for position in self.stones:
      for neighbor in Board.get_neighbor(position):
        if not Board.state[neighbor]:
          self.liberties.add(neighbor)
    return self.liberties

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
    self.komi             = 7.5 if kwargs.get('komi') is None else int(kwargs.get('komi'))
    self.history          = []
    self.winner           = [False, 0]
    self.token            = {1:"X", -1:"O", 0:"."}
    self.score            = {1:0., -1:self.komi}
    self.current_player   = 1
    self.n_pass_disable   = 0 if kwargs.get('n_pass_disable') is None else int(kwargs.get('n_pass_disable'))
    self.state            = np.zeros(self.height*self.width).reshape(self.height, self.width)

    self.group = {} # a map from stone position to the Stone group.

  def get_legal_action(self):
    """
      Output:
        a list of legal move. ie. [ [2,3], [4,5], ... ]
    """
    result = []
    for j in range(self.width):
      for i in range(self.height):
        if self.is_move_legal(i,j):
          result.append((i,j))
    if len(self.history) > self.n_pass_disable or not result:
      result.append("PASS")
    return result

  def get_current_player_feature_box(self, action = None):
    """
      Implicit assumption: n_feature_plane = 4
      Output:
        a feature box for training CNN having a shape of (n_feature_plane, height, width). It returns [A,B,C].
          A. current  player stone with 1 and others 0
          B. opponent player stone with 1 and others 0
          C. available action with 1 and others 0
          D. constant layer to show the advantage/disadvantage, eg. komi, of the turn player.
    """
    if action:
      last_state = self.state
      last_group = copy.deepcopy(self.group)
      last_score = self.score
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

    D = -self.current_player*np.ones(self.height*self.width).reshape(self.height, self.width)

    if action:
      self.state = last_state
      self.group = last_group
      self.score = last_score
      del self.history[-1]
      # Switch current player
      self.current_player = 1 if self.current_player == -1 else -1

    return np.array([A,B,C,D])

  def get_current_player_feature_box_id(self, action = None):
    # The last term is to prevent cyclic tree nodes
    if action:
      return hash("%s%i" % (self.get_current_player_feature_box(action).tobytes(), len(self.history)+1))
    else:
      return hash("%s%i" % (self.get_current_player_feature_box(action).tobytes(), len(self.history)))

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
    capture_position = self.add(action)
    for pos in capture_position:
      self.state[pos] = 0
    self.score[self.current_player] += len(capture_position)
    self.state[action[0]][action[1]] = self.current_player
    self.history.append((action[0], action[1]))

    # Switch current player
    self.current_player = 1 if self.current_player == -1 else -1

  def check_winner(self):
    """
      Using the last move to check whether the game has a winner
    """
    if len(self.history) > 1:
      if type(self.history[-1]) == str and type(self.history[-2]) == str:
        final_score = np.sum(self.state)
        if final_score > 0:
          self.winner = [True, 1]
        elif final_score < 0:
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
    print("")

  def reset(self):
    self.current_player = 1
    self.state          = np.zeros(self.height*self.width).reshape(self.height, self.width)
    self.history        = []
    self.winner         = [False, 0]
    self.score          = {1:0, -1:self.komi}
    self.group          = {}

  def is_move_legal(self, x, y):
    if self.state[x][y]:
      return False
    elif self.check_suicide((x,y)):
      return False
    else:
      return True
      
  #================================================================
  # Group function
  #================================================================
  
  def check_suicide(self, position):
    for pos in self.get_neighbor(position):
      if not self.state[pos]:
        return False
    ally_Stone_set, enemy_Stone_set = self.get_nearby_Stone(position)
    for ally_Stone in ally_Stone_set:
      if (ally_Stone.liberties - set(position)):
        return False
    for enemy_Stone in enemy_Stone_set:
      if not (enemy_Stone.liberties - set(position)):
        return False
    return True

  def check_simple_ko(self, position, Board):
    pass
    
  def add(self, position):
    """
      Assumed the move is valid, this function will be called to add the selected move to self.group (a dict of Stone).
      Output:
        number of captured stones
    """
    tmp_Stone = Stone(self.current_player)
    tmp_Stone.add(position)
    ally_Stone_set, enemy_Stone_set = self.get_nearby_Stone(position)
    # Merge ally Stone if any
    for ally_Stone in ally_Stone_set:
      tmp_Stone.stones = tmp_Stone.stones | ally_Stone.stones
    tmp_Stone.update_liberties(self)
    for pos in tmp_Stone.stones:
      self.group[pos] = tmp_Stone

    # Update enemy Stone liberties
    capture_set = set()
    for enemy_Stone in enemy_Stone_set:
      enemy_Stone.liberties.remove(position)
      if not enemy_Stone.liberties:
        capture_set = capture_set | enemy_Stone.stones
        self.delete_group(enemy_Stone)
    return capture_set

  def delete_group(self, Stone):
    for position in Stone.stones:
      del self.group[position]

  def get_nearby_Stone(self, position):
    ally_Stone_set  = set()
    enemy_Stone_set = set()
    for neighbor in self.get_neighbor(position):
      Stone = self.group.get(neighbor)
      if Stone: # if it is registered, it means either black or white.
        if Stone.color == self.current_player:
          ally_Stone_set.add(Stone)
        else:
          enemy_Stone_set.add(Stone)
    return ally_Stone_set, enemy_Stone_set

  def get_neighbor(self, position):
    neighbor = set()
    dirx = [0, 1, 0, -1]
    diry = [1, 0, -1, 0]
    for d in range(4):
      x_new = position[0] + dirx[d]
      if x_new < 0 or x_new >= self.height:
        continue
      y_new = position[1] + diry[d]
      if y_new < 0 or y_new >= self.width:
        continue
      neighbor.add((x_new, y_new))
    return neighbor

