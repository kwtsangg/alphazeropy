#!/usr/bin/env python
from __future__ import print_function # to prevent Py2 from interpreting it as a tuple
__file__       = "board.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2018"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2018-Feb-13"

Description=""" To make a general go game (Tromp–Taylor rules).
"""

#===============================================================================
#  Module
#===============================================================================
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#===============================================================================
#  Main
#===============================================================================
class Group:
  def __init__(self):
    self.stone = {} # a map from stone position to the Stone group.

  def add(self, position, color, Board):
    """
      If the move is valid, this function will be called to add the selected move.
      Output:
        number of captured stones
    """
    # safety check:
    # if Board.state[position] != 0:
    #   raise ValueError("The move can be performed on an empty position only.")

    ally_Stone_set, enemy_Stone_set = self.get_nearby_Stone(position, color, Board)
    tmp_Stone = Stone(color)
    tmp_Stone.add(position)

    if ally_Stone_set:
      # Merge ally Stone
      for Stone in ally_Stone_set:
        tmp_Stone.stones = tmp_Stone.stones | Stone.stones
      tmp_Stone.update_liberties(Board)
      for position in tmp_Stones.stones:
        self.stone[position] = tmp_Stones
    else:
      # If no ally group
      self.stone[position] = tmp_Stone
      self.stone[position].update_liberties(Board)
    
    # Update enemy Stone liberties
    n_capture = 0
    for Stone in enemy_Stone_set:
      Stone.liberties.remove(position)
      if not Stone.liberties:
        n_capture += len(Stone.stones)
        self.delete_group(Stone)
    return n_capture

  def check_suicide(self, position, color, Board):
    ally_liberties, enemy_liberties = self.get_liberties(position, color, Board):
    if ally_liberties:
      return False
    else:
      if enemy_liberties:
        return True
      else:
        return False
    
  def get_liberties(self, position, color, Board):
    """
      if an action on position is taken, get the resultant liberties
    """
    ally_liberties  = self.Board._get_liberties(position)
    enemy_liberties = set()
    for neighbor in Board._get_neighbor(position):
      Stone = self.stone.get(neighbor)
        if Stone: # if it is registered, it means either black or white.
          if Stone.color == color:
            ally_liberties  = ally_liberties | Stone.liberties
          else:
            enemy_liberties = enemy_liberties | Stone.liberties
    ally_liberties.remove(position)
    enemy_liberties.remove(position)
    return ally_liberties, enemy_liberties

  def get_nearby_Stone(self, position, color, Board):
    ally_Stone_set  = set()
    enemy_Stone_set = set()
    for neighbor in Board._get_neighbor(position):
      Stone = self.stone.get(neighbor)
        if Stone: # if it is registered, it means either black or white.
          if Stone.color == color:
            ally_Stone_set.add(Stone)
          else:
            enemy_Stone_set.add(Stone)
    return ally_Stone_set, enemy_Stone_set

  def delete_group(self, Stone):
    for position in Stone.stones:
      del self.stone[position]

class Stone:
  def __init__(self, color):
    self.stones    = set() # a set of position of stones
    self.liberties = set() # a set of position of liberties
    self.color     = color

  def add(self, position):
    self.stones.add(position)

  def update_liberties(self, Board):
    """
      It is a full update of liberties
    """
    self.liberties = set()
    for position in self.stones:
      self.liberties = self.liberties | Board._get_liberties(position)
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
    self.score            = {1:0, -1:self.komi}
    self.current_player   = 1
    self.state            = np.zeros(self.height*self.width).reshape(self.height, self.width)
    self.group            = Group()
    self.n_pass_disable   = 0 if kwargs.get('n_pass_disable') is None else int(kwargs.get('n_pass_disable'))

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
    if len(self.history) > self.n_pass_disable or not result:
      result.append["PASS"]
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
    if action is not None:
      last_state = self.state
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

    if action is not None:
      self.state = last_state
      del self.history[-1]
      # Switch current player
      self.current_player = 1 if self.current_player == -1 else -1

    D = self.current_player*self.komi*np.ones(self.height*self.width).reshape(self.height, self.width)

    return np.array([A,B,C,D])

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
    print("")

  def reset(self):
    self.current_player = 1
    self.state          = np.zeros(self.height*self.width).reshape(self.height, self.width)
    self.history        = []
    self.winner         = [False, 0]

  def _is_move_legal(self, x, y):
    if not suicide or ko
      
  def _get_neighbor(self, position):
    neighbor = set()
    for i, j in [ [0,-1], [0,1], [1,0], [-1,0] ]:
      x_new = position[0] + i
      if x_new < 0 or x_new > self.height:
        break
      y_new = position[1] + j
      if y_new < 0 or y_new > self.width:
        break
      neighbor.add((x_new, y_new))
    return neighbor

  def _get_liberties(self, position):
    liberties = set()
    for neighbor in self._get_neighbor(position):
      if self.state[neighbor] == 0:
        liberties.add(neighbor)
    return liberties

