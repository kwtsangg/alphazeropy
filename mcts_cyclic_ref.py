#!/usr/bin/env python
from __future__ import print_function # to prevent Py2 from interpreting it as a tuple
__file__       = "mcts_cyclic_ref.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2018"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2018-Feb-15"

Description=""" To make MCTS by PUCT algorithm
"""

#===============================================================================
#  Module
#===============================================================================
import numpy as np
import sys
sys.setrecursionlimit(15000)
import copy
import time

#===============================================================================
#  Functions
#===============================================================================
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#===============================================================================
#  Main
#===============================================================================

class TreeNode:
    def __init__(self, parent, prior_p):
      self.parent      = parent  # previous TreeNode
      self.children    = {}      # a map from action to TreeNode
      self.N           = 0       # visit count
      self.Q           = 0       # mean action-value
      self.P           = prior_p # prior probability of selecting this node from parent
#      self.best_child_value = 0

    def select(self, c_puct):
      """
      Select action among children that gives maximum Q+U.
      Output:
        A tuple of (action, children_node)
      """
      return max(self.children.items(), key=lambda node: node[1].get_QplusU(c_puct))

    def get_QplusU(self, c_puct):
      return self.Q + self.get_U(c_puct)

    def get_U(self, c_puct):
      return c_puct*self.P*np.sqrt(self.parent.N)/(1.+self.N)

    def expand(self, policy, legal_action):
      """
      To create children node
      Input:
        policy: the output[0][:-1] of the predict function in the model class. eg. AlphaZero_Gomoku.predict(,False)[0][:-1]
                policy[0] is a 2D array representing the probability of playing that move on board
                policy[1] is a number   representing the probability of playing "PASS"
      """
      for action in legal_action:
        if type(action) == str:
          # 'PASS' move
          if action not in self.children:
            self.children[action] = TreeNode(self, policy[1])
        else:
          if action not in self.children:
            self.children[action] = TreeNode(self, policy[0][action])

    def update(self, leaf_value):
      self.N += 1
      self.Q += (leaf_value - self.Q)/self.N

    def update_parent_recursively(self, leaf_value):
      """
        Update myself and all ancestors 
      """
      if not self.is_root():
        self.parent.update_parent_recursively(-leaf_value)
      self.update(leaf_value)

    def is_root(self):
      return not self.parent

    def is_leaf(self):
      return not self.children

class MCTS:
  def __init__(self, policy_value_fn, c_puct=10., n_rollout=100, s_thinking=None, use_thinking=False):
    """
      Input:
        policy_value_fn : the predict function in the model class. eg. AlphaZero_Gomoku.predict(,False)
    """
    self.policy_value_fn   = policy_value_fn
    self.root_node         = TreeNode(None, 1.0)
    self.c_puct            = float(c_puct)
    self.n_rollout         = int(n_rollout)
    self.thinking_time     = thinking_time
    self.use_thinking_time = use_thinking_time
    self.s_thinking        = s_thinking
    self.use_thinking      = use_thinking

  def rollout(self, Board):
    """
      a rollout from the root node to the leaf node (may or may not be the end of the game)
      CAUTION: This function will modify the input Board. So a copy.deepcopy must be provided.
    """
    first_player = Board.current_player
    node         = self.root_node

    while not node.is_leaf():
      # greedily select next move according to Q+U
      action, node = node.select(self.c_puct)
      Board.move(action)
    # check whether the game ends
    Board.check_winner()
    if Board.winner[0]:
      if Board.winner[1] == 0:
        # if draw game
        leaf_value = 0.
      else:
        leaf_value = 1. if Board.winner[1] == Board.current_player else -1.
    else:
      policy_value    = self.policy_value_fn(np.array([Board.get_current_player_feature_box()]), raw_output = False)
      policy          = policy_value[0][:-1]
      leaf_value      = policy_value[0][-1]
      node.expand(policy, Board.get_legal_action())

    # Update the leaf and its ancestors
    node.update_parent_recursively(-leaf_value)

  def get_move_probability(self, Board, temp=1.):
    """
      Input:
        Board:    current board
        temp :  T to control level of exploration. temp = 1. or high encourages exploration while temp = 1e-3 or small means to select strongest move.
      Output:
        move probability on board
    """
    if self.use_thinking:
      start_time = time.time()
      while time.time()-start_time < self.s_thinking:
        Board_deepcopy = copy.deepcopy(Board)
        self.rollout(Board_deepcopy)
    else:
      for i in range(self.n_rollout):
        Board_deepcopy = copy.deepcopy(Board)
        self.rollout(Board_deepcopy)

    move_N_Q   = [(move, node.N, node.Q) for move, node in self.root_node.children.items()] # transform a dictionary to tuple
    move, N, Q = list(zip(*move_N_Q)) # unzip the tuple into move and N
    if temp:
      probs = softmax(np.log(N)/temp + 1e-9)
    else:
      probs = np.zeros(len(N))
      probs[np.argmax(N)] = 1.
    return move, probs, Q

  def update_with_move(self, last_move):
    """
      After the opponent player moves, the child node corresponding to the played action becomes the new root node;
      the subtree below this child is retained along with all its statistics, while the remainder of the tree is discarded
    """
    last_move = tuple(last_move)
    if last_move in self.root_node.children:
      self.root_node        = self.root_node.children[last_move]
      self.root_node.parent = None
    else:
      self.root_node = TreeNode(None, 1.0)

  def reset(self):
    self.root_node = TreeNode(None, 1.0)

class MCTS_player:
  def __init__(self, policy_value_fn, c_puct = 10., n_rollout = 100, temp = 1., is_self_play = True, name = "", s_thinking = None, use_thinking = False):
    self.name            = str(name)
    self.token           = None
    self.policy_value_fn = policy_value_fn
    self.c_puct          = float(c_puct)
    self.n_rollout       = int(n_rollout)
    self.temp            = float(temp)
    self.is_self_play    = is_self_play
    self.s_thinking      = s_thinking
    self.use_thinking    = use_thinking
    self.MCTS            = MCTS(self.policy_value_fn, c_puct=self.c_puct, n_rollout=self.n_rollout, s_thinking=self.s_thinking, use_thinking=self.use_thinking)

  def get_move(self, Board, **kwargs):
    """
      epsilon [0,1] is to control how much dirichlet noise is added for exploration. 1 means complete noise.
    """
    epsilon         = float(kwargs.get('epsilon', 0.25))
    dirichlet_param = float(kwargs.get('dirichlet_param', 0.3))
    is_return_probs = kwargs.get('is_return_probs', False)
    temp            = float(kwargs.get('temp', self.temp))

    if Board.get_legal_action():
      move, probs, Q = self.MCTS.get_move_probability(Board, temp)
      if self.is_self_play:
        # add Dirichlet Noise for exploration (needed for self-play training)  
        actual_probs  = probs*(1.-epsilon) + epsilon*np.random.dirichlet(dirichlet_param*np.ones(len(probs)))
        # sometimes, if move[0] = 'PASS', the code crashes. So use selected_move_index to avoid it.
        #selected_move = np.random.choice(move, p=probs)
        selected_move_index = np.random.choice(np.arange(len(move)), p=actual_probs)
        selected_move = move[selected_move_index]
        selected_move_probs = actual_probs[selected_move_index]
      else:
        #selected_move = np.random.choice(move, p=probs)
        actual_probs = probs
        selected_move_index = np.random.choice(np.arange(len(move)), p=actual_probs)
        selected_move = move[selected_move_index]
        selected_move_probs = actual_probs[selected_move_index]

      self.MCTS.update_with_move(selected_move)

      if is_return_probs:
        return_probs = np.zeros(Board.height*Board.width+1)
        return_Q     = np.zeros(Board.height*Board.width+1)
        for imove, iprobs, iQ in list(zip(move, actual_probs, Q)):
          if imove == "PASS":
            return_probs[-1] = iprobs
            return_Q[-1]     = iQ
          else:
            return_probs[imove[0]*Board.width+imove[1]] = iprobs
            return_Q[imove[0]*Board.width+imove[1]]     = iQ
        return selected_move, return_probs, selected_move_probs, return_Q, Q[selected_move_index]
      else:
        return selected_move

    else:
      print("No legal move anymore. It should not happen because the game otherwise ends")

  def update_opponent_move(self, opponent_last_move, children_id=None):
    """
      children_id is unused but needed.
    """
    self.MCTS.update_with_move(opponent_last_move)

  def reset(self):
    self.MCTS.reset()


