#!/usr/bin/env python
__file__       = "mcts_id.py"
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
import math
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

class Tree:
  def __init__(self):
    self.root_id = 0
    self.nodes   = {} # a map from id to TreeNode

  def register(self, feature_box_id, parent_id, prior_p):
    if not feature_box_id in self.nodes:
      self.nodes[feature_box_id] = TreeNode()
    self.nodes[feature_box_id].add_parent(parent_id, prior_p)
    return feature_box_id

  def expand(self, leaf_id, policy, Board):
    """
    To create children node
    Input:
      policy: the output[0][:-1] of the predict function in the model class. eg. AlphaZero_Gomoku.predict(,False)[0][:-1]
              policy[0] is a 2D array representing the probability of playing that move on board
              policy[1] is a number   representing the probability of playing "PASS"
    """
    for action in Board.get_legal_action():
      if type(action) == str:
        # 'PASS' move
        children_id = self.register(Board.get_current_player_feature_box_id(action), leaf_id, policy[1])
      else:
        children_id = self.register(Board.get_current_player_feature_box_id(action), leaf_id, policy[0][action])
      self.nodes[leaf_id].add_children(children_id, action)

  def select(self, parent_id, c_puct):
    """
    Select action among children that gives maximum Q+U.
    Output:
      A tuple of (children_id, action tuple)
    """
    children_action = self.nodes[parent_id].children_action
    parent_N        = self.nodes[parent_id].N
    ID_max_QplusU = max(children_action, key=lambda x: self.nodes[x].get_QplusU(c_puct, parent_id, parent_N))
    return (ID_max_QplusU, self.nodes[parent_id].children_action[ID_max_QplusU])

  def update_parents_recursively(self, leaf_id, leaf_value):
    """
      Update myself and all ancestors 
    """
    self.nodes[leaf_id].update(leaf_value)
    for parent_id in self.nodes[leaf_id].parent_prior:
      self.update_parents_recursively(parent_id, -leaf_value)

  def delete_children_recursively(self, leaf_id):
    """
      If the leaf has no parents, delete itself and its nodes.
    """
    if self.nodes[leaf_id].is_root():
      for children_id in self.nodes[leaf_id].children_action:
        self.nodes[children_id].delete_parent(leaf_id)
        self.delete_children_recursively(children_id)
      self.delete_node(leaf_id)

  def delete_node(self, leaf_id):
    del self.nodes[leaf_id]

  def update_root_id(self, next_root_id):
    """
      Here I can delete other children nodes only but not branches to save time because the total number of nodes is not many, in O(10^6) now.
    """
    if not next_root_id in self.nodes:
      self.reset()
      self.nodes[next_root_id] = TreeNode()
      self.root_id             = next_root_id
    else:
      for children_id in self.nodes[self.root_id].children_action:
        self.nodes[children_id].delete_parent(self.root_id)
        if children_id != next_root_id:
          self.delete_children_recursively(children_id)
      self.delete_node(self.root_id)
      self.root_id = next_root_id

  def reset(self):
    self.__init__()

class TreeNode:
  def __init__(self):
    self.parent_prior    = {}      # a map from the parent id to the corresponded prior probability of selecting this node from the parent
    self.children_action = {}      # a map from the children id to the action by which this node can move to the children
    self.N               = 0       # total visit count  (It is used to calculate the mean of action-value)
    self.N_select        = 0       # select count (It is used to calculate the move-select probabilty)
    self.Q               = 0       # mean of action-value

  def add_parent(self, parent_id, prior_p):
    self.parent_prior[parent_id] = prior_p

  def delete_parent(self, parent_id):
    del self.parent_prior[parent_id]

  def add_children(self, children_id, action):
    self.children_action[children_id] = action

  def get_QplusU(self, c_puct, parent_id, parent_N):
    return self.Q + c_puct*self.parent_prior[parent_id]*math.sqrt(parent_N)/(1.+self.N)

  def update(self, leaf_value):
    self.N += 1
    self.Q += (leaf_value - self.Q)/self.N  # incremental mean

  def is_root(self):
    return not self.parent_prior  # self.parent_prior == {}

  def is_leaf(self):
    return not self.children_action # self.children_action == {}

class MCTS:
  def __init__(self, policy_value_fn, c_puct=10., n_rollout=100, s_thinking=None, use_thinking=False):
    """
      Input:
        policy_value_fn : the predict function in the model class. eg. AlphaZero_Gomoku.predict(,False)
    """
    self.policy_value_fn = policy_value_fn
    self.Tree            = Tree()
    self.c_puct          = float(c_puct)
    self.n_rollout       = int(n_rollout)
    self.s_thinking      = s_thinking
    self.use_thinking    = use_thinking

  def rollout(self, Board, epsilon=0.25, dirichlet_param=0.1):
    """
      a rollout from the root node to the leaf node (may or may not be the end of the game)
      CAUTION: This function will modify the input Board. So a copy.deepcopy must be provided.
    """
    node_id = self.Tree.root_id
    while not self.Tree.nodes[node_id].is_leaf():
      # greedily select next move according to Q+U
      node_id, action = self.Tree.select(node_id, self.c_puct)
      self.Tree.nodes[node_id].N_select += 1
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
      # a random dihedral transformation is performed before feeding into AlphaZero
      rotation_order   = np.random.choice(Board.rotation_symmetry)
      reflection_order = np.random.choice(Board.reflection_symmetry)
      feature_box      = Board.get_current_player_feature_box()
      for i in range(len(feature_box)):
        feature_box[i] = self.dihedral_transformation(feature_box[i], rotation_order, reflection_order)
      policy_value     = self.policy_value_fn(np.array([feature_box]), raw_output = False)
      policy           = list(policy_value[0][:-1])
      policy[0]        = self.dihedral_transformation(policy[0], rotation_order, reflection_order, inverse=True)
      # add Dirichlet Noise to encourage exploration
      noise     = np.random.dirichlet(dirichlet_param*np.ones(Board.height*Board.width+1))
      policy[0] = policy[0]*(1.-epsilon) + epsilon*noise[:-1].reshape(Board.height, Board.width)
      policy[1] = policy[1]*(1.-epsilon) + epsilon*noise[-1]
      # expand
      leaf_value = policy_value[0][-1]
      self.Tree.expand(node_id, policy, Board)

    # Update the leaf and its ancestors
    self.Tree.update_parents_recursively(node_id, -leaf_value)

  def dihedral_transformation(self, feature_plane, rotation_order, reflection_order, inverse=False):
    """
      rotation and reflection are not commutative. Here I decided to first perform reflection.
    """
    if not inverse:
      if reflection_order:
        result = np.rot90(np.fliplr(feature_plane), rotation_order)
      else:
        result = np.rot90(feature_plane, rotation_order)
    else:
      if reflection_order:
        result = np.fliplr(np.rot90(feature_plane, -rotation_order))
      else:
        result = np.rot90(feature_plane, -rotation_order)
    return result

  def get_move_probability(self, Board, temp=1., epsilon=0.25, dirichlet_param=0.1):
    """
      Input:
        Board:    current board
        temp :  T to control level of exploration. temp = 1. or high encourages exploration while temp = 1e-3 or small means to select strongest move.
      Output:
        move probability on board
    """
    root_id = Board.get_current_player_feature_box_id()
    if not root_id in self.Tree.nodes:
      self.Tree.reset()
      self.Tree.nodes[root_id] = TreeNode()
    self.Tree.root_id = root_id

    if self.use_thinking:
      start_time = time.time()
      while time.time()-start_time < self.s_thinking:
        Board_deepcopy = copy.deepcopy(Board)
        self.rollout(Board_deepcopy, epsilon, dirichlet_param)
    else:
      for i in range(self.n_rollout):
        Board_deepcopy = copy.deepcopy(Board)
        self.rollout(Board_deepcopy, epsilon, dirichlet_param)

    move        = []
    N_select    = []
    children_id = []
    Q           = []
    for iter_children_id in self.Tree.nodes[self.Tree.root_id].children_action:
      move.append(self.Tree.nodes[self.Tree.root_id].children_action[iter_children_id])
      N_select.append(self.Tree.nodes[iter_children_id].N_select)
      children_id.append(iter_children_id)
      Q.append(self.Tree.nodes[iter_children_id].Q)

    if temp:
      probs = softmax(np.log(N_select)/temp + 1e-9)
    else:
      probs = np.zeros(len(N_select))
      probs[np.argmax(N_select)] = 1.
    return move, probs, children_id, Q

  def update_with_move(self, children_id):
    """
      After the opponent player moves, the child node corresponding to the played action becomes the new root node;
      the subtree below this child is retained along with all its statistics, while the remainder of the tree is discarded
    """
    self.Tree.update_root_id(children_id)

  def reset(self):
    self.Tree.reset()

class MCTS_player:
  def __init__(self, policy_value_fn, c_puct = 5., n_rollout = 100, epsilon = 0.25, dirichlet_param = 0.1, temp = 1., name = "", s_thinking = None, use_thinking = False):
    self.name            = str(name)
    self.policy_value_fn = policy_value_fn
    self.c_puct          = float(c_puct)
    self.n_rollout       = int(n_rollout)
    self.epsilon         = float(epsilon)
    self.dirichlet_param = float(dirichlet_param)
    self.temp            = float(temp)
    self.s_thinking      = float(s_thinking)
    self.use_thinking    = use_thinking
    self.MCTS            = MCTS(self.policy_value_fn, c_puct=self.c_puct, n_rollout=self.n_rollout, s_thinking=self.s_thinking, use_thinking=self.use_thinking)

  def get_move(self, Board, **kwargs):
    """
      epsilon [0,1] is to control how much dirichlet noise is added for exploration. 1 means complete noise.
    """
    epsilon         = float(kwargs.get('epsilon', self.epsilon))
    dirichlet_param = float(kwargs.get('dirichlet_param', self.dirichlet_param))
    is_return_probs = kwargs.get('is_return_probs', False)
    temp            = float(kwargs.get('temp', self.temp))

    if Board.get_legal_action():
      move, probs, children_id, Q = self.MCTS.get_move_probability(Board, temp, epsilon, dirichlet_param)
      selected_move_index = np.random.choice(np.arange(len(move)), p=probs)
      selected_move       = move[selected_move_index]
      selected_move_probs = probs[selected_move_index]
      selected_move_value = Q[selected_move_index]

      self.MCTS.update_with_move(children_id[selected_move_index])

      if is_return_probs:
        return_probs = np.zeros(Board.height*Board.width+1)
        return_Q     = np.zeros(Board.height*Board.width+1)
        for imove, iprobs, iQ in list(zip(move, probs, Q)):
          if imove == "PASS":
            return_probs[-1] = iprobs
            return_Q[-1]     = iQ
          else:
            return_probs[imove[0]*Board.width+imove[1]] = iprobs
            return_Q[imove[0]*Board.width+imove[1]]     = iQ
        return selected_move, return_probs, selected_move_probs, return_Q, selected_move_value
      else:
        return selected_move
    else:
      print("No legal move anymore. It should not happen because the game otherwise ends")

  def update_opponent_move(self, selected_move, children_id):
    """
      selected_action is unused.
    """
    self.MCTS.update_with_move(children_id)

  def reset(self):
    self.MCTS.reset()


