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
import copy

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
        self.nodes[leaf_id].add_children(children_id, action)
      else:
        action = tuple(action)
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
    if not self.nodes[leaf_id].is_root():
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
      Here I delete other children nodes only but not branches to save time because the total number of nodes is not many, in O(10^6) now.
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
      self.N               = 0       # visit count  (It is used to encourage exploration)
      self.N_select        = 0       # select count (It is used to calculate the move-select probabilty)
      self.Q               = 0       # mean of action-value
#      self.Q_var           = 0       # variance of Q times N (not sample variance)

    def add_parent(self, parent_id, prior_p):
      self.parent_prior[parent_id] = prior_p

    def delete_parent(self, parent_id):
      del self.parent_prior[parent_id]

    def add_children(self, children_id, action):
      self.children_action[children_id] = action

    def get_QplusU(self, c_puct, parent_id, parent_N, var_coeff = 1.):
#      return self.Q + var_coeff*np.sqrt(self.Q_var/(self.N+1.)) + c_puct*self.parent_prior[parent_id]*np.sqrt(parent_N)/(1.+self.N)
      return self.Q + c_puct*self.parent_prior[parent_id]*np.sqrt(parent_N)/(1.+self.N)

    def update(self, leaf_value):
      # formula, see http://datagenetics.com/blog/november22017/index.html
      old_Q       = self.Q
      self.N     += 1
      self.Q     += (leaf_value - self.Q)/self.N  # incremental mean
#      self.Q_var += (leaf_value - old_Q)*(leaf_value - self.Q)

    def is_root(self):
      return self.parent_prior == {}

    def is_leaf(self):
      return self.children_action == {}

class MCTS:
  def __init__(self, policy_value_fn, c_puct=10., n_rollout=100):
    """
      Input:
        policy_value_fn : the predict function in the model class. eg. AlphaZero_Gomoku.predict(,False)
    """
    self.policy_value_fn     = policy_value_fn
    self.Tree                = Tree()
    self.c_puct              = float(c_puct)
    self.n_rollout           = int(n_rollout)

  def rollout(self, Board):
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
      policy_value    = self.policy_value_fn(np.array([Board.get_current_player_feature_box()]), raw_output = False)
      policy          = policy_value[0][:-1]
      leaf_value      = policy_value[0][-1]
      self.Tree.expand(node_id, policy, Board)

    # Update the leaf and its ancestors
    self.Tree.update_parents_recursively(node_id, -leaf_value)

  def get_move_probability(self, Board, temp=1.):
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

    for i in range(self.n_rollout):
      Board_deepcopy = copy.deepcopy(Board)
      self.rollout(Board_deepcopy)

    move        = []
    N_select    = []
    children_id = []
    for iter_children_id in self.Tree.nodes[self.Tree.root_id].children_action:
      move.append(self.Tree.nodes[self.Tree.root_id].children_action[iter_children_id])
      N_select.append(self.Tree.nodes[iter_children_id].N_select)
      children_id.append(iter_children_id)

    probs   = softmax(np.log(N_select)/temp + 1e-9)
    return move, probs, children_id

  def update_with_move(self, children_id):
    """
      After the opponent player moves, the child node corresponding to the played action becomes the new root node;
      the subtree below this child is retained along with all its statistics, while the remainder of the tree is discarded
    """
    self.Tree.update_root_id(children_id)

  def reset(self):
    self.Tree.reset()

class MCTS_player:
  def __init__(self, policy_value_fn, c_puct = 10., n_rollout = 100, temp = 1., is_self_play = True, name = ""):
    self.name            = str(name)
    self.policy_value_fn = policy_value_fn
    self.c_puct          = float(c_puct)
    self.n_rollout       = int(n_rollout)
    self.temp            = float(temp)
    self.is_self_play    = is_self_play
    self.MCTS            = MCTS(self.policy_value_fn, c_puct=self.c_puct, n_rollout=self.n_rollout)

  def get_move(self, Board, **kwargs):
    """
      epsilon [0,1] is to control how much dirichlet noise is added for exploration. 1 means complete noise.
    """
    epsilon         = float(kwargs.get('epsilon', 0.25))
    dirichlet_param = float(kwargs.get('dirichlet_param', 0.3))
    is_return_probs = kwargs.get('is_return_probs', False)
    temp            = float(kwargs.get('temp', self.temp))

    if len(Board.get_legal_action()) > 0:
      move, probs, children_id = self.MCTS.get_move_probability(Board, temp)
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

      self.MCTS.update_with_move(children_id[selected_move_index])

      if is_return_probs:
        return_probs = np.zeros(Board.height*Board.width+1)
        for imove, iprobs in list(zip(move, actual_probs)):
          if imove == "PASS":
            return_probs[-1] = iprobs
          else:
            return_probs[imove[0]*Board.width+imove[1]] = iprobs
        return selected_move, return_probs, selected_move_probs
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


