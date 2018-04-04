import numpy as np

#===============================================================================
#  Function being tested
#===============================================================================


def get_dihedral_game_data(game_data_output):
  """
    Because there are symmetries of the game board, this funcion genereates new game data set by rotations and reflections.
  """
  state_result_list  = []
  policy_result_list = []
  value_result_list  = []
  for one_training_set in game_data_output:
    # state has a shape of (n_feature_plane, height, weight)
    # policy has a shape of (height * weight + 1, )
    # value has a shape of (1,)
    state  = one_training_set[0]
    policy = one_training_set[1]
    value  = one_training_set[2]

    n_feature_plane, height, width = state.shape
    if height == width:
      rotation = [0, 1, 2, 3]
    else:
      rotation = [0, 2]

    for i in rotation:
      # rotate the state to generate new game data set
      state_result_list.append( [ np.rot90(state[j], i) for j in xrange(n_feature_plane) ] )
      tmp_policy = np.rot90(policy[:-1].reshape(height, width), i).reshape(-1,)
      policy_result_list.append( np.append(tmp_policy, policy[-1]) )
      value_result_list.append( value )

      # reflect the state horizontally before rotation
      state_result_list.append( [ np.rot90(np.fliplr(state[j]), i) for j in xrange(n_feature_plane) ] )
      tmp_policy = np.rot90(np.fliplr(policy[:-1].reshape(height, width)), i).reshape(-1,)
      policy_result_list.append( np.append(tmp_policy, policy[-1]) )
      value_result_list.append( value )
  return state_result_list, policy_result_list, value_result_list

#===============================================================================
#  Main
#===============================================================================

feature_input = np.array( [np.arange(9).reshape(3,3)] )
policy        = np.arange(10) 
value         = np.array([1])

game_data = [[feature_input, policy, value]]
a,b,c = get_dihedral_game_data(game_data)

for ia in a:
  print ia

