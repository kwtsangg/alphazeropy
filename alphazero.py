#!/usr/bin/env python
__file__       = "alphazero.py"
__author__     = "Ka Wa Tsang"
__copyright__  = "Copyright 2018"
__version__    = "1.0.1"
__email__      = "kwtsang@nikhef.nl"
__date__       = "2018-Feb-13"

Description=""" Build a general AlphaZero.
"""

#===============================================================================
#  Module
#===============================================================================
import numpy as np
import os
from datetime import datetime
try:
  import cPickle as pickle
except ImportError:
  import _pickle as pickle

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Activation, BatchNormalization, Dense, Flatten, Input
from tensorflow.python.keras.layers import add
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras import optimizers, regularizers
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras import backend as K
K.set_image_data_format('channels_first')

#================================================================
# Function
#================================================================
def lr_scheduler_wrapper(lr_i, lr_f, total_epochs, mode="linear"):
  if lr_i == lr_f:
    def lr_scheduler(epochs):
      return lr_i
    return lr_scheduler
  elif mode == "linear":
    def lr_scheduler(epochs):
      return lr_i + (lr_f-lr_i)/total_epochs*epochs
    return lr_scheduler
  elif mode == "exp":
    alpha = np.log(lr_f/lr_i)/total_epochs
    def lr_scheduler(epochs):
      return lr_i*np.exp(alpha*epochs)
    return lr_scheduler
  else:
    print("input mode (%s) is not supported." % mode)

#===============================================================================
#  Main
#===============================================================================
class AlphaZero:
  def __init__(self,
            load_path         = None,
            board_height      = 9,
            board_width       = 9,
            n_feature_plane   = 2,
            n_filter          = 9,
            kernel_size_conv  = (1, 1),
            kernel_size_res   = (1, 1),
            n_res_blocks      = 3,
            l2_regularization = 1e-4,
            bn_axis           = 1
          ):
    if load_path:
      self.load_class(load_path)
    else:
      self.board_height      = board_height
      self.board_width       = board_width
      self.n_feature_plane   = n_feature_plane
      self.n_filter          = n_filter             # number of filters
      self.kernel_size_conv  = kernel_size_conv     # kernel size of convolutional layers
      self.kernel_size_res   = kernel_size_res      # kernel size of residual layers
      self.n_res_blocks      = n_res_blocks         # number of residual blocks
      self.l2_regularization = l2_regularization    # a parameter controlling the level of L2 weight regularizatio to prevent overfitting
      self.bn_axis           = bn_axis              # batch normalization axis. For "channel first" = 1, "channel last" = -1.
      self.model             = self.build_model()

  def build_model(self):
    input_data = Input(shape=(self.n_feature_plane, self.board_height, self.board_width))

    # Build for the first convolutional layer
    x = self._build_conv_block(input_data)
    # Build for the residual tower
    for i in range(self.n_res_blocks):
      x = self._build_residual_block(x)
    # Build for the policy output
    policy_out = self._build_policy_block(x)
    # Build for the value output
    value_out  = self._build_value_block(x)
  
    model = Model(inputs=[input_data], outputs=[policy_out, value_out])
    return model

  def _build_conv_block(self, x):
    y = Conv2D(self.n_filter, self.kernel_size_conv, padding='same', kernel_regularizer=regularizers.l2(self.l2_regularization))(x)
    y = BatchNormalization(axis=self.bn_axis)(y)
    y = Activation('relu')(y)
    return y

  def _build_residual_block(self, x):
    y = Conv2D(self.n_filter, self.kernel_size_res, padding='same', kernel_regularizer=regularizers.l2(self.l2_regularization))(x)
    y = BatchNormalization(axis=self.bn_axis)(y)
    y = Activation('relu')(y)
    y = Conv2D(self.n_filter, self.kernel_size_res, padding='same', kernel_regularizer=regularizers.l2(self.l2_regularization))(y)
    y = BatchNormalization(axis=self.bn_axis)(y)
    y = add([x, y])
    y = Activation('relu')(y)
    return y 

  def _build_policy_block(self, x):
    y = Conv2D(2, (1, 1), padding='same', kernel_regularizer=regularizers.l2(self.l2_regularization))(x)
    y = BatchNormalization(axis=self.bn_axis)(y)
    y = Activation('relu')(y)
    y = Flatten()(y)
    y = Dense(self.board_height*self.board_width + 1, kernel_regularizer=regularizers.l2(self.l2_regularization), activation='softmax', name='policy_out')(y) # all board positions + PASS move
    return y
 
  def _build_value_block(self, x):
    y = Conv2D(1, (1, 1), padding='same', kernel_regularizer=regularizers.l2(self.l2_regularization))(x)
    y = BatchNormalization(axis=self.bn_axis)(y)
    y = Activation('relu')(y)
    y = Dense(self.n_filter, kernel_regularizer=regularizers.l2(self.l2_regularization))(y)
    y = Activation('relu')(y)
    y = Flatten()(y)
    y = Dense(1, kernel_regularizer=regularizers.l2(self.l2_regularization), activation='tanh', name='value_out')(y)
    return y

  def predict(self, feature_4Dbox, raw_output = True):
    """
      Input:
        feature_4Dbox : [ b, ... ] where b is a feature_box which have a shape = n_feature_plane, board_height, board_width
      Output:
        If raw_output is False, the output is a list where
        [index of feature_volumns][0] is a 2D array representing the probability of selecting a board position
        [index of feature_volumns][1] is a number   representing the probability of selecting pass
        [index of feature_volumns][2] is a number   representing the reward [-1, 1], 1 means win and -1 means lose
    """
    policy_value = self.model.predict(feature_4Dbox)
    if raw_output:
      return policy_value
    else:
      policy_without_pass   = policy_value[0][:,:-1]
      policy_without_pass   = policy_without_pass.reshape(len(policy_without_pass), self.board_height, self.board_width)

      policy_with_only_pass = policy_value[0][:,-1]
      value                 = policy_value[1].flatten()
      return list(zip(policy_without_pass, policy_with_only_pass, value))

  def train(self, Board_state_array, policy_value_array, learning_rate=1e-1, learning_rate_f=None, epochs=100, batch_size=512):
    if learning_rate_f is None:
      learning_rate_f = learning_rate
    try:
      self.first_train_loop
    except AttributeError:
      self.first_train_loop = True

    if self.first_train_loop:
      myoptimizer = optimizers.Adam(lr=learning_rate)
      self.model.compile(loss={'policy_out': 'kullback_leibler_divergence', 'value_out': 'mean_squared_error'}, optimizer=myoptimizer)
      self.lrate = LearningRateScheduler(lr_scheduler_wrapper(learning_rate, learning_rate_f, epochs, mode="exp"), verbose="1")
      self.first_train_loop = False
    # actual model fit
    self.model.fit(Board_state_array, policy_value_array, epochs=epochs, batch_size=batch_size, callbacks=[self.lrate], verbose=1)

  def save_class(self, name, **kwargs):
    """
      This function will save both the model (architecture + weights + optimizer state) and other information by cPickle
      Input:
        path
        savename
      Output:
        a folder contains both the model.h5 and model.txt
    """
    dir_path = str(kwargs.get('path', os.getcwd()))
    model_no = datetime.today().strftime('%Y%m%d%H%M')
    savename = "%s_%s_board_%i_%i_res_blocks_%i_filters_%i" % (model_no, name, self.board_height, self.board_width, self.n_res_blocks, self.n_filter)
    if not os.path.exists(dir_path+"/"+savename):
      os.makedirs(dir_path+"/"+savename)

    # Save the keras model first
    self.model.save("%s/%s/%s.h5" % (dir_path, savename, savename))

    # Save also the weight
    self.model.save_weights("%s/%s/%s_weight.h5" % (dir_path, savename, savename))

    # Save also the model structure
    model_json = self.model.to_json()
    with open("%s/%s/%s.json" % (dir_path, savename, savename), "w") as json_file:
      json_file.write(model_json)

    # Make a dictionary to save all other information by cPickle
    save_dict = {}
    save_dict["board_height"]      = self.board_height
    save_dict["board_width"]       = self.board_width
    save_dict["n_feature_plane"]   = self.n_feature_plane
    save_dict["n_filter"]          = self.n_filter
    save_dict["kernel_size_conv"]  = self.kernel_size_conv
    save_dict["kernel_size_res"]   = self.kernel_size_res
    save_dict["n_res_blocks"]      = self.n_res_blocks
    save_dict["l2_regularization"] = self.l2_regularization
    save_dict["bn_axis"]           = self.bn_axis

    with open("%s/%s/%s.pkl" % (dir_path, savename, savename), 'w') as Output:
      pickle.dump(save_dict, Output, protocol=2)

    return model_no, dir_path+"/"+savename

  def load_class(self, dir_path, clear_session=True, engine=None):
    savename   = dir_path.split("/")
    if len(savename[-1]) == 0:
      savename = savename[-2]
    else:
      savename = savename[-1]
    if clear_session:
      K.clear_session()

    self.model = load_model('%s/%s.h5' % (dir_path, savename))

    if engine == "tpu":
      TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
      self.model = tf.contrib.tpu.keras_to_tpu_model(
        self.model,
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
        )
      )

    with open('%s/%s.pkl' % (dir_path, savename), 'rb') as Input:
      save_dict = pickle.load(Input)
    self.board_height      = save_dict["board_height"]  
    self.board_width       = save_dict["board_width"]      
    self.n_feature_plane   = save_dict["n_feature_plane"]  
    self.n_filter          = save_dict["n_filter"]     
    self.kernel_size_conv  = save_dict["kernel_size_conv"] 
    self.kernel_size_res   = save_dict["kernel_size_res"]  
    self.n_res_blocks      = save_dict["n_res_blocks"]     
    self.l2_regularization = save_dict["l2_regularization"]
    self.bn_axis           = save_dict["bn_axis"]          

