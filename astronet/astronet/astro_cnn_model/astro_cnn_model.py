# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A model for classifying light curves using a convolutional neural network.

See the base class (in astro_model.py) for a description of the general
framework of AstroModel and its subclasses.

The architecture of this model is:


                                     predictions
                                          ^
                                          |
                                       logits
                                          ^
                                          |
                                (fully connected layers)
                                          ^
                                          |
                                   pre_logits_concat
                                          ^
                                          |
                                    (concatenate)

              ^                           ^                          ^
              |                           |                          |
   (convolutional blocks 1)  (convolutional blocks 2)   ...          |
              ^                           ^                          |
              |                           |                          |
     time_series_feature_1     time_series_feature_2    ...     aux_features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sys

from astronet.astro_model import astro_model

import numpy as np
import matplotlib.pyplot as plt

class AstroCNNModel(astro_model.AstroModel):
  """A model for classifying light curves using a convolutional neural net."""

  def __init__(self, features, labels, hparams, mode):
    """Basic setup. The actual TensorFlow graph is constructed in build().

    Args:
      features: A dictionary containing "time_series_features" and
          "aux_features", each of which is a dictionary of named input Tensors.
          All features have dtype float32 and shape [batch_size, length].
      labels: An int64 Tensor with shape [batch_size]. May be None if mode is
          tf.estimator.ModeKeys.PREDICT.
      hparams: A ConfigDict of hyperparameters for building the model.
      mode: A tf.estimator.ModeKeys to specify whether the graph should be built
          for training, evaluation or prediction.

    Raises:
      ValueError: If mode is invalid.
    """
    super(AstroCNNModel, self).__init__(features, labels, hparams, mode)

  def _build_cnn_layers(self, inputs, hparams, scope="cnn"):
    """Builds convolutional layers.

    The layers are defined by convolutional blocks with pooling between blocks
    (but not within blocks). Within a block, all layers have the same number of
    filters, which is a constant multiple of the number of filters in the
    previous block. The kernel size is fixed throughout.

    Args:
      inputs: A Tensor of shape [batch_size, length].
      hparams: Object containing CNN hyperparameters.
      scope: Name of the variable scope.

    Returns:
      A Tensor of shape [batch_size, output_size], where the output size depends
      on the input size, kernel size, number of filters, number of layers,
      convolution padding type and pooling.
    """
    def store_inputs(x):
      #print('the input shape is:', x.shape)
      #print('  values:', x[0,:,0])
      np.save('/tmp/input', x[0,:,0])
      return x

    def store_conv_block1_output(x):
      layer_num = 1 if x.shape[1]==201 else 2
      #print('the conv{}_block1 output shape is: {}'.format(layer_num, x.shape))
      #print('  values:', x[0,:,:])
      if layer_num==1:
        np.save('/tmp/convl1b1', x[0,:,:])
      else:
        np.save('/tmp/convl2b1', x[0,:,:])
      return x

    def store_conv_block2_output(x):
      layer_num = 1 if x.shape[1]==201 else 2
      #print('the conv{}_block2 output shape is: {}'.format(layer_num, x.shape))
      #print('  values:', x[0,:,:])
      if layer_num==1:
        np.save('/tmp/convl1b2', x[0,:,:])
      else:
        np.save('/tmp/convl2b2', x[0,:,:])
      return x

    def store_pooling_output(x):
      block_num = 1 if x.shape[1]==98 else 2
      #print('the pooling{} output shape is: {}'.format(block_num, x.shape))
      #print('  values:', x[0,:,:])
      if block_num==1:
        np.save('/tmp/pool1', x[0,:,:])
      else:
        np.save('/tmp/pool2', x[0,:,:])
      return x

    def store_flattened_output(x):
      block_num = 1 if x.shape[1]==98*16 else 2
      #print('the flattened{} output shape is: {}'.format(block_num, x.shape))
      #print('  values:', x[0,:,:])
      if block_num==1:
        np.save('/tmp/flat1', x[0,:])
      else:
        np.save('/tmp/flat2', x[0,:])
      return x

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      net = tf.expand_dims(inputs, -1)  # [batch, length, channels]
      #net = tf.identity(net, name="cnn_input")
      #print('\nCNN input: {}'.format(net))
      #net = tf.Print(net, [net.shape],
      #              '\ninput to the CNN: ')
      net_shape = [val if val is not None else -1 for val in net.get_shape().as_list()]
      net = tf.reshape(tf.py_func(store_inputs, [net], [tf.float32])[0],net_shape)

      for i in range(hparams.cnn_num_blocks):
        num_filters = int(hparams.cnn_initial_num_filters *
                          hparams.cnn_block_filter_factor**i)
        #print('\n----- Convolution block {} -----'.format(i+1))
        #print('number of filters for block_{}: {}'.format(i+1, num_filters))
        with tf.variable_scope("block_%d" % (i + 1), reuse=tf.AUTO_REUSE):
          for j in range(hparams.cnn_block_size):
            net = tf.layers.conv1d(
                inputs=net,
                filters=num_filters,
                kernel_size=int(hparams.cnn_kernel_size),
                padding=hparams.convolution_padding,
                activation=tf.nn.relu,
                name="conv_%d" % (j + 1))
            
            net_shape = [val if val is not None else -1 for val in net.get_shape().as_list()]
            if j==0:
              net = tf.reshape(tf.py_func(store_conv_block1_output, [net], [tf.float32])[0],net_shape)
            elif j==1:
              net = tf.reshape(tf.py_func(store_conv_block2_output, [net], [tf.float32])[0],net_shape)
            #print('CNN 1D layer{}: {}'.format(j+1, net))
            #net = tf.Print(net, [net[0,:,0].shape, net[0,:,0]],
            #        '\nCNN 1D layer{} values: '.format(j+1))
            
          if hparams.pool_size > 1:  # pool_size 0 or 1 denotes no pooling
            net = tf.layers.max_pooling1d(
                inputs=net,
                pool_size=int(hparams.pool_size),
                strides=int(hparams.pool_strides),
                name="pool")
            #print('max pooling{}: {}'.format(j+1, net))
            net_shape = [val if val is not None else -1 for val in net.get_shape().as_list()]
            net = tf.reshape(tf.py_func(store_pooling_output, [net], [tf.float32])[0],net_shape)

      # Flatten.
      net.get_shape().assert_has_rank(3)
      net_shape = net.get_shape().as_list()
      output_dim = net_shape[1] * net_shape[2]
      net = tf.reshape(net, [-1, output_dim], name="flatten")
      #print('\nCNN output layer: {}'.format(net))
      #net = tf.Print(net, [net],
      #              '\nOutput of convolution block{} = '.format(i+1),
      #              name='conv_out{}'.format(i+1))    
      net_shape = [val if val is not None else -1 for val in net.get_shape().as_list()]
      net = tf.reshape(tf.py_func(store_flattened_output, [net], [tf.float32])[0],net_shape)

    return net

  def build_time_series_hidden_layers(self):
    """Builds hidden layers for the time series features.

    Inputs:
      self.time_series_features

    Outputs:
      self.time_series_hidden_layers
    """
    time_series_hidden_layers = {}
    for name, time_series in self.time_series_features.items():
      time_series_hidden_layers[name] = self._build_cnn_layers(
          inputs=time_series,
          hparams=self.hparams.time_series_hidden[name],
          scope=name + "_hidden")

    self.time_series_hidden_layers = time_series_hidden_layers
