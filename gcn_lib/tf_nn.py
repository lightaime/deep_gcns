'''
  Project:
    Can GCNs Go as Deep as CNNs?
    https://sites.google.com/view/deep-gcns
    http://arxiv.org/abs/1904.03751
  Author:
    Guohao Li, Matthias Mueller, Ali K. Thabet and Bernard Ghanem.
    King Abdullah University of Science and Technology.
'''
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util

class MLP(object):
  ''' Multi-layer preceptrons for gcn '''
  def __init__(self,
              kernel_size=None,
              stride=None,
              padding=None,
              weight_decay=None,
              bn=None,
              bn_decay=None,
              is_dist=None):
    ''' Define common paramters for every layers '''
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.weight_decay = weight_decay
    self.bn = bn
    self.bn_decay = bn_decay
    self.is_dist = is_dist

  def build(self,
            inputs,
            num_outputs,
            scope=None,
            activation_fn=tf.nn.relu,
            is_training=None):
    ''' Build Multi-layer preceptrons '''
    outputs = tf_util.conv2d(inputs,
                             num_outputs,
                             self.kernel_size,
                             padding=self.padding,
                             stride=self.stride,
                             bn=self.bn,
                             is_training=is_training,
                             weight_decay=self.weight_decay,
                             activation_fn = activation_fn,
                             scope=scope,
                             bn_decay=self.bn_decay,
                             is_dist=self.is_dist)

    return outputs
