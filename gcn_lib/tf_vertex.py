# -*- coding: utf-8 -*-
'''
  Project:
    Can GCNs Go as Deep as CNNs?
    https://sites.google.com/view/deep-gcns
    http://arxiv.org/abs/1904.03751
  Author:
    Guohao Li, Matthias Müller, Ali K. Thabet and Bernard Ghanem.
    King Abdullah University of Science and Technology.
'''

import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util

def max_relat_conv_layer(inputs,
                         neigh_idx,
                         nn,
                         k,
                         num_outputs,
                         scope=None,
                         is_training=None):
  '''
    Max relative conv layer:
      Guohao Li, Matthias Müller, Ali K. Thabet and Bernard Ghanem.
      “Can GCNs Go as Deep as CNNs?”
      CoRR abs/1904.03751 (2019)
  '''
  aggr_features = get_max_relat_feature(inputs, neigh_idx, k)
  out = nn.build(aggr_features,
                 num_outputs,
                 scope=scope,
                 is_training=is_training)
  vertex_features = out

  return vertex_features

def get_max_relat_feature(inputs, neigh_idx, k):
  '''
    Aggregate neighbor features for each point with MaxRelativeGCN
    Max relative conv layer:
      Guohao Li, Matthias Müller, Ali K. Thabet and Bernard Ghanem.
      “Can GCNs Go as Deep as CNNs?” CoRR abs/1904.03751 (2019)
    Args:
      inputs: (batch_size, num_vertices, 1, num_dims)
      neigh_idx: (batch_size, num_vertices, k)
      k: int
    Returns:
      aggregated features: (batch_size, num_vertices, 1, 2*num_dims)
  '''
  batch_size = inputs.get_shape().as_list()[0]
  in_copy = inputs
  inputs = tf.squeeze(inputs)
  if batch_size == 1:
    inputs = tf.expand_dims(inputs, 0)

  inputs_central = inputs

  inputs_shape = inputs.get_shape()
  batch_size = inputs_shape[0].value
  num_vertices = inputs_shape[1].value
  num_dims = inputs_shape[2].value

  idx = tf.range(batch_size) * num_vertices
  idx = tf.reshape(idx, [batch_size, 1, 1])

  inputs_flat = tf.reshape(inputs, [-1, num_dims])
  inputs_neighbors = tf.gather(inputs_flat, neigh_idx+idx)
  inputs_central = tf.expand_dims(inputs_central, axis=-2)

  inputs_central = tf.tile(inputs_central, [1, 1, k, 1])

  aggr_neigh_features = tf.reduce_max(inputs_neighbors - inputs_central, axis=-2, keep_dims=True)
  aggr_features = tf.concat([in_copy, aggr_neigh_features], axis=-1)
  return aggr_features

def edge_conv_layer(inputs,
                    neigh_idx,
                    nn,
                    k,
                    num_outputs,
                    scope=None,
                    is_training=None):
  '''
    EdgeConv layer:
      Wang, Y, Yongbin S, Ziwei L, Sanjay S, Michael B, Justin S.
      "Dynamic graph cnn for learning on point clouds."
      arXiv:1801.07829 (2018).
  '''
  edge_features = tf_util.get_edge_feature(inputs, neigh_idx, k)
  out = nn.build(edge_features,
                 num_outputs,
                 scope=scope,
                 is_training=is_training)
  vertex_features = tf.reduce_max(out, axis=-2, keep_dims=True)

  return vertex_features

def graphsage_conv_layer(inputs,
                    neigh_idx,
                    nn,
                    k,
                    num_outputs,
                    normalize=True,
                    scope=None,
                    is_training=None):
  '''
    GraphSage conv layer:
      Hamilton, Will, Zhitao Ying, and Jure Leskovec.
      "Inductive representation learning on large graphs."
      NIPS. 2017.
  '''
  aggr_features = get_graphsage_feature(inputs,
                                   neigh_idx,
                                   k,
                                   nn,
                                   scope=scope+'_aggr',
                                   is_training=is_training)
  out = nn.build(aggr_features,
                 num_outputs,
                 scope=scope,
                 is_training=is_training)
  if normalize == True:
    out = tf.nn.l2_normalize(out, axis=-1)

  vertex_features = out

  return vertex_features

def get_graphsage_feature(inputs,
                     neigh_idx,
                     k,
                     nn,
                     scope=None,
                     is_training=None):
  """
    Aggregate neighbor features for each point with GraphSage
    GraphSage conv layer:
      Hamilton, Will, Zhitao Ying, and Jure Leskovec.
      "Inductive representation learning on large graphs."
      NIPS. 2017.
    Args:
      inputs: (batch_size, num_vertices, 1, num_dims)
      neigh_idx: (batch_size, num_vertices, k)
      k: int
    Returns:
      aggregated features: (batch_size, num_vertices, 1, 2*num_dims)
  """
  batch_size = inputs.get_shape().as_list()[0]
  in_copy = inputs
  inputs = tf.squeeze(inputs)
  if batch_size == 1:
    inputs = tf.expand_dims(inputs, 0)

  inputs_central = inputs

  inputs_shape = inputs.get_shape()
  batch_size = inputs_shape[0].value
  num_vertices = inputs_shape[1].value
  num_dims = inputs_shape[2].value

  idx = tf.range(batch_size) * num_vertices
  idx = tf.reshape(idx, [batch_size, 1, 1])

  inputs_flat = tf.reshape(inputs, [-1, num_dims])
  inputs_neighbors = tf.gather(inputs_flat, neigh_idx+idx)
  neigh_features = nn.build(inputs_neighbors,
                            num_dims,
                            scope=scope,
                            is_training=is_training)
  aggr_neigh_features = tf.reduce_max(neigh_features, axis=-2, keep_dims=True)
  aggr_features = tf.concat([in_copy, aggr_neigh_features], axis=-1)
  return aggr_features

def gin_conv_layer(inputs,
                   neigh_idx,
                   nn,
                   k,
                   num_outputs,
                   zero_epsilon=False,
                   scope=None,
                   is_training=None):
  '''
    GIN conv layer:
      Xu, Keyulu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka.
      "How Powerful are Graph Neural Networks?."
      arXiv:1810.00826 (2018).
  '''
  aggr_features = get_gin_feature(inputs, neigh_idx, k)
  if zero_epsilon == True:
    epsilon = tf.get_variable(scope+'_epsilon',
                              1,
                              dtype=tf.float32,
                              initializer=tf.zeros_initializer,
                              trainable=False)
  else:
    epsilon = tf.get_variable(scope+'_epsilon',
                              1,
                              dtype=tf.float32,
                              initializer=tf.zeros_initializer,
                              trainable=True)

  aggr_features = inputs*(1+epsilon) + aggr_features
  out = nn.build(aggr_features,
                 num_outputs,
                 scope=scope,
                 is_training=is_training)
  vertex_features = out

  return vertex_features

def get_gin_feature(inputs, neigh_idx, k):
  """
    Aggregate neighbor features for each point with GIN
    GIN conv layer:
      Xu, Keyulu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka.
      "How Powerful are Graph Neural Networks?."
      arXiv:1810.00826 (2018).
    Args:
      inputs: (batch_size, num_vertices, 1, num_dims)
      neigh_idx: (batch_size, num_vertices, k)
      k: int
    Returns:
      aggregated features: (batch_size, num_vertices, 1, num_dims)
  """
  batch_size = inputs.get_shape().as_list()[0]
  in_copy = inputs
  inputs = tf.squeeze(inputs)
  if batch_size == 1:
    inputs = tf.expand_dims(inputs, 0)

  inputs_central = inputs

  inputs_shape = inputs.get_shape()
  batch_size = inputs_shape[0].value
  num_vertices = inputs_shape[1].value
  num_dims = inputs_shape[2].value

  idx = tf.range(batch_size) * num_vertices
  idx = tf.reshape(idx, [batch_size, 1, 1])

  inputs_flat = tf.reshape(inputs, [-1, num_dims])
  inputs_neighbors = tf.gather(inputs_flat, neigh_idx+idx)
  neigh_features = inputs_neighbors
  aggr_neigh_features = tf.reduce_sum(neigh_features, axis=-2, keep_dims=True)
  aggr_features = aggr_neigh_features
  return aggr_features
