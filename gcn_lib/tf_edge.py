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

def knn_graph(vertex_features,
              k,
              distance_metric=tf_util.pairwise_distance):
  '''
    Find the neighbors' indices based on knn
  '''
  dists = distance_metric(vertex_features)
  neigh_idx = tf_util.knn(dists, k=k) # (batch, num_points, k)

  return neigh_idx

def dilated_knn_graph(vertex_features,
                      k,
                      distance_metric=tf_util.pairwise_distance,
                      dilation=1,
                      stochastic=False,
                      epsilon=0.0,
                      is_training=None):
  '''
    Find the neighbors' indices based on dilated knn
  '''
  dists = distance_metric(vertex_features)
  neigh_idx = tf_util.knn(dists, k=k*dilation)
  neigh_idx = dilated(neigh_idx,
                     k,
                     dilation=dilation,
                     stochastic=stochastic,
                     epsilon=epsilon,
                     is_training=is_training)

  return neigh_idx

def dilated(neigh_idx, k, dilation=1, stochastic=False, epsilon=0.0, is_training=None):
  """ Sample stochastic dilated indices
  Args:
    neigh_idx: vertices' indecies # (batch, num_vertices, k*dilation)
    k: k neigbours
    dilation: dilation rate
    epsilon: random dilated rate
    is_training: tf.bool
    stochastic: bool

  Returns:
    neigh_idx: dilated indices # (batch, num_vertices, k)
  """
  if stochastic:
    def fn1():
      idxs = tf.range(tf.shape(neigh_idx)[2])
      ridxs = tf.random_shuffle(idxs)[:k]
      neigh_idx_1 = tf.gather(neigh_idx, ridxs, axis=2)
      return tf.identity(neigh_idx_1)
    def fn2():
      neigh_idx_2 = neigh_idx[:, :, ::dilation]
      return tf.identity(neigh_idx_2)
    cond = tf.random_uniform([1, 1], minval=0.0, maxval=1.0, dtype=tf.float32)
    epsilon = tf.constant(epsilon)
    neigh_idx = tf.cond(tf.math.logical_and(tf.less(cond[0][0], epsilon), is_training), fn1, fn2)
    return neigh_idx
  else:
    neigh_idx = neigh_idx[:, :, ::dilation]
    return neigh_idx
