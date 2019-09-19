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
import math
import time
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util

class Model(object):
  """ Build model """
  def __init__(self,
            batch_size,
            num_vertices,
            num_layers,
            num_neighbors,
            num_filters,
            num_classes,
            vertex_layer_builder=None,
            edge_layer_builder=None,
            mlp_builder=None,
            skip_connect=None,
            dilations=None):
    print("#"*100)
    print("Building model {} {} {} with {} layers".format(skip_connect,
                                                         vertex_layer_builder.layer.__name__,
                                                         edge_layer_builder.layer.__name__,
                                                         num_layers))

    inputs_ph, labels_ph, is_training_ph = self.get_placeholders(batch_size, num_vertices)

    self.mlp_builder = mlp_builder
    self.inputs = inputs_ph
    self.labels = labels_ph
    self.is_training = is_training_ph
    graphs = self.build_gcn_backbone_block(self.inputs,
                                           vertex_layer_builder,
                                           edge_layer_builder,
                                           num_layers,
					   num_neighbors,
					   num_filters,
                                           skip_connect,
                                           dilations)
    fusion = self.build_fusion_block(graphs, num_vertices)
    self.pred = self.build_mlp_pred_block(fusion, num_classes)

    print("Done!!!")
    print("#"*100)

  def get_placeholders(self, batch_size, num_vertices):
    inputs_ph = tf.placeholder(tf.float32,
                               shape=(batch_size, num_vertices, 9))
    labels_ph = tf.placeholder(tf.int32,
                               shape=(batch_size, num_vertices))
    is_training_ph = tf.placeholder(tf.bool, shape=())
    return inputs_ph, labels_ph, is_training_ph

  def build_gcn_backbone_block(self,
                               input_graph,
                               vertex_layer_builder,
                               edge_layer_builder,
                               num_layers,
			       num_neighbors,
			       num_filters,
                               skip_connect,
                               dilations):
    '''Build the gcn backbone block'''
    input_graph = tf.expand_dims(input_graph, -2)
    graphs = []

    for i in range(num_layers):
      if i == 0:
        neigh_idx = edge_layer_builder.build(input_graph[:, :, :, 6:],
                                            num_neighbors[i],
                                            dilation=dilations[i],
                                            is_training=self.is_training)

        vertex_features  = vertex_layer_builder.build(input_graph,
                                                      num_neighbors[i],
                                                      num_filters[i],
                                                      neigh_idx=neigh_idx,
                                                      scope='adj_conv_'+str(i),
                                                      is_training=self.is_training)
        graph = vertex_features
        graphs.append(graph)
      else:
        neigh_idx = edge_layer_builder.build(graphs[-1],
                                            num_neighbors[i],
                                            dilation=dilations[i],
                                            is_training=self.is_training)
        vertex_features  = vertex_layer_builder.build(graphs[-1],
                                                      num_neighbors[i],
                                                      num_filters[i],
                                                      neigh_idx=neigh_idx,
                                                      scope='adj_conv_'+str(i),
                                                      is_training=self.is_training)
        graph = vertex_features
        if skip_connect == 'residual':
          graph = graph + graphs[-1]
        elif skip_connect == 'dense':
          graph = tf.concat([graph, graphs[-1]], axis=-1)
        elif skip_connect == 'none':
          graph = graph
        else:
          raise Exception('Unknown connections')
        graphs.append(graph)

    return graphs

  def build_fusion_block(self, graphs, num_vertices):
    out = self.mlp_builder.build(tf.concat(graphs, axis=-1),
                                 1024,
                                 scope='adj_conv_'+'final',
                                 is_training=self.is_training)
    out_max = tf_util.max_pool2d(out, [num_vertices, 1], padding='VALID', scope='maxpool')
    expand = tf.tile(out_max, [1, num_vertices, 1, 1])
    fusion = tf.concat(axis=3, values=[expand]+graphs)

    return fusion

  def build_mlp_pred_block(self, fusion, num_classes):
    self.mlp_builder.bn_decay = None
    out = self.mlp_builder.build(fusion,
                                 512,
                                 scope='seg/conv1',
                                 is_training=self.is_training)
    out = self.mlp_builder.build(out,
                                 256,
                                 scope='seg/conv2',
                                 is_training=self.is_training)
    out = tf_util.dropout(out,
                          keep_prob=0.7,
                          scope='dp1',
                          is_training=self.is_training)
    self.mlp_builder.bn = False
    out = self.mlp_builder.build(out,
                                 num_classes,
                                 scope='seg/conv3',
                                 activation_fn=None)
    pred = tf.squeeze(out, [2])

    return pred

  def get_loss(self, pred, label):
    """ pred: B,N,num_classes; label: B,N """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    return tf.reduce_mean(loss)
