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
class VertexLayer(object):
  '''
    Wrapper of GCN's vertex functions
  '''
  def __init__(self, layer, nn, k, num_outputs):
    self.layer = layer
    self.nn = nn
    self.k = k
    self.num_outputs = num_outputs

  def build(self, inputs, neigh_idx, scope=None, is_training=None):
     vertex_features = self.layer(inputs,
                                  neigh_idx,
                                  self.nn,
                                  self.k,
                                  self.num_outputs,
                                  scope=scope,
                                  is_training=is_training)
     return vertex_features

class EdgeLayer(object):
  '''
    Wrapper of GCN's edge functions
  '''
  def __init__(self, layer, k, distance_metric):
    self.layer = layer
    self.k = k
    self.distance_metric = distance_metric

  def build(self, inputs, dilation=None, is_training=None):
    if self.layer.__name__ == 'knn_graph':
      neigh_idx = self.layer(inputs,
                             self.k,
                             distance_metric=self.distance_metric)
    elif self.layer.__name__ == 'dilated_knn_graph':
      neigh_idx = self.layer(inputs,
                             self.k,
                             distance_metric=self.distance_metric,
                             dilation=dilation,
                             is_training=is_training)
    else:
      raise Exception('Unknown layer function {}'.format(self.layer.__name__))

    return neigh_idx
