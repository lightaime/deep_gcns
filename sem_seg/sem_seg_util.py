'''
  Utility functions parsing args.
  Project:
    Can GCNs Go as Deep as CNNs?
    https://sites.google.com/view/deep-gcns
    http://arxiv.org/abs/1904.03751
  Author:
    Guohao Li, Matthias Mueller, Ali K. Thabet and Bernard Ghanem.
    King Abdullah University of Science and Technology.
'''

import argparse
import provider
import numpy as np

def add_bool_arg(parser, name, default=False):
  group = parser.add_mutually_exclusive_group(required=False)
  group.add_argument('--' + name, dest=name, action='store_true')
  group.add_argument('--no-' + name, dest=name, action='store_false')
  parser.set_defaults(**{name:default})

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset (s3dis, vkitti) [default: s3dis]')
  parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
  parser.add_argument('--model', type=str, default='model', help='Model file')
  parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
  parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint to continue')

  parser.add_argument('--tower_name', type=str, default='tower', help='Tower name [default: tower]')
  parser.add_argument('--num_gpu', type=int, default=2, help='The number of GPUs to use [default: 2]')
  parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training for each GPU [default: 8]')
  parser.add_argument('--num_points', type=int, default=4096, help='Points number [default: 4096]')
  parser.add_argument('--num_layers', type=int, default=28, help='GCN_layers number [default: 28]')
  parser.add_argument('--num_classes', type=int, default=13, help='Classes number [default: 13]')
  parser.add_argument('--max_epoch', type=int, default=151, help='Epoch to run [default: 150]')
  parser.add_argument('--optimizer', default='adam', help='Adam or momentum [default: adam]')
  parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
  parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
  parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
  parser.add_argument('--bn_init_decay', type=float, default=0.5, help='Initial decay rate for bn decay [default: 0.5]')
  parser.add_argument('--bn_decay_decay_rate', type=float, default=0.5, help='BN decay rate for bn decay [default: 0.5]')
  parser.add_argument('--bn_decay_decay_step', type=int, default=300000, help='BN decay rate decay step [default: 300000]')
  parser.add_argument('--bn_decay_clip', type=float, default=0.99, help='BN decay clip [default: 0.99]')

  parser.add_argument('--k', type=int, default=16, help='K the number of k nearest neighbors [Default: 16]')
  add_bool_arg(parser, 'stochastic_dilation', default=True)
  parser.add_argument('--sto_dilated_epsilon', type=float, default=0.2, help='Stochastic probability of dilatioin [Default: 0.2]')
  parser.add_argument('--skip_connect', type=str, default='residual', help='Skip Connections (residual, dense, none) [default: residual]')
  parser.add_argument('--edge_lay', type=str, default='dilated', help='The type of edge layers (dilated, knn) [default: dilated]')
  parser.add_argument('--gcn_num_filters', type=int, default=64, help='The num of filers in gcn layers [default: 64]')
  parser.add_argument('--gcn', type=str, default='edgeconv', help='The type of GCN layers (mrgcn, edgeconv, graphsage, gin) [default: edgeconv]')
  add_bool_arg(parser, 'normalize_sage')
  add_bool_arg(parser, 'zero_epsilon_gin')

  return parser.parse_args()


def load_data(all_files, room_filelist, test_area_idx):
  # Load all data
  data_batch_list = []
  label_batch_list = []
  for h5_filename in all_files:
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
  data_batches = np.concatenate(data_batch_list, 0)
  label_batches = np.concatenate(label_batch_list, 0)

  test_area = 'Area_'+test_area_idx
  train_idxs = []
  test_idxs = []
  for i,room_name in enumerate(room_filelist):
    if test_area in room_name:
      test_idxs.append(i)
    else:
      train_idxs.append(i)

  return data_batches[train_idxs,...], label_batches[train_idxs], data_batches[test_idxs,...], label_batches[test_idxs]

def log_string(LOG_FOUT, out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)
