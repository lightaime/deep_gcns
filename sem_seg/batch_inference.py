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
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'gcn_lib'))

import tensorflow as tf
import numpy as np
import argparse
import indoor3d_util
import tf_util
import sem_seg_util
import tf_vertex
import tf_edge
from tf_nn import MLP
from gcn_utils import VertexLayer
from gcn_utils import EdgeLayer
from functools import partial, update_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_points', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--dump_dir', required=True, help='dump folder path')
parser.add_argument('--output_filelist', required=True, help='TXT filename, filelist, each line is an output for a room')
parser.add_argument('--room_data_filelist', required=True, help='TXT filename, filelist, each line is a test room data label file.')
parser.add_argument('--no_clutter', action='store_true', help='If true, do not count the clutter class')
parser.add_argument('--visu', action='store_true', help='Whether to output OBJ file for prediction visualization.')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINTS = FLAGS.num_points
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
ROOM_PATH_LIST = [os.path.join(ROOT_DIR,line.rstrip()) for line in open(FLAGS.room_data_filelist)]

train_path = MODEL_PATH.split('/epoch')[0]

with open(train_path + '/log_train.txt') as f:
  first_line = f.readline()
  PARAMS = eval('argparse.'+first_line)
print(PARAMS)

# Network Settings
MODEL_FILE = PARAMS.model
NUM_LAYERS = PARAMS.num_layers
NUM_CLASSES = PARAMS.num_classes

# GCN parameters
NUM_NEIGHBORS = PARAMS.num_neighbors
if (len(NUM_NEIGHBORS) < NUM_LAYERS):
    while (len(NUM_NEIGHBORS) < NUM_LAYERS):
        NUM_NEIGHBORS.append(NUM_NEIGHBORS[-1]) 

NUM_FILTERS = PARAMS.num_filters
if (len(NUM_FILTERS) < NUM_LAYERS):
    while (len(NUM_FILTERS) < NUM_LAYERS):
        NUM_FILTERS.append(NUM_FILTERS[-1]) 

DILATIONS = PARAMS.dilations
if DILATIONS[0] < 0:
    DILATIONS = [1] + list(range(1, NUM_LAYERS))
elif (len(DILATIONS) < NUM_LAYERS):
    while (len(DILATIONS) < NUM_LAYERS):
        DILATIONS.extend(DILATIONS) 
    while (len(DILATIONS) > NUM_LAYERS):
        DILATIONS.pop() 

STOCHASTIC_DILATION = PARAMS.stochastic_dilation
STO_DILATED_EPSILON = PARAMS.sto_dilated_epsilon
SKIP_CONNECT = PARAMS.skip_connect
EDGE_LAY = PARAMS.edge_lay


GCN = PARAMS.gcn
if GCN == "mrgcn":
  print("Using max relative gcn")
elif GCN == 'edgeconv':
  print("Using edgeconv gcn")
elif GCN == 'graphsage':
  NORMALIZE_SAGE = PARAMS.normalize_sage
  print("Using graphsage with normalize={}".format(NORMALIZE_SAGE))
elif GCN == 'gin':
  ZERO_EPSILON_GIN = PARAMS.zero_epsilon_gin
  print("Using gin with zere epsilon={}".format(ZERO_EPSILON_GIN))
else:
  raise Exception("Unknow gcn")

#Import the model
model_builder = __import__(MODEL_FILE)


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def evaluate():
  is_training = False

  with tf.device('/gpu:'+str(GPU_INDEX)):

    # Configure the neural network using every layers
    nn = MLP(kernel_size=[1,1],
              stride=[1,1],
              padding='VALID',
              weight_decay=0.0,
              bn=True,
              bn_decay=None,
              is_dist=True)

    # Configure the gcn vertex layer object
    if GCN == 'mrgcn':
      v_layer = tf_vertex.max_relat_conv_layer
    elif GCN == 'edgeconv':
      v_layer = tf_vertex.edge_conv_layer
    elif GCN == 'graphsage':
      v_layer = wrapped_partial(tf_vertex.graphsage_conv_layer,
                        normalize=NORMALIZE_SAGE)
    elif GCN == 'gin':
      v_layer = wrapped_partial(tf_vertex.gin_conv_layer,
                        zero_epsilon=ZERO_EPSILON_GIN)
    else:
      raise Exception("Unknown gcn type")
    v_layer_builder = VertexLayer(v_layer,
                                  nn)

    # Configure the gcn edge layer object
    if EDGE_LAY == 'dilated':
      e_layer = wrapped_partial(tf_edge.dilated_knn_graph,
                        stochastic=STOCHASTIC_DILATION,
                        epsilon=STO_DILATED_EPSILON)
    elif EDGE_LAY == 'knn':
      e_layer = tf_edge.knn_graph
    else:
      raise Exception("Unknown edge layer type")
    distance_metric = tf_util.pairwise_distance

    e_layer_builder = EdgeLayer(e_layer,
                                distance_metric)

    # Get the whole model builer
    model_obj = model_builder.Model(BATCH_SIZE,
                                    NUM_POINTS,
                                    NUM_LAYERS,
                                    NUM_NEIGHBORS,
                                    NUM_FILTERS,
                                    NUM_CLASSES,
                                    vertex_layer_builder=v_layer_builder,
                                    edge_layer_builder=e_layer_builder,
                                    mlp_builder=nn,
                                    skip_connect=SKIP_CONNECT,
                                    dilations=DILATIONS)

    inputs_ph = model_obj.inputs
    labels_ph = model_obj.labels
    is_training_ph = model_obj.is_training
    pred = model_obj.pred
    loss = model_obj.get_loss(pred, labels_ph)

    pred_softmax = tf.nn.softmax(pred)

    saver = tf.train.Saver()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  sess = tf.Session(config=config)

  saver.restore(sess, MODEL_PATH)
  sem_seg_util.log_string(LOG_FOUT, "Model restored.")

  ops = {'pointclouds_pl': inputs_ph,
       'labels_pl': labels_ph,
       'is_training_pl': is_training_ph,
       'pred': pred,
       'pred_softmax': pred_softmax,
       'loss': loss}

  total_correct = 0
  total_seen = 0
  fout_out_filelist = open(FLAGS.output_filelist, 'w')
  for room_path in ROOM_PATH_LIST:
    out_data_label_filename = os.path.basename(room_path)[:-4] + '_pred.txt'
    out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
    out_gt_label_filename = os.path.basename(room_path)[:-4] + '_gt.txt'
    out_gt_label_filename = os.path.join(DUMP_DIR, out_gt_label_filename)

    print(room_path, out_data_label_filename)
    # Evaluate room one by one.
    a, b = eval_one_epoch(sess, ops, room_path, out_data_label_filename, out_gt_label_filename)
    total_correct += a
    total_seen += b
    fout_out_filelist.write(out_data_label_filename+'\n')
  fout_out_filelist.close()
  sem_seg_util.log_string(LOG_FOUT, 'all room eval accuracy: %f'% (total_correct / float(total_seen)))

def eval_one_epoch(sess, ops, room_path, out_data_label_filename, out_gt_label_filename):
  error_cnt = 0
  is_training = False
  total_correct = 0
  total_seen = 0
  loss_sum = 0
  total_seen_class = [0 for _ in range(NUM_CLASSES)]
  total_correct_class = [0 for _ in range(NUM_CLASSES)]

  if FLAGS.visu:
    fout = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4]+'_pred.obj'), 'w')
    fout_gt = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4]+'_gt.obj'), 'w')
    fout_real_color = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4]+'_real_color.obj'), 'w')
  fout_data_label = open(out_data_label_filename, 'w')
  fout_gt_label = open(out_gt_label_filename, 'w')

  current_data, current_label = indoor3d_util.room2blocks_wrapper_normalized(room_path, NUM_POINTS)
  current_data = current_data[:,0:NUM_POINTS,:]
  current_label = np.squeeze(current_label)
  # Get room dimension..
  data_label = np.load(room_path)
  data = data_label[:,0:6]
  max_room_x = max(data[:,0])
  max_room_y = max(data[:,1])
  max_room_z = max(data[:,2])

  file_size = current_data.shape[0]
  num_batches = file_size // BATCH_SIZE
  print(file_size)

  for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = (batch_idx+1) * BATCH_SIZE
    cur_batch_size = end_idx - start_idx

    feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
           ops['labels_pl']: current_label[start_idx:end_idx],
           ops['is_training_pl']: is_training}
    loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                    feed_dict=feed_dict)

    if FLAGS.no_clutter:
      pred_label = np.argmax(pred_val[:,:,0:12], 2) # BxN
    else:
      pred_label = np.argmax(pred_val, 2) # BxN

    # Save prediction labels to OBJ file
    for b in range(BATCH_SIZE):
      pts = current_data[start_idx+b, :, :]
      l = current_label[start_idx+b,:]
      pts[:,6] *= max_room_x
      pts[:,7] *= max_room_y
      pts[:,8] *= max_room_z
      pts[:,3:6] *= 255.0
      pred = pred_label[b, :]
      for i in range(NUM_POINTS):
        color = indoor3d_util.g_label2color[pred[i]]
        color_gt = indoor3d_util.g_label2color[current_label[start_idx+b, i]]
        if FLAGS.visu:
          fout.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color[0], color[1], color[2]))
          fout_gt.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color_gt[0], color_gt[1], color_gt[2]))
        fout_data_label.write('%f %f %f %d %d %d %f %d\n' % (pts[i,6], pts[i,7], pts[i,8], pts[i,3], pts[i,4], pts[i,5], pred_val[b,i,pred[i]], pred[i]))
        fout_gt_label.write('%d\n' % (l[i]))

    correct = np.sum(pred_label == current_label[start_idx:end_idx,:])
    total_correct += correct
    total_seen += (cur_batch_size*NUM_POINTS)
    loss_sum += (loss_val*BATCH_SIZE)
    for i in range(start_idx, end_idx):
      for j in range(NUM_POINTS):
        l = current_label[i, j]
        total_seen_class[l] += 1
        total_correct_class[l] += (pred_label[i-start_idx, j] == l)

  sem_seg_util.log_string(LOG_FOUT, 'eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINTS)))
  sem_seg_util.log_string(LOG_FOUT, 'eval accuracy: %f'% (total_correct / float(total_seen)))
  fout_data_label.close()
  fout_gt_label.close()
  if FLAGS.visu:
    fout.close()
    fout_gt.close()
  return total_correct, total_seen

if __name__=='__main__':
  with tf.Graph().as_default():
    evaluate()
  LOG_FOUT.close()
