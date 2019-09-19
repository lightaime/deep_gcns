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
import numpy as np
import tensorflow as tf

import tf_util
import sem_seg_util
import provider
import tf_vertex
import tf_edge
from tf_nn import MLP
from gcn_utils import VertexLayer
from gcn_utils import EdgeLayer
from functools import partial, update_wrapper

FLAGS = sem_seg_util.parse_args()
print(FLAGS)

# Files setup
DATASET = FLAGS.dataset
TEST_AREA = str(FLAGS.test_area)
MODEL_FILE = FLAGS.model
model_builder = __import__(MODEL_FILE)
LOG_DIR = FLAGS.log_dir
CHECKPOINT = FLAGS.checkpoint
if not os.path.exists(LOG_DIR):
  os.makedirs(LOG_DIR)
os.system('cp ' + MODEL_FILE + '.py' ' %s/model.py' % (LOG_DIR))
os.system('cp train.py %s' % (LOG_DIR))
if (CHECKPOINT != ''):
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
else:
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# Training Settings
TOWER_NAME = FLAGS.tower_name
NUM_GPU = FLAGS.num_gpu
BATCH_SIZE = FLAGS.batch_size
NUM_POINTS = FLAGS.num_points
NUM_LAYERS = FLAGS.num_layers
NUM_CLASSES = FLAGS.num_classes
MAX_EPOCH = FLAGS.max_epoch
OPTIMIZER = FLAGS.optimizer
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = FLAGS.bn_init_decay
BN_DECAY_DECAY_RATE = FLAGS.bn_decay_decay_rate
BN_DECAY_DECAY_STEP = FLAGS.bn_decay_decay_step
BN_DECAY_CLIP = FLAGS.bn_decay_clip

# GCN parameters
NUM_NEIGHBORS = FLAGS.num_neighbors
if (len(NUM_NEIGHBORS) < NUM_LAYERS):
    while (len(NUM_NEIGHBORS) < NUM_LAYERS):
        NUM_NEIGHBORS.append(NUM_NEIGHBORS[-1]) 

NUM_FILTERS = FLAGS.num_filters
if (len(NUM_FILTERS) < NUM_LAYERS):
    while (len(NUM_FILTERS) < NUM_LAYERS):
        NUM_FILTERS.append(NUM_FILTERS[-1]) 

DILATIONS = FLAGS.dilations
if DILATIONS[0] < 0:
    DILATIONS = [1] + list(range(1, NUM_LAYERS))
elif (len(DILATIONS) < NUM_LAYERS):
    while (len(DILATIONS) < NUM_LAYERS):
        DILATIONS.extend(DILATIONS) 
    while (len(DILATIONS) > NUM_LAYERS):
        DILATIONS.pop() 

STOCHASTIC_DILATION = FLAGS.stochastic_dilation
STO_DILATED_EPSILON = FLAGS.sto_dilated_epsilon
SKIP_CONNECT = FLAGS.skip_connect
EDGE_LAY = FLAGS.edge_lay


GCN = FLAGS.gcn

if GCN == "mrgcn":
  print("Using max relative gcn")
elif GCN == 'edgeconv':
  print("Using edgeconv gcn")
elif GCN == 'graphsage':
  NORMALIZE_SAGE = FLAGS.normalize_sage
  print("Using graphsage with normalize={}".format(NORMALIZE_SAGE))
elif GCN == 'gin':
  ZERO_EPSILON_GIN = FLAGS.zero_epsilon_gin
  print("Using gin with zero epsilon={}".format(ZERO_EPSILON_GIN))
else:
  raise Exception("Unknow gcn")

if DATASET == 'vkitti':
  print("Training on vKITTI") # NUM_CLASSES should be 14
  ALL_FILES = provider.getDataFiles('vkitti_hdf5/all_files.txt')
  ROOM_FILELIST = [line.rstrip() for line in open('vkitti_hdf5/room_filelist.txt')]
elif DATASET == 's3dis':
  print("Training on Stanford 3D Indoor Spaces Dataset") # NUM_CLASSES should be 13
  ALL_FILES = provider.getDataFiles('indoor3d_sem_seg_hdf5_data/all_files.txt')
  ROOM_FILELIST = [line.rstrip() for line in open('indoor3d_sem_seg_hdf5_data/room_filelist.txt')]
else:
  raise Exception("Unknown dataset")

print('Room files length {}'.format(len(ROOM_FILELIST)))

train_data, train_label, test_data, test_label = sem_seg_util.load_data(ALL_FILES, ROOM_FILELIST, TEST_AREA)
print("Train set shape inputs {}, labels {}".format(train_data.shape, train_label.shape))
print("Test set shape inputs {}, labels {}".format(test_data.shape, test_label.shape))

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def train():
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    batch = tf.Variable(0, trainable=False)

    learning_rate = tf_util.get_learning_rate(batch,
                                              BASE_LEARNING_RATE,
                                              BATCH_SIZE,
                                              DECAY_STEP,
                                              DECAY_RATE)
    tf.summary.scalar('learning_rate', learning_rate)

    bn_decay = tf_util.get_bn_decay(batch,
                                    BN_INIT_DECAY,
                                    BATCH_SIZE,
                                    BN_DECAY_DECAY_STEP,
                                    BN_DECAY_DECAY_RATE,
                                    BN_DECAY_CLIP)
    tf.summary.scalar('bn_decay', bn_decay)

    if OPTIMIZER == 'momentum':
      print('Using SGD with Momentum as optimizer')
      trainer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM)
    elif OPTIMIZER == 'adam':
      print('Using Adam as optimizer')
      trainer = tf.train.AdamOptimizer(learning_rate)
    else:
      raise Exception("Unknown optimizer")

    tower_grads = []
    inputs_phs = []
    labels_phs = []
    is_training_phs =[]

    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(NUM_GPU):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:

            # Configure the neural network using every layers
            nn = MLP(kernel_size=[1,1],
                      stride=[1,1],
                      padding='VALID',
                      weight_decay=0.0,
                      bn=True,
                      bn_decay=bn_decay,
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
            inputs_phs.append(inputs_ph)
            labels_phs.append(labels_ph)
            is_training_phs.append(is_training_ph)

            loss = model_obj.get_loss(pred, labels_phs[-1])
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_phs[-1]))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINTS)
            tf.summary.scalar('accuracy', accuracy)

            tf.get_variable_scope().reuse_variables()

            grads = trainer.compute_gradients(loss)

            tower_grads.append(grads)

    grads = tf_util.average_gradients(tower_grads)

    train_op = trainer.apply_gradients(grads, global_step=batch)

    saver = tf.train.Saver(tf.global_variables(), sharded=True, max_to_keep=None)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Add summary writers
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                  sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

    # Init variables on GPUs
    init = tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer())
    sess.run(init)

    if (CHECKPOINT != ''):
        saver.restore(sess, CHECKPOINT)
        sem_seg_util.log_string(LOG_FOUT, "Model restored.")
        start_epoch = int(CHECKPOINT.split('.')[0].split('epoch_')[1])
        print('Resuming from epoch: {}'.format(start_epoch))
    else:
        start_epoch = 0

    ops = {'inputs_phs': inputs_phs,
         'labels_phs': labels_phs,
         'is_training_phs': is_training_phs,
         'pred': pred,
         'loss': loss,
         'train_op': train_op,
         'merged': merged,
         'step': batch}

    for epoch in range(start_epoch+1, MAX_EPOCH):
      sem_seg_util.log_string(LOG_FOUT, '**** EPOCH %03d ****' % (epoch))
      sys.stdout.flush()

      train_one_epoch(sess, ops, train_writer)

      # Save the variables to disk.
      if epoch % 10 == 0:
        save_path = saver.save(sess, os.path.join(LOG_DIR,'epoch_' + str(epoch)+'.ckpt'))
        sem_seg_util.log_string(LOG_FOUT, "Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
  """ ops: dict mapping from string to tf ops """
  is_training = True

  sem_seg_util.log_string(LOG_FOUT, '----')
  current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINTS,:], train_label)

  file_size = current_data.shape[0]
  num_batches = file_size // (NUM_GPU * BATCH_SIZE)

  total_correct = 0
  total_seen = 0
  loss_sum = 0

  for batch_idx in range(num_batches):

    if batch_idx % 100 == 0:
      print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))

    start_idx = []
    end_idx = []

    for gpu_idx in range(NUM_GPU):
      start_idx.append((batch_idx + gpu_idx) * BATCH_SIZE)
      end_idx.append((batch_idx + gpu_idx + 1) * BATCH_SIZE)

    feed_dict = dict()
    for gpu_idx in range(NUM_GPU):
      feed_dict[ops['inputs_phs'][gpu_idx]] = current_data[start_idx[gpu_idx]:end_idx[gpu_idx], :, :]
      feed_dict[ops['labels_phs'][gpu_idx]] = current_label[start_idx[gpu_idx]:end_idx[gpu_idx]]
      feed_dict[ops['is_training_phs'][gpu_idx]] = is_training

    summary, step, _, loss_val, pred_val = sess.run([ops['merged'],
                                                     ops['step'],
                                                     ops['train_op'],
                                                     ops['loss'],
                                                     ops['pred']],
                                                    feed_dict=feed_dict)

    train_writer.add_summary(summary, step)
    pred_val = np.argmax(pred_val, 2)
    correct = np.sum(pred_val == current_label[start_idx[-1]:end_idx[-1]])
    total_correct += correct
    total_seen += (BATCH_SIZE*NUM_POINTS)
    loss_sum += loss_val

  sem_seg_util.log_string(LOG_FOUT, 'mean loss: %f' % (loss_sum / float(num_batches)))
  sem_seg_util.log_string(LOG_FOUT, 'accuracy: %f' % (total_correct / float(total_seen)))

if __name__ == "__main__":
  train()
  LOG_FOUT.close()
