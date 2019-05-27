import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--area', type=int, default=-1, help='Which area to use (default: -1 [all areas])')
parser.add_argument('--folder', type=str, default='', help='folder (default: ./)')
parser.add_argument('--path', type=str, default='', help='path (default: ./')
parser.add_argument('--num_classes', type=int, default=13, help='Number of classes (default: 13)')


FLAGS = parser.parse_args()

AREA = FLAGS.area
FOLDER = FLAGS.folder
PATH = FLAGS.path
NUM_CLASSES = FLAGS.num_classes

print(FLAGS)

if AREA>0:
  start_idx = AREA
  end_idx = AREA+1
else:
  start_idx = 1
  end_idx = 7

for a in range(start_idx,end_idx):
  print("Area: ", a)
  pred_data_label_filenames = []
  file_name = os.path.join(PATH, FOLDER, 'log{}/output_filelist.txt'.format(a))

  pred_data_label_filenames += [os.path.join(PATH, line.rstrip()) for line in open(file_name)]

  gt_label_filenames = [f.rstrip('_pred\.txt') + '_gt.txt' for f in pred_data_label_filenames]

  num_room = len(gt_label_filenames)

  gt_classes = [0 for _ in range(NUM_CLASSES)]
  positive_classes = [0 for _ in range(NUM_CLASSES)]
  true_positive_classes = [0 for _ in range(NUM_CLASSES)]

  for i in range(num_room):
    print(i)
    data_label = np.loadtxt(pred_data_label_filenames[i])
    pred_label = data_label[:,-1]
    gt_label = np.loadtxt(gt_label_filenames[i])
    print(gt_label.shape)
    for j in range(gt_label.shape[0]):
      gt_l = int(gt_label[j])
      pred_l = int(pred_label[j])
      gt_classes[gt_l] += 1
      positive_classes[pred_l] += 1
      true_positive_classes[gt_l] += int(gt_l==pred_l)

  print(gt_classes)
  print(positive_classes)
  print(true_positive_classes)

  print('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))

  print('IoU:')
  iou_list = []
  for i in range(NUM_CLASSES):
    if float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) > 0:
      iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
    else:
      iou = -1
    print(iou)
    iou_list.append(iou)

  print('avg IoU:')
  print(sum(iou_list)/float(NUM_CLASSES))
