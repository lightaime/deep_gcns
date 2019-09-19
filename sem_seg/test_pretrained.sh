#!/bin/bash

MODEL="pretrained/ResGCN-28" #ResGCN-56, ResGCN-28W, ResGCN-28, DenseGCN-28, PlainGCN-28
EPNUM="best"

echo "\nProcessing Model $MODEL \n"
python batch_inference.py --model_path $MODEL/log5/epoch_$EPNUM.ckpt --dump_dir $MODEL/log5/dump --output_filelist $MODEL/log5/output_filelist.txt --room_data_filelist meta/area5_data_label.txt
echo "\nEvaluating Model $MODEL \n"     
python eval_iou_accuracy.py --folder $MODEL --area 5


