#!/bin/bash

EPNUM="best"
MODELS="pretrained/ResGCN-28 pretrained/ResGCN-28W pretrained/ResGCN-56"

# Iterate the string array using for loop
for MODEL in $MODELS; do
  echo "\nProcessing Model $MODEL \n"
  python batch_inference.py --model_path $MODEL/log5/epoch_$EPNUM.ckpt --dump_dir $MODEL/log5/dump --output_filelist $MODEL/log5/output_filelist.txt --room_data_filelist meta/area5_data_label.txt --visu
done


