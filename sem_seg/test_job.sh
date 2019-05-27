#!/bin/bash          
MODEL="ResGCN-28"       # model name
EPNUM="best"            # replace with the number of the checkpoint you wish to evaluate

python batch_inference.py --model_path $MODEL/log1/epoch_$EPNUM.ckpt --dump_dir $MODEL/log1/dump --output_filelist $MODEL/log1/output_filelist.txt --room_data_filelist meta/area1_data_label.txt
python batch_inference.py --model_path $MODEL/log2/epoch_$EPNUM.ckpt --dump_dir $MODEL/log2/dump --output_filelist $MODEL/log2/output_filelist.txt --room_data_filelist meta/area2_data_label.txt
python batch_inference.py --model_path $MODEL/log3/epoch_$EPNUM.ckpt --dump_dir $MODEL/log3/dump --output_filelist $MODEL/log3/output_filelist.txt --room_data_filelist meta/area3_data_label.txt
python batch_inference.py --model_path $MODEL/log4/epoch_$EPNUM.ckpt --dump_dir $MODEL/log4/dump --output_filelist $MODEL/log4/output_filelist.txt --room_data_filelist meta/area4_data_label.txt
python batch_inference.py --model_path $MODEL/log5/epoch_$EPNUM.ckpt --dump_dir $MODEL/log5/dump --output_filelist $MODEL/log5/output_filelist.txt --room_data_filelist meta/area5_data_label.txt
python batch_inference.py --model_path $MODEL/log6/epoch_$EPNUM.ckpt --dump_dir $MODEL/log6/dump --output_filelist $MODEL/log6/output_filelist.txt --room_data_filelist meta/area6_data_label.txt

python eval_iou_accuracy.py --folder $MODEL
