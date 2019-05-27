## Semantic segmentation of indoor scenes

### Dataset

1. Donwload prepared HDF5 data for training:
```
sh +x download_data.sh
```
2. Download 3D indoor parsing dataset (<a href="http://buildingparser.stanford.edu/dataset.html">S3DIS Dataset</a>) for testing and visualization. "Stanford3dDataset_v1.2_Aligned_Version.zip" of the dataset is used. Unzip the downloaded file into "dgcnn/data/", and then run
```
python collect_indoor3d_data.py
```
to generate "deep_gcns/data/stanford_indoor3d"

### Train

We use 6-fold training, such that 6 models are trained leaving 1 of 6 areas as the testing area for each model. We keep using 2 GPUs for distributed training. To train 6 models sequentially, run
```
sh +x train_job.sh
```
If you want to train model with other gcn layers (for example mrgcn), run
```
MODEL="ResMRGCN-28"
python train.py --log_dir $MODEL/log5 --test_area 5 --gcn 'mrgcn'
```

### Evaluation

1. To generate predicted results for all 6 areas, run 
```
sh +x test_job.sh
```

2. To obtain overall quantitative evaluation results, run
```
python eval_iou_accuracy.py
```
If you want to evaluate on one area (for example area5), 
1. To generate predicted results for this area (area5), run
```
python batch_inference.py --model_path $MODEL/log5/epoch_$EPNUM.ckpt --dump_dir $MODEL/log5/dump --output_filelist $MODEL/log5/output_filelist.txt --room_data_filelist meta/area5_data_label.txt
```
2. To obtain the quantitative evaluation results on this area, run
```
python eval_iou_accuracy.py --folder $MODEL --area 5
```
