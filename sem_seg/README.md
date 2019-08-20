## Semantic segmentation of indoor scenes

### Dataset

1. Download prepared HDF5 data for training:
```
sh +x download_data.sh
```
2. Download 3D indoor parsing dataset (<a href="http://buildingparser.stanford.edu/dataset.html">S3DIS Dataset</a>) for testing and visualization. "Stanford3dDataset_v1.2_Aligned_Version.zip" of the dataset is used. Unzip the downloaded file into "deep_gcns/data" and merge with the folder `Stanford3dDataset_v1.2_Aligned_Version` which already contains the patches `S3DIS_PATCH.diff` and `DS_STORE_PATCH.diff` then run,

```
cd ../data/Stanford3dDataset_v1.2_Aligned_Version
git apply S3DIS_PATCH.diff
find . -name ".DS_Store" -delete
```

Thanks to (<a href="https://github.com/loicland/superpoint_graph">loicland</a>) for the fix.

Next run,

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

Make sure you have successfully setup the dataset for testing (Dataset, Step 2). You should have .npy files with the dataset in the folder "deep_gcns/data/stanford_indoor3d" and auxilary files pointing to the data in the folder "meta". 

If you want to evaluate across all 6 areas (6-fold cross-validation):

1. To generate predicted results for all 6 areas, run
```
sh +x test_job.sh
```

2. Obtain overall quantitative evaluation results
```
python eval_iou_accuracy.py
```

If you want to evaluate on one area (for example area5):

1. To generate predicted results for this area (area5), run
```
python batch_inference.py --model_path $MODEL/log5/epoch_$EPNUM.ckpt --dump_dir $MODEL/log5/dump --output_filelist $MODEL/log5/output_filelist.txt --room_data_filelist meta/area5_data_label.txt
```
2. To obtain the quantitative evaluation results on this area, run
```
python eval_iou_accuracy.py --folder $MODEL --area 5
```

#### Pretrained Models

Several pretrained models (ResGCN-56, ResGCN-28W, ResGCN-28, PlainGCN-28) are available in the folder "pretrained".

If you want to evaluate one of these models:

1. Select the corresponding model in the file "test_pretrained.sh" 

2. To generate the predicted results and evaluate on area 5, run
```
sh +x test_pretrained.sh
```

#### Visualization

In order to compare several pretrained models visually you can follow these steps:

1. To generate the predicted results including files for visualization, run
```
sh +x visualize_pretrained.sh
```

2. Open the Jupyter notebook "visualization.ipynb" and follow the steps in the notebook.

