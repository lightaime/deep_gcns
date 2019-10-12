# DeepGCNs: Can GCNs Go as Deep as CNNs?
In this work, we present new ways to successfully train very deep GCNs. We borrow concepts from CNNs, mainly residual/dense connections and dilated convolutions, and adapt them to GCN architectures. Through extensive experiments, we show the positive effect of these deep GCN frameworks.

[[Project]](https://www.deepgcns.org/) [[Paper]](https://arxiv.org/abs/1904.03751) [[Slides]](https://docs.google.com/presentation/d/1L82wWymMnHyYJk3xUKvteEWD5fX0jVRbCbI65Cxxku0/edit?usp=sharing) [[Tensorflow Code]](https://github.com/lightaime/deep_gcns) [[Pytorch Code]](https://github.com/lightaime/deep_gcns_torch)

<div style="text-align:center"><img src='./misc/intro.png' width=800>

## Overview
We do extensive experiments to show how different components (#Layers, #Filters, #Nearest Neighbors, Dilation, etc.) effect `DeepGCNs`. We also provide ablation studies on different type of Deep GCNs (MRGCN, EdgeConv, GraphSage and GIN).

<div style="text-align:center"><img src='./misc/pipeline.png' width=800>

Further information and details please contact [Guohao Li](https://ghli.org) and [Matthias Müller](https://matthias.pw/).

## Requirements
* [TensorFlow 1.12.0](https://www.tensorflow.org/)
* [h5py](https://www.h5py.org/)
* [vtk](https://vtk.org/) (only needed for visualization)
* [jupyter notebook](https://jupyter.org/) (only needed for visualization)

## Conda Environment
In order to setup a conda environment with all neccessary dependencies run,
```
conda env create -f environment.yml
```

## Getting Started
You will find detailed instructions how to use our code for semantic segmentation of 3D point clouds, in the folder [sem_seg](sem_seg/). Currently, we provide the following:
* Conda environment
* Setup of <a href="http://buildingparser.stanford.edu/dataset.html">S3DIS Dataset</a>
* Training code
* Evaluation code
* Several pretrained models
* Visualization code

## Citation
Please cite our paper if you find anything helpful,
```
@InProceedings{li2019deepgcns,
    title={DeepGCNs: Can GCNs Go as Deep as CNNs?},
    author={Guohao Li and Matthias Müller and Ali Thabet and Bernard Ghanem},
    booktitle={The IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
}
```


## License
MIT License

## Acknowledgement
This code is heavily borrowed from [PointNet](https://github.com/charlesq34/pointnet) and [EdgeConv](https://github.com/WangYueFt/dgcnn). We would also like to thank [3d-semantic-segmentation](https://github.com/VisualComputingInstitute/3d-semantic-segmentation) for the visualization code.
