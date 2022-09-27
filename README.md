# Renal Vessel Segmentation pytorch
Code for the pytorch implementation of the 3D UNet-VGG16 backbone for the SAIAD project.

## Folder Structure
```bash
  .
  ├── README.md : read me file
  ├── dataset_test.ipynb : (ignore) ipynb for debugging some of the dataset functions
  ├── job.sge : job to submit to mesocenter
  ├── model_test.ipynb : (ignore) ipynb for debugging the model
  ├── runs : folder for Tensorboard runs
  ├── test.ipynb : ipynb with testing and inference pipelines
  ├── test_notebooks : (ignore) notebooks for debugging
  ├── train.py : train script
  ├── unet3d : main functions implementing UNet3D with VGG16 backbone
  ├── unet3dvgg16_tensorflow_summary.txt : (ignore) a model summary of the Tensorflow version of the model
  └── utils : other useful functions
```
## General Details
This repo implements the UNet3D model with a VGG16 backbone on Pytorch, as the one obtained by the Tensorflow based package `segmentation-models-3d`.
