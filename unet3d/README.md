# Pytorch implementation of 3D UNet

This implementation is based on the orginial 3D UNet paper and adapted to be used for MRI or CT image segmentation task   
> Link to the paper: [https://arxiv.org/pdf/1606.06650v1.pdf](https://arxiv.org/pdf/1606.06650v1.pdf)

## Model Architecture
3D UNet with a VGG16 backbone. Inspired by the segmentation-models-3d package.

## Configure the network

All the configurations and hyperparameters are set in the config.py file.
Please note that you need to change the path to the dataset directory in the config.py file before running the model.

**Parameters:**

- DATASET_PATH -> the directory path to dataset .tar files

- TASK_ID -> specifies the the segmentation task ID (see the dict below for hints)

- IN_CHANNELS -> number of input channels

- NUM_CLASSES -> specifies the number of output channels for dispirate classes

- TRAIN_VAL_TEST_SPLIT -> delineates the ratios in which the dataset shoud be splitted. The length of the array should be 3.

- TRAINING_EPOCH -> number of training epochs

- VAL_BATCH_SIZE -> specifies the batch size of the training DataLoader

- TEST_BATCH_SIZE -> specifies the batch size of the test DataLoader

- TRAIN_CUDA -> if True, moves the model and inference onto GPU

- CE_WEIGHTS -> the class weights for the Categorical Cross Entropy loss

