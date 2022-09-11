
"""""
Dataset configurations:
    :param DATASET_PATH -> the directory path to the processed SAIAD patients (each with respective subfolder)
    :param TASK_ID -> specifies the the segmentation task ID (see the dict below for hints)
    :param IN_CHANNELS -> number of input channels
    :param NUM_CLASSES -> specifies the number of output channels for classes (including background as class)
    :param PATCH_SIZE -> volume of patch (x,y,z)

"""""
DATASET_PATH = '../SAIAD-project/Data/SAIAD_data_processed/'
IN_CHANNELS = 1
NUM_CLASSES = 5
PATCH_SIZE = (96,96,96)
"""""
Training configurations:
    :param TRAIN_VAL_TEST_SPLIT -> delineates the ratios in which the dataset shoud be splitted. The length of the array should be 3.
    :param SPLIT_SEED -> the random seed with which the dataset is splitted
    :param TRAINING_EPOCH -> number of training epochs
    :param VAL_BATCH_SIZE -> specifies the batch size of the training DataLoader
    :param TEST_BATCH_SIZE -> specifies the batch size of the test DataLoader
    :param TRAIN_CUDA -> if True, moves the model and inference onto GPU
    :param CE_WEIGHTS -> the class weights for the Categorical Cross Entropy loss
"""""
TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]
SPLIT_SEED = 42
EPOCHS = 100
TRAIN_BATCH_SIZE = 8
TRAIN_BATCHES_PER_EPOCH = 40
VAL_BATCHES_PER_EPOCH = 10
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
TRAIN_CUDA = True
CE_WEIGHTS = [0.2,1,10,6,1]
