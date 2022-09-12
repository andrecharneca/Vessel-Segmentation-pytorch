
"""""
Dataset configurations:
    :param DATASET_PATH -> the directory path to the processed SAIAD patients (each with respective subfolder)
    :param IN_CHANNELS -> number of input channels
    :param NUM_CLASSES -> specifies the number of output channels for classes (including background as class)
    :param PATCH_SIZE -> volume of patch (x,y,z)
    :param NUM_WORKERS -> num_workers param for torch DataLoader (best one for Mesocentre seems to be 2)
    :param USE_SOFTMAX_END -> use softmax after final conv. Set to false if loss=torch.CrossEntropyLoss

"""""
DATASET_PATH = '../SAIAD-project/Data/SAIAD_data_processed/'
IN_CHANNELS = 1
NUM_CLASSES = 5
PATCH_SIZE = (96,96,96)
NUM_WORKERS = 2
USE_SOFTMAX_END = False

"""""
Training configurations:
    :param TRAINING_EPOCH -> number of training epochs
    :param LR -> learning rate
    :param VAL_BATCH_SIZE -> specifies the batch size of the training DataLoader
    :param TEST_BATCH_SIZE -> specifies the batch size of the test DataLoader
    :param CE_WEIGHTS -> the class weights for the Categorical Cross Entropy loss
"""""

EPOCHS = 100
LR = 0.0001
TRAIN_BATCH_SIZE = 8
TRAIN_BATCHES_PER_EPOCH = 40
VAL_BATCHES_PER_EPOCH = 10
VAL_BATCH_SIZE = 8
TEST_BATCH_SIZE = 1
CE_WEIGHTS = [0.2,1,10,6,1]
