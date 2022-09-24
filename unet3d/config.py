

"""""
Folders:
    :param DATASET_PATH -> the directory path to the processed SAIAD patients (each with respective subfolder)
    :param NON_PROCESSED_DATASET_PATH ->
    :param PREDICTIONS_UNIF_PATH ->
    :param PREDICTIONS_ORIGIN_PATH ->
"""""
DATASET_PATH = '../SAIAD-project/Data/SAIAD_data_processed/'
NON_PROCESSED_DATASET_PATH = '../SAIAD-project/Data/SAIAD_data_cleared/'
PREDICTIONS_UNIF_PATH = 'Data/Predicted_Segms_UnifSpacing/'
PREDICTIONS_ORIGIN_PATH = 'Data/Predicted_Segms_OriginSpacing/'

"""""
Dataset configurations:
    :param DATASET_PATH -> the directory path to the processed SAIAD patients (each with respective subfolder)
    :param IN_CHANNELS -> number of input channels
    :param NUM_CLASSES -> specifies the number of output channels for classes (including background as class)
    :param PATCH_SIZE -> volume of patch (x,y,z)
    :param NUM_WORKERS -> num_workers param for torch DataLoader (best one for Mesocentre seems to be 2)
"""""
IN_CHANNELS = 1
NUM_CLASSES = 5
PATCH_SIZE = (96,96,96)
NUM_WORKERS = 4

"""""
Training configurations:
    :param EPOCHS -> number of training epochs
    :param LR -> learning rate
    :param [X]_BATCH_SIZE -> specifies the batch size of [X] in the DataLoader
    :param [X]_BATCHES_PER_EPOCH -> number of batches per epoch
    :param AUG_PROB -> probability of each of the data augmentation transforms being applied to a patch
    :param CE_WEIGHTS -> the class weights for the Categorical Cross Entropy loss
"""""

EPOCHS = 250
LR = 5e-5#0.0001
TRAIN_BATCH_SIZE = 8
TRAIN_BATCHES_PER_EPOCH = 40
VAL_BATCHES_PER_EPOCH = 20#10
AUG_PROB = 0.3
VAL_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
CE_WEIGHTS = [0.2,1,10,6,1]
