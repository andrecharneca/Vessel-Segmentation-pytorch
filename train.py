from json import load
import math
import numpy as np
import torch
import pkbar
import sys
from unet3d.config import *
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import Adam
#from unet3d.unet3d_vgg16 import UNet3D_VGG16
from utils.Other import get_headers
from unet3d.dataset import SAIADDataset


torch.manual_seed(0)
_,_,patient_names = get_headers(DATASET_PATH)

## Parameters ##
torch.backends.cudnn.benchmark = True # Speeds up stuff
torch.backends.cudnn.enabled = True
device = torch.device('cuda')

excl_patients_training = ['SAIAD 15', 'SAIAD 11'] #patients for validation/testing
excl_patients_val = list(set(patient_names) - set(excl_patients_training))

print(excl_patients_val)

## Load dataset ##
train_dataset = SAIADDataset(
    excl_patients=excl_patients_training,
    load_data_to_memory=True
    )
val_dataset = SAIADDataset(
    excl_patients=excl_patients_val,
    load_data_to_memory=True
)

## Training ##
model = UNet3D_VGG16(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(device)

