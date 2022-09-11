from json import load
import math
import numpy as np
import torch
import pkbar
import sys
import pkbar
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
    load_data_to_memory=True,
    n_batches=TRAIN_BATCHES_PER_EPOCH
    )
val_dataset = SAIADDataset(
    excl_patients=excl_patients_val,
    load_data_to_memory=True,
    n_batches=VAL_BATCHES_PER_EPOCH
)

## Training ##
model = UNet3D_VGG16(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(device)

criterion = CrossEntropyLoss(weight=torch.Tensor(np.array(CE_WEIGHTS)/np.array(CE_WEIGHTS).sum())).cuda()
optimizer = Adam(params=model.parameters())

min_valid_loss = math.inf

for epoch in range(EPOCHS):
    # progress bar
    kbar = pkbar.Kbar(target=120, epoch=epoch, num_epochs=EPOCHS, width=8, always_stateful=False)

    train_loss = 0.0
    model.train()
    i=0
    for X_batch, y_batch in train_dataset:    
        optimizer.zero_grad(set_to_none=True)
        target = model(X_batch)
        loss = criterion(target, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        kbar.update(i, values=[("loss", train_loss)])
        i+=1
    
    valid_loss = 0.0
    model.eval()
    for X_batch, y_batch in val_dataset:
        target = model(X_batch)
        loss = criterion(target,y_batch)
        valid_loss += loss.item()
    kbar.update(i, values=[("Validation loss", valid_loss)])

    #writer.add_scalar("Loss/Train", train_loss / 120, epoch)
    #writer.add_scalar("Loss/Validation", valid_loss / 8, epoch)
    
    #print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / 8} \t\t Validation Loss: {valid_loss / 8}')
    
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), f'checkpoints/test_epoch{epoch}_valLoss{min_valid_loss}.pth')
