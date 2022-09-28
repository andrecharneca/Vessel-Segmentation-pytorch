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
from torch.nn.functional import one_hot, softmax
from torch.optim import Adam, SGD
from unet3d.unet3d_vgg16 import UNet3D_VGG16, init_weights
from utils.Other import get_headers
from unet3d.dataset import SAIADDataset, WrappedDataLoader, to_device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pynvml.smi import nvidia_smi
from unet3d.transforms import train_transform, val_transform
from unet3d.dice import *

date='28sep'
model_name = f'normalized_glorotinit_saiad1and18tervasc_{date}'
writer = SummaryWriter(log_dir=f'runs/{model_name}')
torch.manual_seed(0)
_,_,patient_names = get_headers(DATASET_PATH)
nvsmi = nvidia_smi.getInstance()


## Parameters ##
torch.backends.cudnn.benchmark = True # Speeds up stuff
torch.backends.cudnn.enabled = True
device = torch.device('cuda')
pin_memory = True###

excl_patients_training = ['SAIAD 1', 'SAIAD 18 TER vasculaire'] #patients for validation/testing
excl_patients_val = list(set(patient_names) - set(excl_patients_training))

print("Training with val patients:", excl_patients_training)



## Load dataset ##
train_dataset = SAIADDataset(
    excl_patients=excl_patients_training,
    load_data_to_memory=True,
    n_batches=TRAIN_BATCHES_PER_EPOCH,
    transform = train_transform
    )
val_dataset = SAIADDataset(
    excl_patients=excl_patients_val,
    load_data_to_memory=True,
    n_batches=VAL_BATCHES_PER_EPOCH,
    transform = val_transform
)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=False, 
    pin_memory=pin_memory, 
    num_workers=NUM_WORKERS
    )
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=VAL_BATCH_SIZE,
    shuffle=False, 
    pin_memory=pin_memory, 
    num_workers=NUM_WORKERS
    )

val_dataloader = WrappedDataLoader(val_dataloader, to_device, device)
train_dataloader = WrappedDataLoader(train_dataloader, to_device, device)




## Model ##
model = UNet3D_VGG16(
    in_channels=IN_CHANNELS, 
    num_classes=NUM_CLASSES,
    use_softmax_end=False #set this to false for training with CELoss
    ).to(device)
model.apply(init_weights)

loss_fn = CrossEntropyLoss(
    weight=torch.Tensor(np.array(CE_WEIGHTS)/np.array(CE_WEIGHTS).sum()),
    reduction = 'mean'
    ).cuda()

optimizer = Adam(params=model.parameters(), lr=LR, eps=1e-7)
scaler = torch.cuda.amp.GradScaler()


## Training ##
min_valid_loss = math.inf

for epoch in range(EPOCHS):
    # Check memory usage
    mem_query = nvsmi.DeviceQuery('memory.free, memory.total')['gpu'][0]['fb_memory_usage']

    # progress bar
    kbar = pkbar.Kbar(target=TRAIN_BATCHES_PER_EPOCH+VAL_BATCHES_PER_EPOCH, epoch=epoch, num_epochs=EPOCHS, width=8, always_stateful=True)
    
    ## Training ##
    train_loss = 0.0
    dice_vals = np.zeros(5)
    model.train()
    i=1
    batch_num = 1
    kbar.update(i, values=[("Mem. Usage (MB)", mem_query['total']-mem_query['free'])])

    for X_batch, y_batch in train_dataloader:  
        if torch.isnan(X_batch).any():
            print("X_batch has nan value")
        if torch.isnan(y_batch).any():
            print("y_batch has nan value")


        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

        # Using gradient scaling bc of float16 precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
            
        # Compute dice coefs
        dice_vals += dice_logits(y_batch.float(), pred.float()).cpu().detach().numpy()
        train_loss += loss.cpu().detach()
        kbar.update(i, values=[("Loss/train", train_loss/batch_num), 
                               ("Vein Dice/train", dice_vals[2]/batch_num),
                               ("Artery Dice/train", dice_vals[3]/batch_num)])
        i+=1
        batch_num+=1
        
    # Tensorboard #
    writer.add_scalar("Loss/train", train_loss/TRAIN_BATCHES_PER_EPOCH, epoch)
    writer.add_scalar("Tumor Dice/train", dice_vals[1]/TRAIN_BATCHES_PER_EPOCH, epoch)
    writer.add_scalar("Vein Dice/train", dice_vals[2]/TRAIN_BATCHES_PER_EPOCH, epoch)
    writer.add_scalar("Artery Dice/train", dice_vals[3]/TRAIN_BATCHES_PER_EPOCH, epoch)
    writer.add_scalar("Kidney Dice/train", dice_vals[4]/TRAIN_BATCHES_PER_EPOCH, epoch)

    ## Validation ##
    valid_loss = 0.0
    dice_vals = np.zeros(5)
    model.eval()
    batch_num = 1
    with torch.no_grad():
        for X_batch, y_batch in val_dataloader:
            with torch.cuda.amp.autocast():
                pred = model(X_batch)
                loss = loss_fn(pred,y_batch)
            
            dice_vals += dice_logits(y_batch.float(), pred.float()).cpu().detach().numpy()
            valid_loss += loss.cpu().detach()
            kbar.update(i, values=[("Loss/val", valid_loss/batch_num),                                                                      ("Vein Dice/val", dice_vals[2]/batch_num),
                                   ("Artery Dice/val", dice_vals[3]/batch_num)])
            i+=1
            batch_num+=1
            
    # Tensorboard #
    writer.add_scalar("Loss/val", valid_loss/VAL_BATCHES_PER_EPOCH, epoch)
    writer.add_scalar("Tumor Dice/val", dice_vals[1]/VAL_BATCHES_PER_EPOCH, epoch)
    writer.add_scalar("Vein Dice/val", dice_vals[2]/VAL_BATCHES_PER_EPOCH, epoch)
    writer.add_scalar("Artery Dice/val", dice_vals[3]/VAL_BATCHES_PER_EPOCH, epoch)
    writer.add_scalar("Kidney Dice/val", dice_vals[4]/VAL_BATCHES_PER_EPOCH, epoch)
    
    # Checkpoints #
    if min_valid_loss > valid_loss/VAL_BATCHES_PER_EPOCH:
        print(f'\t Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss/VAL_BATCHES_PER_EPOCH:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss/VAL_BATCHES_PER_EPOCH
        # Saving State Dict
        torch.save(model.state_dict(), f'checkpoints/{model_name}_epoch{epoch+1}.pth')
    elif (epoch+1)%(EPOCHS//10) == 0:
        print(f'\t Reached checkpoint. \t Saving The Model')
        torch.save(model.state_dict(),f'checkpoints/{model_name}_epoch{epoch+1}.pth')


# Tensorboard #
writer.flush()
writer.close()
