import torchsummary
import torch
from torch.nn import functional as F
import numpy as np

def dice_coef_torch(y_true, y_pred, eps=1e-6):
    """
    Dice coefficient for binary labels
    """
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    
    return (2. * intersection) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + eps)


def dice_coef_torch_multiclass(y_true, y_pred, n_classes, one_hot_encoded=False, batch=False):
    """
    Computes dice_coefficient of 1 volume, for each class
    Args:
        y_true,y_pred: true and predicted data, with labels as values 0->(n_classes-1) or one-hot encoded (with one-hot dim first)
    Output:
        dice_coefs: array with dice coef of each class
    """
    dice_coefs = []
      
    if one_hot_encoded==False:
        # Convert to one-hot
        if batch:
            y_true = F.one_hot(torch.Tensor(y_true).type(torch.int64), num_classes=n_classes).permute(0,4,1,2,3)
            y_pred = F.one_hot(torch.Tensor(y_pred).type(torch.int64), num_classes=n_classes).permute(0,4,1,2,3)
            
        else :
            y_true = F.one_hot(torch.Tensor(y_true).type(torch.int64), num_classes=n_classes).permute(3,0,1,2)
            y_pred = F.one_hot(torch.Tensor(y_pred).type(torch.int64), num_classes=n_classes).permute(3,0,1,2)
            
    for i in range(n_classes):
        # Compute dice_coef of each class
        
        if batch:
            # In case it comes in a batch
            dice_coefs.append(dice_coef_torch(y_true[:,i], y_pred[:,i]))
        else:
            # In case its just 1 patch
            dice_coefs.append(dice_coef_torch(y_true[i], y_pred[i]))
   
    return torch.Tensor(dice_coefs)

def dice_logits(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice coef.
    From: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        #true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        #true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true = true.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true, dims)
    cardinality = torch.sum(probas + true, dims)
    dice = (2. * intersection / (cardinality + eps))
    return dice