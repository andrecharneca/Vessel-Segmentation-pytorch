import torchsummary
import torch
from torch.nn.functional import one_hot

def dice_coef_torch(y_true, y_pred):
    """
    Dice coefficient for binary labels
    """
    y_true_f = torch.flatten(y_true_f)
    y_pred_f = torch.flatten(y_pred_f)
    intersection = torch.sum(y_true_f * y_pred_f)
    
    return (2. * intersection) / (torch.sum(y_true_f) + torch.sum(y_pred_f))


def dice_coef_torch_multiclass(y_true, y_pred, n_classes, one_hot=False):
    """
    Computes dice_coefficient of 1 volume, for each class
    Args:
        y_true,y_pred: true and predicted data, with labels as values 0->(n_classes-1) or one-hot encoded
    Output:
        dice_coefs: array with dice coef of each class
    """
    dice_coefs = []
    
    if one_hot==False:
        # Convert to one-hot
        y_true = torch.unsqueeze(y_true,0)
        y_true = one_hot(y_true, num_classes=n_classes).permute(3,0,1,2)
        y_pred = torch.unsqueeze(y_pred,0)
        y_pred = one_hot(y_pred, num_classes=n_classes).permute(3,0,1,2)
        
    for i in range(n_classes):
        # Compute dice_coef of each class
        dice_coefs.append(dice_coef(y_true[i], y_pred[i]))
                          
    return np.array(dice_coefs)