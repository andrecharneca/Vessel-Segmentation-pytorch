from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd
)
from .config import AUG_PROB

KEYS = ['patch_scan', 'patch_segm']

#Cuda version of "train_transform"
train_transform_cuda = Compose(
    [   
        RandFlipd(keys=KEYS, prob=AUG_PROB, spatial_axis=0),
        RandGaussianNoised(keys=KEYS, prob=AUG_PROB, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=KEYS, prob=AUG_PROB, gamma=(0.5,2)),
        #ToTensord(keys=KEYS, device='mps')###change to cuda
    ]
)