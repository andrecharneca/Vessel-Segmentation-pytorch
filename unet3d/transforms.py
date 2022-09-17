from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    NormalizeIntensityd
)
from .config import AUG_PROB

KEYS = ['patch_scan', 'patch_segm']
device = 'cuda'

train_transform = Compose(
    [   
        RandFlipd(keys=KEYS, prob=AUG_PROB, spatial_axis=0),
        RandGaussianNoised(keys=KEYS[0], prob=AUG_PROB, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=KEYS[0], prob=AUG_PROB, gamma=(0.5,2)),
        NormalizeIntensityd(keys=KEYS[0], nonzero=True, allow_missing_keys=False) #only normalize non-zero values
        
    ]
)
val_transform = Compose(
    [   
        NormalizeIntensityd(keys=KEYS[0], nonzero=True, allow_missing_keys=False)
        
    ]
)
val_transform
#NOTE: add patch intensity normalization?