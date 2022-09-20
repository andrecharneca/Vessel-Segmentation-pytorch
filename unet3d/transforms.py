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
KEYS_TEST = ['patch_scan', 'patch_scan_flipped', 'patch_scan_noise', 'patch_scan_contrast']

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


# Transforms for multiple inference DON'T CHANGE ORDER OF THIS, it's important to re-flip the segmentation
test_transform = Compose(
    [   
        NormalizeIntensityd(keys=KEYS_TEST, nonzero=True, allow_missing_keys=True),
        RandFlipd(keys=KEYS_TEST[1], prob=1, spatial_axis=0, allow_missing_keys=True),
        RandGaussianNoised(keys=KEYS_TEST[2], prob=1, mean=0.0, std=0.1, allow_missing_keys=True),
        RandAdjustContrastd(keys=KEYS_TEST[3], prob=1, gamma=(0.5,2), allow_missing_keys=True)
    ]
)