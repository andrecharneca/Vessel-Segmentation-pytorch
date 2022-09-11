# File for other helper functions
import numpy as np
import glob
import nrrd
from tensorflow.keras.utils import to_categorical

def get_headers(data_path):
    """
    Get headers from .nrrd files.
    Supply a path '/data_path'.
    """
    scan_paths = [name for name in glob.glob(data_path + '*/scan.nrrd')]
    segm_paths = [name for name in glob.glob(data_path + '*/segm.nrrd')]

    patient_names = [name.split('/')[-2] for name in scan_paths]
    headers_scans = []
    headers_segm = []
    for i in range(len(scan_paths)):
        _, header_scan = nrrd.read(scan_paths[i])
        _, header_segm = nrrd.read(segm_paths[i])
        headers_scans.append(header_scan)
        headers_segm.append(header_segm)
        
    return headers_scans, headers_segm, patient_names
        
def get_class_weights(y_train, n_classes):
    """
    Computes class weights inversely proportional to #of voxels of that class
    
    Args:
        y_train: one-hot encoded array of size #patches x patch_dimension x 5 (one-hot encoding)
        n_classes: number of classes (including background)
    Output:
        class_weights: normalized vector
    """
    class_weights = []
    
    for i in range(n_classes):
        # Inverse of Number of voxels belonging to class
        class_weights.append(1/np.count_nonzero(y_train[:,:,:,:,i]))
        
    # Return normalized vector
    return np.array(class_weights/np.sum(class_weights))

def read_data(data_path, verbose = 0):
    """ 
    Reads scans and segmentations in the data folder
    
    Args:
        data_folder: main folder where the individual patient folders are
        verbose: choose if prints are shown
    Outputs:
        scans, segmentations: scans and segmentations read from .nrrd files
        patient_names: list of patient names
    """

    
    scans = []
    segms = []

    scan_paths = [name for name in glob.glob(data_path + '*/scan.nrrd')]
    segm_paths = [name for name in glob.glob(data_path + '*/segm.nrrd')]
    patient_names = [name.split('/')[-2] for name in scan_paths]

    for i in range(len(scan_paths)):
        scan,_ = nrrd.read(scan_paths[i])
        segm,_ = nrrd.read(segm_paths[i])
            
        # Print
        if verbose == 1:
            print(f"Patient: {patient_names[i]}, Shape: {scan.shape}")
            
        scans.append(scan)
        segms.append(segm.astype(np.uint8))
    
    return scans, segms, patient_names

def dice_coef(y_true, y_pred):
    """
    Dice coefficient for binary labels
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))


def dice_coef_multiclass(y_true, y_pred, n_classes, one_hot=False):
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
        y_true = np.expand_dims(y_true, axis=3)
        y_true = to_categorical(y_true, num_classes=n_classes)
        y_pred = np.expand_dims(y_pred, axis=3)
        y_pred = to_categorical(y_pred, num_classes=n_classes)
        
    for i in range(n_classes):
        # Compute dice_coef of each class
        dice_coefs.append(dice_coef(y_true[:,:,:,i], y_pred[:,:,:,i]))
                          
    return np.array(dice_coefs)


def restore_from_patches_onehot(out_shape, patches,xstep=12,ystep=12,zstep=12):
    """ 
    Unpatchifies taking into account overlap, by averaging.
    patches has shape (#x patches, #y patches, #z patches, patch_shape, n_classes)
    """
    patch_shape = patches.shape[-4:-1]
    
    n_classes = patches.shape[-1]
    out_total = np.zeros(out_shape + (n_classes,), patches.dtype)
    denom = np.zeros(out_shape, patches.dtype)

    # Loop through each class
    for n in range(n_classes):
        out = np.zeros(out_shape, patches.dtype)
        patches_6D = np.lib.stride_tricks.as_strided(out, ((out.shape[0] - patch_shape[0]) // xstep+1, (out.shape[1] - patch_shape[1]) // ystep+1,
                                                      (out.shape[2] - patch_shape[2]) // zstep+1, patch_shape[0], patch_shape[1], patch_shape[2]),
                                                      (out.strides[0] * xstep, out.strides[1] * ystep,out.strides[2] * zstep, out.strides[0], out.strides[1],out.strides[2]))
        if n==0:
            # only need to compute denom 1 time
            denom_6D = np.lib.stride_tricks.as_strided(denom, ((denom.shape[0] - patch_shape[0]) // xstep+1, (denom.shape[1] - patch_shape[1]) // ystep+1,
                                                          (denom.shape[2] - patch_shape[2]) // zstep+1, patch_shape[0], patch_shape[1], patch_shape[2]),
                                                          (denom.strides[0] * xstep, denom.strides[1] * ystep,denom.strides[2] * zstep, denom.strides[0], denom.strides[1],denom.strides[2]))
            np.add.at(denom_6D, tuple(x.ravel() for x in np.indices(patches_6D.shape)), 1)

        np.add.at(patches_6D, tuple(x.ravel() for x in np.indices(patches_6D.shape)), patches[:,:,:,:,:,:,n].ravel())
        out_total[:,:,:,n] = out/denom
        
    return out_total

def read_patch_data_saiad(patch_data_folder, skip_patient = None):
    """
    Reads patches and appends to numpy array. Skips the test patient chosen.
    Args:
        patch_data_folder: Location of the patient folders
        skip_patient: which patient to skip (e.g. 'SAIAD 2')
    Outputs:
        X: Patchified scans in array of dimensions #patches x patch_dimension x 3 (rgb channels)
        y: Patchified segms in array of dimensions #patches x patch_dimension x n_classes (one-hot encoding)
    """
    i = 0
    patient_names = []
    for name in glob.glob(patch_data_folder + '/*/'):
        
        if name.split('/')[-2] != skip_patient:

            # Read .npy data
            scan_patches = np.load(name + 'scan_patches.npy')
            segm_patches = np.load(name + 'segm_patches.npy')

            if i==0:
                # Initialize the np.array of patches
                X = scan_patches
                y = segm_patches
                
            else:
                # Append to already initialized arrays
                X = np.concatenate((X,scan_patches))
                y = np.concatenate((y,segm_patches))
                
            i+=1
            
    return X,y