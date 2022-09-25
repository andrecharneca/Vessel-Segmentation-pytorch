# Includes pre-processing functions #

import numpy as np
#from volumentations import *
from patchify import patchify, unpatchify
#from tensorflow.keras.utils import to_categorical
import sys
import nibabel as nib
#from starter_code.utils import load_case
import nrrd
import os
import glob
import SimpleITK as sitk

def resample_volume(volume_path, interpolator = sitk.sitkLinear, new_spacing = [0.78162485, 0.78162485, 3.0]):
    """
    Resamples volume to specified voxel spacings.
    Function from: https://discourse.itk.org/t/resample-volume-to-specific-voxel-spacing-simpleitk/3531
    """
    
    volume = sitk.ReadImage(volume_path, sitk.sitkFloat32) # read and cast to float32
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]

    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())

def cut_or_pad_xy(scan, segm, xshape=512, yshape=512):
    """
    Adds padding or slices x,y dimensions of scan.
    """
    cut_x = scan.shape[0] - xshape
    cut_y = scan.shape[1] - yshape
    new_scan = np.copy(scan)
    new_segm = np.copy(segm)

    # Check if the scan needs cutting
    if cut_x > 0:
        new_scan = new_scan[cut_x//2: -cut_x//2, :, :]
        new_segm = new_segm[cut_x//2: -cut_x//2, :, :]
    if cut_y > 0:
        new_scan = new_scan[:, cut_y//2: -cut_y//2, :]
        new_segm = new_segm[:, cut_y//2: -cut_y//2, :]

    # Check if scan needs padding
    if cut_x < 0:
        new_scan = np.pad(new_scan, ((-cut_x//2, -cut_x//2), (0,0), (0,0)), constant_values = -1000)
        new_segm = np.pad(new_segm, ((-cut_x//2, -cut_x//2), (0,0), (0,0)), constant_values = 0)
    if cut_y < 0:
        new_scan = np.pad(new_scan, ((0,0), (-cut_y//2,-cut_y//2), (0,0)), constant_values = -1000)
        new_segm = np.pad(new_segm, ((0,0), (-cut_y//2,-cut_y//2), (0,0)), constant_values = 0)

    return new_scan, new_segm

def convert_255(img):
    """
    Convert 'img' pixel values to [0,255] uint8
    """
    img_255 = np.copy(img)
    img_255 -= img.min()
    img_255 = img_255 * 255/img_255.max()
    return img_255.astype(np.uint8)

def apply_intensity_window(img, min_int, max_int):
    """
    All pixel values with intensity < min_int or > max_int
    are set to the minimum or maximum, respectively.
    """
    img_new = np.copy(img)
    
    # Find indices of values < min_int
    below_min_index = np.nonzero(img_new < min_int)
    above_max_index = np.nonzero(img_new > max_int)

    # Convert those values to min_int
    img_new[below_min_index] = min_int
    img_new[above_max_index] = max_int
    
    return img_new.astype(float)

def remove_slices(img, segm, classes):
    """
    Removes the beggining and end slices that don't have
    the segmentation in 'classes'
    """
    n_slices = segm.shape[2]
    
    # Loop through first slices
    cut_beggin = 0
    for i in range(n_slices):
        slice_ = segm[:,:,i]
        
        # At the 1st slice with at least 1 of the classes, stop cutting
        if np.any(np.isin(classes, slice_)):
            cut_beggin = i
            break
            
    
    # Same for last slices, starting from end
    cut_end = 0
    for i in range(n_slices):
        slice_ = segm[:,:,-i]
        
        # At the 1st slice with at least 1 of the classes, stop cutting
        if np.any(np.isin(classes, slice_)):
            cut_end = i
            break
    
    return img[:,:,cut_beggin:(n_slices-cut_end+1)], segm[:,:,cut_beggin:(n_slices-cut_end+1)]

def add_padding(img, segm):
    """
    Adds zero-padding in z-axis until the number of slices
    is a multiple of 64
    """
    img_copy = np.copy(img)
    segm_copy = np.copy(segm)
    
    n_slices = img.shape[2]
    
    # Find next multiple of 64
    if n_slices/64 == int(n_slices/64):
        return img_copy, segm_copy
    else:
        next_mult_64 = 64*(int(n_slices/64) + 1)
    
    # Pad
    pad_amount = (next_mult_64-n_slices)/2

    # Case: pad_amount is even on both sides
    if pad_amount == int(pad_amount):
        pad_amount = int(pad_amount)
        img_copy = np.pad(img, ((0,0), (0,0), (pad_amount, pad_amount)), constant_values = 0)
        segm_copy = np.pad(segm, ((0,0), (0,0), (pad_amount, pad_amount)), constant_values = 0)
        
    else: # Case: pad amount is odd
        pad_amount = int(pad_amount)
        img_copy = np.pad(img, ((0,0), (0,0), (pad_amount, pad_amount+1)), constant_values = 0)
        segm_copy = np.pad(segm, ((0,0), (0,0), (pad_amount, pad_amount+1)), constant_values = 0)

    return img_copy, segm_copy
    
def unify_classes(segm, name):
    """
    - Makes the kidneys (classes 4,5) to just one class (=4)
    - Also does different things depending on what scan it is   
    """
    
    # Scan-specific changes
    if "SAIAD 17 vasculaire" in name:
        #class=6 is nothing here
        segm[segm==6] = 0
        
    if ("SAIAD 18 BIS vasculaire" in name) or ("SAIAD 18 TER vasculaire" in name):
        #class=6 is renal cavities (marked as kidney)
        #class=7 is nothing
        
        segm[segm==7] = 0
        segm[segm==6] = 4
    
    if "SAIAD 16 bis vasculaire" in name:
        #class=6 is intestins (marked as background)
        
        segm[segm==6] = 0
    
    if "SAIAD 20 vasculaire" in name:
        # labels 5,4 should be 2,3 
        segm_copy = np.copy(segm)
        segm_copy[segm==5] = 2
        segm_copy[segm==2] = 5
        segm_copy[segm==4] = 3
        segm_copy[segm==3] = 4
        segm = segm_copy
    
    segm[segm==5] = 4

        
    return segm


def get_augmentation(patch_size=64,p_Rotate=0,p_ElasticTransform=0,
                     p_Flip0=0,p_Flip1=0,p_Flip2=0,p_RandomRotate90=0,
                     p_GaussianNoise=0, p_RandomGamma=0, Rot_z_limit = 0, Rot_y_limit = 0, Rot_x_limit = 0):
    """  
    Defines the augmentations we want to do.
    Args: 
        patch_size = size of cubic patch
        p_[augmentation] = probability of performing it
        Rot_[x,y,x]_limit = random rotations will be performed in the range [-limit,limit]
    
    Note: Resize throws error
    """
    return Compose([
        Rotate((-Rot_x_limit,Rot_x_limit), (-Rot_y_limit, Rot_y_limit), (-Rot_z_limit, Rot_z_limit), p=p_Rotate),
        ElasticTransform((0, 0.1), interpolation=2, p=p_ElasticTransform),
        Flip(0, p=p_Flip0),
        Flip(1, p=p_Flip1),
        Flip(2, p=p_Flip2),
        RandomRotate90((0,1), p=p_RandomRotate90),
        GaussianNoise(var_limit=(0, 3), p=p_GaussianNoise),
        RandomGamma(gamma_limit=(70,160), p=p_RandomGamma),
    ], p=1.0)


def read_data_and_resample(data_path, voxel_spacings=[0.5,0.5,1], verbose = 0):
    """ 
    Reads scans and segmentations in the data folder, and resamples them to the
    specified spacings.
    
    Args:
        data_folder: main folder where the individual patient folders are
        verbose: choose if prints are shown
        voxel_spacings: [x,y,z] voxel spacings to resample to
    Outputs:
        scans, segmentations: scans and segmentations read from .nrrd files
        headers_scans, headers_segms: headers of the original .nrrd files, with changed
                                    spacings only.
        patient_names: list of patient names
    """

    resampled_scans = []
    resampled_segms = []
    headers_scans = []
    headers_segm = []

    scan_paths = [name for name in glob.glob(data_path + '*/scan.nrrd')]
    segm_paths = [name for name in glob.glob(data_path + '*/segm.nrrd')]
    patient_names = [name.split('/')[-2] for name in scan_paths]

    for i in range(len(scan_paths)):
        # Get headers
        _, header_scan = nrrd.read(scan_paths[i])
        _, header_segm = nrrd.read(segm_paths[i])
        
        # Change spacings in header
        header_scan['space directions'] = header_segm['space directions'] = np.diag(voxel_spacings)
        
        # Resample volumes with new voxel spacings
        resampled_scan = resample_volume(scan_paths[i], new_spacing = voxel_spacings)
        resampled_segm = resample_volume(segm_paths[i], new_spacing = voxel_spacings, 
                                         interpolator = sitk.sitkNearestNeighbor)
        
        # Convert axes to (width, height, slices) where slice = 0 is at the bottom of patient
        scan_data_aligned = np.swapaxes(sitk.GetArrayFromImage(resampled_scan), 0, 2)[:,:,::-1]
        segm_data_aligned = np.swapaxes(sitk.GetArrayFromImage(resampled_segm), 0, 2)[:,:,::-1]
        
        
        # Cut or pad height x width to get to 512 x 512
        scan, segm = cut_or_pad_xy(scan_data_aligned, segm_data_aligned, xshape=512, yshape=512)
        
        # Print
        if verbose == 1:
            print(f"Patient: {patient_names[i]}, Shape: {scan.shape}")
            
        resampled_scans.append(scan)
        resampled_segms.append(segm.astype(np.uint8))
        headers_scans.append(header_scan)
        headers_segm.append(header_segm)
    
    return resampled_scans, resampled_segms, headers_scans, headers_segm,  patient_names
    
def process_data_saiad(scans, segmentations, patient_names, verbose=0):
    """
    Processes the SAIAD data, this includes:
     - Cutting the intensity window to [-200,600]
     - Rescale to [0,255] values (needed for segmentation-models-3D)
     - Remove the slices that don't have vessel segmentations.
     - Pad (zero) in z-axis until nearest multiple of 64, so patches of 64x64x64 can be done.
     - Convert the 2 labels assigned for kidneys into just 1, and remove the extra classes.
          - Renal cavities were converted to kidneys here
    Args:
        scans, segmentations: 3D arrays read from read_data
        patient_names: list of patient names, in the same order as the scans and segmentations
        verbose: choose if prints are shown
    Output:
        scans, segmentations: prepared data
    """
    
    for i in range(len(scans)):
        if verbose: print("\nProcessing scan ", patient_names[i], end=" , ")
            
        # Cutting intensity window
        scans[i] = apply_intensity_window(scans[i], -200,600)

        # Convert to [0,255]
        scans[i] = convert_255(scans[i])

        # Remove slices that don't have vessels (class=2,3)
        classes = [2,3]
        scans[i], segmentations[i] = remove_slices(scans[i], segmentations[i], classes)
        
        """
        if patient_names[i].split("/")[-1] == 'SAIAD 1':
            # For some reason, for this patient "remove_slices" doesn't work
            scans[i] = scans[i][:,:,0:164]
            segmentations[i] = segmentations[i][:,:,0:164]
        """

        # Pad z-axis
        scans[i], segmentations[i] = add_padding(scans[i], segmentations[i])

        # Unify classes and correct some labeling mistakes
        segmentations[i] = unify_classes(segmentations[i], patient_names[i])

        if verbose: print("Shape:", scans[i].shape)
    
    return scans, segmentations

def process_data_kits(kits_data_path, processed_data_write_path, voxel_spacing = [0.5, 0.5, 1.0], verbose=0):
    """
    Process KiTS data. Besides the same processing as the SAIAD data, also resamples the scans
    and rotates them to be consistent with SAIAD.
    
    Writes the newly processed data to a specified path.
    """    
    # Loop through cases
    for i in range(210):
        patient_folder = kits_data_path + 'case_' + str(i).zfill(5)
        
        # Resample according to voxel spacing
        # Take into account KiTS data is (Z,Y,X) but we want (X,Y,Z)
        voxel_spacing_kits = [voxel_spacing[2], voxel_spacing[1], voxel_spacing[0]]
        sitk_vol = resample_volume(patient_folder+'/imaging.nii.gz', new_spacing=voxel_spacing_kits) 
        sitk_segm = resample_volume(patient_folder+'/segmentation.nii.gz', new_spacing=voxel_spacing_kits, 
                              interpolator=sitk.sitkNearestNeighbor) 

        scan = sitk.GetArrayFromImage(sitk_vol)
        segm = sitk.GetArrayFromImage(sitk_segm)
        
        # Flip Z to be consistent with SAIAD data
        scan = scan[:,:,::-1]
        segm = segm[:,:,::-1]

        # Rescale x-y to 512x512 if needed
        scan, segm = cut_or_pad_xy(scan, segm, xshape=512, yshape=512)
        
        # Cutting intensity window
        scan = apply_intensity_window(scan, -200,600)

        # Convert to [0,255]
        scan = convert_255(scan)

        # Remove slices that don't have segmentations (class=1,2)
        classes = [1,2]
        scan, segm = remove_slices(scan, segm, classes)

        # Pad z-axis
        scan, segm = add_padding(scan, segm)
        
        # Write to processed folder
        patient_folder_write = processed_data_write_path + 'case_' + str(i).zfill(5)

        os.mkdir(patient_folder_write)
        nrrd.write(patient_folder_write + '/scan.nrrd', scan)
        nrrd.write(patient_folder_write + '/segm.nrrd', segm)
        
        # Verbose
        if verbose == 1:
            print(f"case {i} = ",scan.shape)
    
def patchify_saiad(scans, segmentations, n_classes, verbose=1):
    """
    Processing and patchify training data for the segmentation-models-3D package, this includes:
    - Separate data into 64x64x64 patches.
    - Put all patches into a 4D array: #patches x dimensions
    - Converting the gray scale images into RGB (by copying the values 3x)
    - Convert segmentations into one-hot encoding.
    
    Args:
        scans, segmentations: prepared data from process_saiad_data function
        n_classes: number of classes of segmentation (including background)
        verbose: choose if prints are shown
    Outputs:
        scans_patches_list: List of patchified scans in array of dimensions #scans x #patches x patch_dimension x 3 (rgb channels)
        segm_patches_list: List of patchified segms in array of dimensions #scans x #patches x patch_dimension x n_classes (one-hot encoding)
    """
    scans_patches_list = []
    segm_patches_list = []
    
    for i in range(len(scans)):
        
        if verbose: print("Processing scan ", i+1)
        scan = scans[i]
        segm = segmentations[i]

        # Patchify the scan/segm
        scan_patches = patchify(scan, (64,64,64), step=64)  #Step=64 for 64 patches means no overlap
        segm_patches = patchify(segm, (64,64,64), step=64)

        # Re-shape into 4D array
        scan_patches = np.reshape(scan_patches, (-1, scan_patches.shape[3], scan_patches.shape[4], scan_patches.shape[5]))
        segm_patches = np.reshape(segm_patches, (-1, segm_patches.shape[3], segm_patches.shape[4], segm_patches.shape[5]))

        # Convert to RGB channels
        scan_patches = np.stack((scan_patches,)*3, axis=-1)

        # Convert segmentations to one-hot
        segm_patches = np.expand_dims(segm_patches, axis=4)
        segm_patches = to_categorical(segm_patches, num_classes=n_classes)
    
        # Append to lists
        scans_patches_list.append(scan_patches.astype(np.float32))
        segm_patches_list.append(segm_patches.astype(np.uint8))
    return scans_patches_list, segm_patches_list

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

def patchify_kits(processed_kits_path, n_classes, remove_portion, verbose=0):
    """
    Processing and patchify KiTS 2019 data for the segmentation-models-3D package, this includes:
    - Separate data into 64x64x64 patches.
    - Put all patches into a 4D array: #patches x dimensions
    - Converting the gray scale images into RGB (by copying the values 3x)
    - Convert segmentations into one-hot encoding.
    - Remove a portion of "just background" patches
    
    Args:
        processed_kits_path: path to folder with processed data, with subfolders case_00xxx
        n_classes: number of classes of segmentation (including background)
        remove_portion: portion of "just background" patches to remove
        verbose: choose if prints are shown
    Outputs:
        X: all scan patches in array of dimensions #patches x patch_dimension x 3 (rgb channels)
        y: all segmentation patches in array of dimensions #patches x patch_dimension x 5 (one-hot encoding)
    """
    
    for i,name in enumerate(glob.glob(processed_kits_path + '/case_00***/')):
        if verbose==1:
            print(name)
            
        segm, _ = nrrd.read(name + '/segm.nrrd')
        scan, _ = nrrd.read(name + '/scan.nrrd')

        # Patchify the scan/segm
        scan_patches = patchify(scan, (64,64,64), step=64)  #Step=64 for 64 patches means no overlap
        segm_patches = patchify(segm, (64,64,64), step=64)

        # Re-shape into 4D array
        scan_patches = np.reshape(scan_patches, (-1, scan_patches.shape[3], scan_patches.shape[4], scan_patches.shape[5]))
        segm_patches = np.reshape(segm_patches, (-1, segm_patches.shape[3], segm_patches.shape[4], segm_patches.shape[5]))

        # Convert to RGB channels
        scan_patches = np.stack((scan_patches,)*3, axis=-1)

        # Convert segmentations to one-hot
        segm_patches = np.expand_dims(segm_patches, axis=4)
        segm_patches = to_categorical(segm_patches, num_classes=n_classes)

        # Remove percentage of background patches
        scan_patches_removed, segm_patches_removed = remove_background_patches(scan_patches, segm_patches, 
                                                                               patch_size=64, remove_portion = remove_portion)

        # Append to X,y arrays
        if i==0:
                # Initialize the np.array of patches
                X = scan_patches_removed.astype(np.float32)
                y = segm_patches_removed.astype(np.uint8) # Cast the type, otherwise it'll be float32 and occupy a lot of ram
        else:
            # Append to already initialized arrays
            X = np.concatenate((X,scan_patches_removed.astype(np.float32)))
            y = np.concatenate((y,segm_patches_removed.astype(np.uint8)))
            
    return X,y

def remove_background_patches(scan_patches, segm_patches, patch_size=64, remove_portion = 0.9):
    """ 
    Remove a certain percentage of patches that are just background
    Args:
        scan_patches, segm_patches: patchified data (segmentations in one-hot)
        remove_portion: portion of background patches to remove
        
    Output:
        scan_patches_removed, segm_patches_removed: patchified data with background patches removed
    """
    scan_patches_removed = []
    segm_patches_removed = []
    np.random.seed(7)
    for i in range(len(segm_patches)):
        
        if np.count_nonzero(segm_patches[i,:,:,:,0]==1) < patch_size**3:
            
            # Case: not all pixels are background
            scan_patches_removed.append(scan_patches[i])
            segm_patches_removed.append(segm_patches[i])
            
        elif np.count_nonzero(segm_patches[i,:,:,:,0]==1) == patch_size**3 and np.random.rand() > remove_portion:
            
            # Case: all pixels are background but keep it anyway
            scan_patches_removed.append(scan_patches[i])
            segm_patches_removed.append(segm_patches[i])
            
    return np.array(scan_patches_removed), np.array(segm_patches_removed)
