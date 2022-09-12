import copy
import nrrd
import nibabel as nib
import numpy as np
import os
import glob
from sklearn.utils import shuffle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import random_split
from unet3d.config import *
from torch.nn.functional import one_hot
from scipy.ndimage import gaussian_filter



def get_class_weights_torch(y_train, n_classes):
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
        class_weights.append(1/torch.count_nonzero(y_train[:,:,:,i]).item())
    return class_weights

def sample(probabilities, n=1):
    """ Sample from probabability distribution of any dimension, n times"""
    choices = np.prod(probabilities.shape)
    index = np.random.choice(choices, size=n, p=probabilities.ravel())
    return np.unravel_index(index, shape=probabilities.shape)


class SAIADDataset(Dataset):
    """SAIAD dataset."""

    def __init__(self,  data_folder = DATASET_PATH,
                        n_batches = TRAIN_BATCHES_PER_EPOCH,
                        patch_size = PATCH_SIZE, 
                        batch_size = TRAIN_BATCH_SIZE, 
                        epochs = EPOCHS,
                        transform = None, 
                        non_unif_sampling = True, 
                        n_classes = NUM_CLASSES, 
                        excl_patients = [], 
                        load_data_to_memory = False,
                        sigma=5, truncate=5):
        """
        Args:
            data_folder (string): /path/to/patients/ 
            n_batches (int): number of batches per epoch
            patch_size (tuple): patch size to be used ex:(64,64,64)
            batch_size (int): number of patches per batch
            epochs (int): number of epochs
            transform (Compose): data augmentations
            non_unif_sampling (bool): if True, use the non uniform patch sampling, if False use uniform
            n_classes (int): number of classes including background
            excl_patients (list): patients to exclude from training, ex. ['SAIAD 1', 'SAIAD 2']
            load_data_to_memory (bool): if True, loads all patients scans and segms to memory, faster but more RAM used
            sigma, truncate (float): params for blurring in the creation of sampling distribution

        """
        self.data_folder = data_folder
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.epochs = epochs
        self.transform = transform
        self.n_classes = n_classes
        self.non_unif_sampling = non_unif_sampling
        self.patients_list = self.get_patients_list(excl_patients)
        self.load_data_to_memory = load_data_to_memory
        self.loaded_scans, self.loaded_segms = self.load_data()
        self.patients_probabilities = self.get_patients_probabilities(sigma, truncate)
        self.patients_centers = self.get_patients_centers()
        self.patients_centers_counter = [0 for i in range(len(self.patients_list))]
        self.class_weights = []


    def __len__(self):
        """Return total number of patches to be used"""
        return self.n_batches*self.batch_size


    def get_patients_list(self, excl_patients):
        return list([name for name in glob.glob(self.data_folder+'*') 
                    if not name.split('/')[-1] in excl_patients])
                    

    def load_data(self):
        """ Load all training data (scans, segms) to memory """
        scans = []
        segms = []
        self.loaded_segms = []
        if self.load_data_to_memory:
            for patient in self.patients_list:
                scan,_ = nrrd.read(patient+'/scan.nrrd')
                segm, _ = nrrd.read(patient+'/segm.nrrd')
                scans.append(scan)
                segms.append(segm)
        return scans,segms


    def get_patients_probabilities(self, sigma, truncate):
        """ Returns list of probabilities distribution for sampling.
            It also saves the distribution in each patient's folder"""

        patients_probs = []
        if self.non_unif_sampling == True:
            print("Fetching patients probabilities...")
            
            for patient in tqdm(self.patients_list):
                if len(glob.glob(patient+'/sampling_dist.npy'))>=1:
                    # Prob already exists
                    patients_probs.append(np.load(patient+'/sampling_dist.npy'))
                    
                else:
                    segm,_ = nrrd.read(patient+'/segm.nrrd')
                    segm = torch.tensor(segm).int()

                    ## For class weights ##
                    segm_onehot = one_hot(segm.to(torch.int64), num_classes=self.n_classes)
                    weights = get_class_weights_torch(segm_onehot, n_classes=self.n_classes)

                    # Create 3D probability map based on segm, by blurring it and setting each class to certain weight
                    segm_blurred = segm.to(float)
                    for i in range(self.n_classes):
                        segm_blurred[segm == i] = weights[i]

                    segm_blurred = torch.tensor(gaussian_filter(segm_blurred, sigma=sigma, truncate=truncate))
                    segm_blurred /= segm_blurred.sum()
                    patients_probs.append(segm_blurred)
                    with open(patient+'/sampling_dist.npy', 'wb') as f:
                        np.save(f, segm_blurred)
                        
        return patients_probs
    

    def get_patients_centers(self):
        """ 
        Generate maximum number of possible centers per patient. 
        This is much faster than doing it 1 at a time.
        
        Returns array = # patients x # samples x 3 ->(x,y,z)
        """
        patients_centers = []
        if self.non_unif_sampling == True:
            for i in range(len(self.patients_list)):
                samp = sample(self.patients_probabilities[i], n=self.batch_size*self.n_batches*self.epochs)
                samp = np.swapaxes(samp, 0, 1)
                patients_centers.append(samp)
        return patients_centers

    def __get_patches_from_patient(self, i_patient):
        """ Get patch (scan and segm) from patient 
        Args:
            i_patient: index of the patient for the patch
        Outputs:
            patch_scan: patch of the scan with shape=1,x,y,z
            patch_segm: one-hot patch of the segm with shape=n_classes,x,y,z"""
        if self.load_data_to_memory:
            scan = self.loaded_scans[i_patient]
            segm = self.loaded_segms[i_patient]
        else:
            scan, _ = nrrd.read(self.patients_list[i_patient]+'/scan.nrrd')
            segm, _ = nrrd.read(self.patients_list[i_patient]+'/segm.nrrd')

        scan_shape = scan.shape

        if self.non_unif_sampling == True:
            # Non uniform sampling
            cx, cy, cz = self.patients_centers[i_patient][self.patients_centers_counter[i_patient]]
            self.patients_centers_counter[i_patient]+=1
        else:
            #Uniform sampling
            cx = np.random.randint(0,scan.shape[0])
            cy = np.random.randint(0,scan.shape[1])
            cz = np.random.randint(0,scan.shape[2])
            
        # Get valid bbox
        bbox_x = [max(cx - self.patch_size[0]//2, 0), min(scan_shape[0], cx+self.patch_size[0]//2)]
        bbox_y = [max(cy - self.patch_size[1]//2, 0), min(scan_shape[1], cy+self.patch_size[1]//2)]
        bbox_z = [max(cz - self.patch_size[2]//2, 0), min(scan_shape[2], cz+self.patch_size[2]//2)]

        # Get padding amounts
        pad_x = (-min(cx - self.patch_size[0]//2,0), max(self.patch_size[0]//2 + cx - scan_shape[0], 0))
        pad_y = (-min(cy - self.patch_size[1]//2,0), max(self.patch_size[1]//2 + cy - scan_shape[1], 0))
        pad_z = (-min(cz - self.patch_size[2]//2,0), max(self.patch_size[2]//2 + cz - scan_shape[2], 0))

        patch_scan = scan[bbox_x[0]:bbox_x[1], bbox_y[0]:bbox_y[1], bbox_z[0]:bbox_z[1]]
        patch_segm = segm[bbox_x[0]:bbox_x[1], bbox_y[0]:bbox_y[1], bbox_z[0]:bbox_z[1]]

        patch_scan = np.pad(patch_scan,(pad_x, pad_y, pad_z), 'constant', constant_values=0)
        patch_segm = np.pad(patch_segm,(pad_x, pad_y, pad_z), 'constant', constant_values=0)
        
        patch_scan = torch.unsqueeze(torch.tensor(patch_scan), 0)
        patch_segm = one_hot(torch.tensor(patch_segm, dtype=torch.int64), num_classes=self.n_classes).permute(3,0,1,2)
        return patch_scan, patch_segm


    def __getitem__(self, idx):
        """Returns 1 scan patch and 1 segm patch"""
        # Get random patient indexes
        patient_idx = torch.randint(0, len(self.patients_list), (1,))
    
        X_batch, y_batch = self.__get_patches_from_patient(patient_idx.item())

        return X_batch.float(), y_batch.float()

    
    
class WrappedDataLoader:
    """ Wrapper to output batches to GPU as they come"""
    def __init__(self, dl, func, args):
        self.dl = dl
        self.func = func
        self.args = args

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b, self.args))
            
def to_device(x, y, dev):
    """ Send batch to device"""
    return x.to(dev), y.to(dev)