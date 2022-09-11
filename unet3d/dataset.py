import copy
import nrrd
import nibabel as nib
import numpy as np
import os
import glob
from sklearn.utils import shuffle
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
        """ Returns list of probabilities distribution for sampling
            NOTE: probably better to have the probabilities in each patient folder and just read"""

        patients_probs = []
        if self.non_unif_sampling == True:
            print("Generating patients probabilities...")
            for patient in self.patients_list:
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

    def __len__(self):
        return len(self.n_batches)

    def __getitem__(self, idx):
        pass
