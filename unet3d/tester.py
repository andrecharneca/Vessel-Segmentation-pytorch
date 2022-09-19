## Classes and functions for Testing on patients: Inference and Post-Processing ##
import numpy as np
import glob
import torch
import nrrd
from patchify import patchify,unpatchify
from scipy.ndimage import gaussian_filter
import cc3d
from utils.Visualization import ImageSliceViewer3D
from unet3d.config import *
from unet3d.unet3d_vgg16 import UNet3D_VGG16

class Tester():
    """
    Class for testing the models
    """
    
    def __init__(self,
        model_path, 
        test_patient,  
        device = "cuda"
        n_classes=NUM_CLASSES, 
        patch_size=PATCH_SIZE, 
        step_size=(PATCH_SIZE[0]//2, PATCH_SIZE[1]//2, PATCH_SIZE[2]//2),
        ):
        """
        Modified from the main branch to work with variable patch_size and 1channel input. No preprocess_input applied.
        Args:
            model_path: path to .pth trained model state_dict
            test_patient: Name of test patient with folder with scan.nrrd and segm.nrrd files (e.g. 'SAIAD 1')
        """
        self.device = torch.device(device)
        self.model = load_model(model_path)
        self.n_classes = n_classes
        self.step_size = step_size
        self.patch_size = patch_size
        
        self.test_patient = test_patient
        self.original_spacing = None
        self.current_spacing = [0.5,0.5,1.]
        self.scan = None
        self.scan_patchified = None
        self.scan_patchified_shape = None
        self.truth_segm = None
        self.pred_segm_index = None # Classes as 0,1,2...
        self.pred_segm_onehot = None # Classes as one-hot encoding
        self.paddings = []
        
        # Folders to save segmentations in
        self.uniform_spacing_folder = '../Data/Predicted Segms UnifSpacing/'
        self.original_spacing_folder = '../Data/Predicted Segms OriginSpacing/'
        
        # Folders to read out of
        self.original_data_folder = '../Data/SAIAD_data_cleared/'
        self.processed_data_folder = DATASET_PATH


    def load_model(self, model_path):
        model = UNet3D_VGG16(
            in_channels=IN_CHANNELS, 
            num_classes=NUM_CLASSES,
            use_softmax_end=True
            ).to(self.device)
        model.load_state_dict(torch.load(model_path))
        return model


    def read_test_patient_data_and_pad(self):
        """
        Read test patient scan, segm and get the original spacing.
        Also 0-pads until the next multiple of step_size, for each direction.
        """
        self.scan, _ = nrrd.read(self.processed_data_folder + self.test_patient + '/scan.nrrd')
        self.truth_segm, _ = nrrd.read(self.processed_data_folder + self.test_patient + '/segm.nrrd')
        
        # Compute pad amounts
        pad_amounts = []
        for i in range(3):
            if self.scan.shape[i]/self.step_size[i] != int(self.scan.shape[i]/self.step_size[i]):
                next_mult = self.step_size[i]*(int(self.scan.shape[i]/self.step_size[i]) + 1) 
            else:
                next_mult = self.scan.shape[i]
            pad_amounts.append((next_mult-self.scan.shape[i])/2)
        
        # Pad each direction (the extra factor is to fix the case where the padding is odd)
        paddings = [(int(pad_amounts[i]), int(pad_amounts[i] + 2*(pad_amounts[i]-int(pad_amounts[i])))) for i in range(3)]
        
        self.scan = np.pad(self.scan, (paddings[0],(0,0),(0,0)), constant_values = 0)
        self.scan = np.pad(self.scan, ((0,0), paddings[1],(0,0)), constant_values = 0)
        self.scan = np.pad(self.scan, ((0,0),(0,0),paddings[2]), constant_values = 0)
        self.truth_segm = np.pad(self.truth_segm, (paddings[0],(0,0),(0,0)), constant_values = 0)
        self.truth_segm = np.pad(self.truth_segm, ((0,0), paddings[1],(0,0)), constant_values = 0)
        self.truth_segm = np.pad(self.truth_segm, ((0,0),(0,0),paddings[2]), constant_values = 0)


        _,header = nrrd.read(self.original_data_folder + self.test_patient +'/segm.nrrd')
        self.original_spacing = np.diagonal(header['space directions'])

        # Save the paddings so we can remove them later
        self.paddings = paddings
    
    def unpad_scan_and_truth_segm(self):
        """ Unpadding function. Use this after inference, so the 'space directions' don't get screwed"""
        self.scan = self.scan[self.paddings[0][0]:(self.scan.shape[0]-self.paddings[0][1]),
                        self.paddings[1][0]:(self.scan.shape[1]-self.paddings[1][1]),
                        self.paddings[2][0]:(self.scan.shape[2]-self.paddings[2][1])]
        self.truth_segm = self.truth_segm[self.paddings[0][0]:(self.truth_segm.shape[0]-self.paddings[0][1]),
                        self.paddings[1][0]:(self.truth_segm.shape[1]-self.paddings[1][1]),
                        self.paddings[2][0]:(self.truth_segm.shape[2]-self.paddings[2][1])]

        if type(self.pred_segm_index) is np.ndarray:
            self.pred_segm_index = self.pred_segm_index[self.paddings[0][0]:(self.pred_segm_index.shape[0]-self.paddings[0][1]),
                        self.paddings[1][0]:(self.pred_segm_index.shape[1]-self.paddings[1][1]),
                        self.paddings[2][0]:(self.pred_segm_index.shape[2]-self.paddings[2][1])]
        
        if type(self.pred_segm_onehot) is np.ndarray:
            self.pred_segm_onehot = self.pred_segm_onehot[self.paddings[0][0]:(self.pred_segm_onehot.shape[0]-self.paddings[0][1]),
                        self.paddings[1][0]:(self.pred_segm_onehot.shape[1]-self.paddings[1][1]),
                        self.paddings[2][0]:(self.pred_segm_onehot.shape[2]-self.paddings[2][1])]
                           
    def patchify_scan(self):
        """
        Patchify test scan
        """
        self.scan_patchified = patchify(self.scan, self.patch_size, step=self.step_size)
        self.scan_patchified_shape = self.scan_patchified.shape
    
    def save_uniform_spacing_segm(self):
        """
        Saves the truth and prediction segm with the uniform spacing 
        We need to do this also for truth because we've padded it, otherwise we won't be able to compute dice
        """
        _, unif_header = nrrd.read(self.processed_data_folder + self.test_patient + '/segm.nrrd')
        nrrd.write(self.uniform_spacing_folder + self.test_patient + '_pred_segm.nrrd', self.pred_segm_index, unif_header)
        nrrd.write(self.uniform_spacing_folder + self.test_patient + '_truth_segm.nrrd', self.truth_segm, unif_header)

    
    def resample_to_original_and_save(self):
        """
        Resample predicted segm to original spacing and save to OriginSpacing folder
        """
        # Resample volumes with new voxel spacings
        scan_path = self.processed_data_folder + self.test_patient + '/scan.nrrd'
        truth_segm_path = self.uniform_spacing_folder + self.test_patient + '_truth_segm.nrrd'
        segm_path = self.uniform_spacing_folder + self.test_patient + '_pred_segm.nrrd'
        
        resampled_scan = resample_volume(scan_path, new_spacing = self.original_spacing)
        resampled_segm = resample_volume(segm_path, new_spacing = self.original_spacing, 
                                         interpolator = sitk.sitkNearestNeighbor)
        resampled_truth_segm = resample_volume(truth_segm_path, new_spacing = self.original_spacing, 
                                         interpolator = sitk.sitkNearestNeighbor)
        
        # Convert axes to (width, height, slices) where slice = 0 is at the bottom of patient
        self.scan = np.swapaxes(sitk.GetArrayFromImage(resampled_scan), 0, 2)#[:,:,::-1]
        self.pred_segm_index = np.swapaxes(sitk.GetArrayFromImage(resampled_segm), 0, 2)#[:,:,::-1]
        self.truth_segm = np.swapaxes(sitk.GetArrayFromImage(resampled_truth_segm), 0, 2)#[:,:,::-1]

        
        # Save to OriginSpacing folder
        _, origin_header_segm = nrrd.read(self.original_data_folder + self.test_patient + '/segm.nrrd')
        _, origin_header_scan = nrrd.read(self.original_data_folder + self.test_patient + '/scan.nrrd')
        
        for i in range(3):
            origin_header_scan['space directions'][i][i] = self.original_spacing[i]
            origin_header_segm['space directions'][i][i] = self.original_spacing[i]
        
        nrrd.write(self.original_spacing_folder + self.test_patient + '/pred_segm.nrrd', self.pred_segm_index, origin_header_segm)
        nrrd.write(self.original_spacing_folder + self.test_patient + '/truth_segm.nrrd', self.truth_segm, origin_header_segm)
        nrrd.write(self.original_spacing_folder + self.test_patient + '/scan.nrrd', self.scan, origin_header_scan)
        
        self.current_spacing = self.original_spacing
        
    def predict(self, with_transforms = False, verbose=0):
        """
        Predict on the test scan
        Args:
            with_transforms: False infers on the normal patch once, True averages 4 different inferences with transforms
        Notes:
            - This generates a predicted segmentation with the same voxel spacings as test_scan.
        """
        # Read test data
        self.read_test_patient_data_and_pad()
        
        # Transforms for multiple inference DON'T CHANGE ORDER OF THIS, it's important to re-flip the segmentation
        transforms = [get_augmentation(self.patch_size[0])] if not with_transforms else [get_augmentation(self.patch_size[0]),get_augmentation(self.patch_size[0], p_Flip0=1), get_augmentation(self.patch_size[0], p_GaussianNoise=1), get_augmentation(self.patch_size[0], p_RandomGamma=1)]
        
        predicted_patches_onehot = [] # Saves the patches of each scan
        preprocess_input = sm.get_preprocessing(self.backbone)
        
        if verbose: print("Patchifying scan...")
        self.patchify_scan()
        
        if verbose: print("Predicting on patches...")
        
        for i in range(self.scan_patchified_shape[0]):
            for j in range(self.scan_patchified_shape[1]):
                for k in range(self.scan_patchified_shape[2]):
                    ### Remove enhance contrast
                    single_patch = self.scan_patchified[i,j,k, :,:,:]
                    #single_patch_input = preprocess_input(np.expand_dims(single_patch, axis=0))[0] #pre-process according to backbone
                    
                    # Apply transformations
                    single_patch_transformations = np.array([transform(image=single_patch)["image"] for transform in transforms])
                    single_patch_predictions = self.model.predict(single_patch_transformations) # For rgb

                    # Re-flip the prediction on the flipped patch
                    if with_transforms:
                        single_patch_predictions[1] = transforms[1](mask=single_patch_predictions[1])["mask"]

                    # Average the probabilities and append
                    predicted_patches_onehot.append(np.mean(single_patch_predictions, axis=0))

        #Convert list to numpy array
        predicted_patches_onehot = np.array(predicted_patches_onehot)

        # Reshape so we can use unpatchify
        predicted_patches_onehot = np.reshape(predicted_patches_onehot, self.scan_patchified_shape + (self.n_classes,) )

        # Reconstruct from patches
        if verbose: print("Unpatchifying...")
        self.pred_segm_onehot = restore_from_patches_onehot(self.scan.shape, predicted_patches_onehot, xstep=self.step_size[0],
                                                            ystep=self.step_size[1], zstep=self.step_size[2])

        self.pred_segm_index = np.argmax(self.pred_segm_onehot, axis=3)
        
        # Unpad to original size
        self.unpad_scan_and_truth_segm()

        if verbose: print("Done.")
        
        # Free memory
        self.scan_patchified = self.scan_patchified_shape = None
        
    ## Post processing functions ##
    def apply_gaussian_blur(self, sigma = 2, truncate = 2):
        """
        Apply gaussian blur to one-hot probabilities in segmentation.
        The kernel width is =  2*int(truncate*sigma + 0.5) + 1; for sigma = 2, truncate = 2 it's =9
        """
        
        for n in range(self.n_classes):
            self.pred_segm_onehot[:,:,:,n] = gaussian_filter(self.pred_segm_onehot[:,:,:,n]
                                                                    ,sigma=sigma, truncate=truncate)

        self.pred_segm_index = np.argmax(self.pred_segm_onehot, axis=3)
        
    def clean_connected_components(self):
        """
        Clean connected components of segmentation. Apply this after the gaussian blur.
        """
        
        # Clean 'dust', we define that as CC with less than 100 voxels
        self.pred_segm_index = cc3d.dust(self.pred_segm_index.astype(np.uint16), threshold=100, in_place=False)

        # Connected components of the tumor, keep only largest CC
        n_class = 1
        tumors_filtered = np.copy(self.pred_segm_index)
        tumors_filtered[tumors_filtered != n_class] = 0
        tumor_cc = cc3d.connected_components(tumors_filtered)
        cc_sizes = np.unique(tumor_cc, return_counts=True)[1]
        cc_sizes = cc_sizes[1:] # Remove background
        if len(cc_sizes) > 0: # Case where some tumor is detected
            largest_index = np.argmax(cc_sizes)
            tumors_filtered[tumor_cc != largest_index+1] = 0

        # Connected components of the kidney, keep only 2 largest CC's
        n_class = 4
        kidneys_filtered = np.copy(self.pred_segm_index)
        kidneys_filtered[kidneys_filtered != n_class] = 0
        kidneys_cc = cc3d.connected_components(kidneys_filtered)

        cc_sizes = np.unique(kidneys_cc, return_counts=True)[1]
        cc_sizes = cc_sizes[1:] # Remove background
        largest_index_1 = np.argmax(cc_sizes); cc_sizes[largest_index_1]=0; largest_index_2 = np.argmax(cc_sizes)
        kidneys_filtered[(kidneys_cc != largest_index_1+1) & (kidneys_cc != largest_index_2+1)] = 0

        # Don't do anything to arteries and veins
        vessels_filtered = np.copy(self.pred_segm_index)
        vessels_filtered[(vessels_filtered != 2) & (vessels_filtered != 3)] = 0

        # Final segmentation
        self.pred_segm_index = tumors_filtered + kidneys_filtered + vessels_filtered

    ## Functions to display 3D data ##
    def show_scan_vs_pred(self):
        # Select 1 rgb channel for the scan
        ImageSliceViewer3D(self.scan, self.pred_segm_index, title_left="Scan", title_right="Prediction")
    
    def show_scan_vs_truth(self):
        # Select 1 rgb channel for the scan
        ImageSliceViewer3D(self.scan, self.truth_segm, title_left="Scan", title_right="Ground Truth")
    
    def show_truth_vs_pred(self):
        # Select 1 rgb channel for the scan
        ImageSliceViewer3D(self.truth_segm, self.pred_segm_index, title_left="Ground Truth", title_right="Prediction")
        
    ## Other functions ##
    def compute_dice(self):
        """
        Computes dice between pred_segm_index and truth_index.
        Outputs the dice values in class order (background is class 0)
        """
        return dice_coef_multiclass(self.truth_segm, self.pred_segm_index, self.n_classes, one_hot=False)
