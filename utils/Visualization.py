# Test 
import numpy as np
import ipywidgets as ipyw
import matplotlib.pyplot as plt

class ImageSliceViewer3D:
    """ 
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks. 
    
    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 

    Arguments:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html
    
    Code adapted from: https://github.com/mohakpatel/ImageSliceViewer3D
    
    """
    
    def __init__(self, scan, segm=None, title=None, title_left=None, title_right=None, figsize=(12,8), cmap='plasma'):
        self.title = title
        self.title_left = title_left
        self.title_right = title_right
        self.scan = scan
        self.segm = segm
        self.figsize = figsize
        self.cmap = cmap
        self.v_scan = [np.min(scan), np.max(scan)]
        if not segm is None: self.v_segm = [np.min(segm), np.max(segm)]
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol_scan = np.transpose(self.scan, orient[view])
        maxZ_scan = self.vol_scan.shape[2] - 1
        
        if not self.segm is None:
            self.vol_segm = np.transpose(self.segm, orient[view])
            maxZ_segm = self.vol_segm.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(min=0, max=maxZ_scan, step=1, continuous_update=False, 
            description='Image Slice:'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        if not self.segm is None:
            self.fig, self.axes = plt.subplots(1,2,figsize=self.figsize)
        else:
            self.fig, self.axes = plt.subplots(1,1,figsize=(self.figsize[0]//2, self.figsize[1]//2))
            self.axes = [self.axes]

        self.axes[0].imshow(self.vol_scan[:,:,z], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v_scan[0], vmax=self.v_scan[1])
        self.axes[0].set_title(self.title_left)
        
        if not self.segm is None:
            self.axes[1].imshow(self.vol_segm[:,:,z], cmap=plt.get_cmap(self.cmap), 
                vmin=self.v_segm[0], vmax=self.v_segm[1])
            self.axes[1].set_title(self.title_right)

        
        self.fig.suptitle(self.title, fontsize=15)
        plt.show()


# The static rendering of Github does not display the image widget, and the 
# ability to interact with the image widget did not work with nbviewer when 
# last (26/05/2018) checked. 
# Use as ImageSliceViewer3D(data)
# Requires ipywidgets==7.7.2 ; widgetsnbextension==3.6.1