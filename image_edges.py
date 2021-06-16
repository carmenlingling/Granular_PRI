
from scipy import ndimage
from skimage import filters
import matplotlib.pyplot as plt
import os
import numpy as np
import pims
#uses 2 homemade packages, check KDV GitHub!
import depth_first_search
import ImportImages as II



def edgeordering(image, plot=False):
    
    #ref = (image-image.min())/image.max() #normalize
    ref = image
    val = filters.threshold_otsu(ref) #using otsu thresholding to say where an edge should be
    mask = ref < val 
        
        
    full = ndimage.morphology.binary_dilation(ndimage.binary_fill_holes(mask), iterations=2).astype(int) #dialating the edges (in case the otsu method didn't work)
        
    edges = full - ndimage.morphology.binary_dilation(full)
    edgesloc = np.where(edges < 0)
        
    #uncomment to check    
    #plt.scatter(edgesloc[1],edgesloc[0], cmap = 'gray')
    #ref_img = plt.imshow(ref, cmap = 'viridis', alpha = 0.5)
    #plt.show()

    edges = []
    for i in range(len(edgesloc[0])):
        edges.append([edgesloc[0][i], edgesloc[1][i]])
    midpoint = [np.average(edgesloc[0][:]), np.average(edgesloc[1][:])]

    ordered_edges = depth_first_search.get_dfs_list(edges, midpoint)
    ordered_edges = np.asarray(ordered_edges)
    
    if plot==True:
        plt.imshow(edges, cmap = 'gray')
        ra = np.r_[np.linspace(0, 1,len(ordered_edges))]
        c = plt.get_cmap("viridis")
        colors1 = c(ra)
        plt.scatter(ordered_edges[:,1], ordered_edges[:,0], c=ra, cmap = 'viridis')
        plt.show()

    stop = clean_data(ordered_edges) # use this if there are extra points that didn't get picked up
    return ordered_edges

def clean_data(data):
    for k in range(len(data)-1):
        if abs(data[k,0]-data[k+1,0])>=4 or abs(data[k,1]-data[k+1,1])>=4:
            print(data[k], data[k+1])
            return(k+1)
    return(len(data))

import matplotlib
image = matplotlib.image.imread('sample.tif')
edgeordering(image, plot=True)
