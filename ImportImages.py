# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 00:41:01 2019

@author: John
"""

import pims
from PIL import Image
import numpy as np
#from skimage.filters import unsharp_mask
from skimage import morphology
from skimage import restoration
#from skimage.filters import gaussian_filter
from skimage import filters, feature
import matplotlib  as mpl 
import matplotlib.pyplot as plt
from scipy import ndimage
import os
#from matplotlib_scalebar.scalebar import ScaleBar


mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')


def ImportImg(directory, prefix, preprocess_func):
    frames = pims.ImageSequence(os.path.join(directory, prefix + '*.tif'), process_func=preprocess_func)
    return frames




def stack(directory, prefix):
    frames = pims.TiffStack(os.path.join(directory, prefix))
    
    #plt.imshow(frames[0])
    #plt.show()
    return frames

def crop(img):
    """
    Crop the image to select the region of interest
    """   
    x_min = 474
    x_max = 1024
    y_min = 208
    y_max = 328
    return img[y_min:y_max,x_min:x_max]

def crop_bottom(img):    
    x_min = 240
    x_max = 673
    y_min = 250
    y_max = 540
    return img[y_min:y_max,x_min:x_max]
    

def crop_drop(img):
    """
    Crop a single droplet
    """   
    x_min = 292
    x_max = 1256
    y_min = 164
    y_max = 326
    return img[y_min:y_max,x_min:x_max]

def preprocess_sharpen(img):
    #img = crop(img)
    #plt.imshow(img)
    #plt.show()
    img = img.astype('float64')
    #ref= (img-img.min())/img.max()
    #edges1 = feature.canny(ref)
    #plt.imshow(edges1)
    #plt.show()
    #edgeloc = np.where(edges1 == True)
    
    
    
    #img = (img-img.min())/img.max()
    #print(min(edgeloc[0]), max(edgeloc[0]), min(edgeloc[1]), max(edgeloc[1]))
    #cropped = img[min(edgeloc[0])-10:max(edgeloc[0])+10, min(edgeloc[1]):max(edgeloc[1])]
    #img = filters.unsharp_mask(img, radius = 3, amount = 3)
    
    #plt.imshow(img)
    #plt.show()
    #img = img*(255/img.max())
    
    
    return img


def preprocess_bin(img):
    """
    Apply image processing functions to return a binary image
    """    
    img = crop(img)
    
    mask = filters.gaussian(img)
    mask = mask < 0.875
#    
    img = filters.unsharp_mask(img, radius = 3, amount = 5)
    img *= 255.0/img.max()
    
    img = filters.sobel(img)
    thresh = filters.threshold_local(img, 9, method = 'gaussian')
    img = img > thresh
    plt.imshow(img)
    plt.show()
    out = mask.copy()
    out[mask == True] = img[mask == True]
    out = morphology.binary_closing(out, morphology.disk(1)).astype(np.uint8)
    out = np.invert(out)
    
    return out

def preprocess_grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

def plotimg(img):
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    scalebar = ScaleBar(2.76, units = 'um', location = 'lower right')
    ax.add_artist(scalebar)
    ax.imshow(img)
    return

def default(img):
    return img

