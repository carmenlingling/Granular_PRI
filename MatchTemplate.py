# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:11:22 2019

@author: John
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import match_template
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit

def MatchTemplate(img, template, thresh):
    match = match_template(img, template, pad_input = True)
    ij = np.unravel_index(np.argmax(match), match.shape)
    x,y  = ij[::-1]
    peaks = peak_local_max(match,min_distance=2,threshold_rel=thresh) # find our peaks
    #peaks = peak_local_max(match,footprint=np.ones((51, 51), dtype=np.bool),threshold_rel=0.5) # find our peaks
    #peaks = corner_peaks(match, min_distance=8,threshold_rel=0.5)
    return match, peaks
    
def PlotTemplate(img, template, match, peaks):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(4,6))
    
    ax1.imshow(template)
    ax1.set_axis_off()
    ax1.set_title('template')
    
    ax2.imshow(img)
    ax2.set_axis_off()
    ax2.set_title('image')
    # highlight matched region
    hcoin, wcoin = template.shape
#    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
#    ax2.add_patch(rect)
    ax2.plot(peaks[:,1], peaks[:,0], 'o', markeredgecolor='r', markerfacecolor='none', markersize=5)
    
    
    ax3.imshow(match)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    #ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
    # ax3.plot(peaks[:,1], peaks[:,0], 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
    plt.show()
    return


def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple 
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def GaussianMatch (match, peaks, template, img, thresh, spread, error):
    fit_params = []
    std = []
    for i in range(len(peaks)):
        #20x20 window
        window_size = 4
        window = match[peaks[i][0]-window_size:peaks[i][0]+window_size,peaks[i][1]-window_size:peaks[i][1]+window_size]
    
        # Create x and y indices and the required meshgrid
        xlen=window.shape[1]
        ylen=window.shape[0]
        x=np.linspace(0, xlen-1,xlen)
        y=np.linspace(0, ylen-1,ylen)
        X = np.meshgrid(x, y)
        
        # Provide an initial guess based on the cropped image, could use max to make this better
        initial_guess = (1,window_size,window_size,2,2,0,.05)
        
        #data_noisy = data + 0.2*np.random.normal(size=data.shape)
        data_noisy = np.ravel(window)
        try:
            popt, pcov = curve_fit(twoD_Gaussian,X, data_noisy, p0=initial_guess,maxfev = 100000)
        except Exception:
            continue
    #    fit_params.append(popt)
    
        err = np.sqrt(np.diag(pcov))
        #Neglect peaks with low amplitude
        if popt[0] < thresh:
            continue
        if popt[3] < spread or popt[4] < spread:
            continue
        if err[3] > error or err[4] > error:
            continue
        
        
        if popt[1] > window_size:
            popt[1] = peaks[i][1] + (window_size - popt[1])
        else:
            popt[1] = peaks[i][1] - (window_size - popt[1])
            
        if popt[2] > window_size:
            popt[2] = peaks[i][0] - (window_size - popt[2])
        else:
            popt[2] = peaks[i][0] + (window_size - popt[2])
    
        fit_params.append(popt)
    
        data_fitted = twoD_Gaussian(X, *popt)
        std.append(np.sqrt(np.diag(pcov)))
        
        
    
    fit_params = np.array(fit_params)
    std = np.array(std)
    return np.array(fit_params), popt

##%%
#match, peaks = MatchTemplate(img_example, droplet_template, 0.4)
#fit_params, popt = GaussianMatch(match, peaks, droplet_template, img_example, 0.5)
#
#fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 3))
#
#ax1.imshow(droplet_template)
#ax1.set_axis_off()
#ax1.set_title('template')
#
#ax2.imshow(img_example)
#ax2.set_axis_off()
#ax2.set_title('image')
## highlight matched region
#hcoin, wcoin = droplet_template.shape
##    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
##    ax2.add_patch(rect)
#ax2.plot(fit_params[:,1], fit_params[:,2], 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
#
#
#ax3.imshow(match)
#ax3.set_axis_off()
#ax3.set_title('`match_template`\nresult')
## highlight matched region
#ax3.autoscale(False)
##ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
#ax3.plot(peaks[:,1], peaks[:,0], 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
#plt.show()
