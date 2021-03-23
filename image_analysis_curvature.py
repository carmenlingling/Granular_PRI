#Written for python 3.6
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rc('text', usetex = True) #this makes matplotlib use the latex renderer
import matplotlib.pyplot as plt
import numpy as np
import ImportImages as II

import scipy as sc
from skimage import feature, measure, filters
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.segmentation import active_contour
import tkinter as tk
from tkinter import filedialog
import os 
import time
start_time = time.time()

########################################
pixtomic = 235/6.7
#standard path selection code
'''root = tk.Tk()
root.withdraw()
root.update()
file_path = filedialog.askopenfilename() #asks which file you want to analyze and records the filepath and name
root.destroy()'''
'''
start = 23
xmin = 474
xmax = 1024
ymin = 208
ymax = 328
#######################################
#List all of the image files
directory = os.path.split(file_path)[0]
prefix = os.path.split(file_path)[1]
frames = II.stack(directory, prefix, II.preprocess_sharpen, start)'''
#####################################
#A function that creates a mask of the edges of the pipette
def fill_pipette(edges, threshold):
    filled = np.empty((edges.shape[0],edges.shape[1]))
    for k in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[k,j]< threshold:
                filled[k,j] = 0
            else:
                filled[k,j] = 1
    return filled
def fit_spline(xdata, ydata, averaging_interval):
    newx = []
    newy = []
    for z in range(len(xdata)-averaging_interval):
        fit = np.polyfit(xdata[z:z+averaging_interval], ydata[z:z+averaging_interval], 1)
        newx.append(xdata[z+int(averaging_interval/2)])
        newy.append(np.polyval(fit, xdata[z+int(averaging_interval/2)]))
    return newx, newy


def clean_data(data):
    for k in range(len(data)-1):
        if abs(data[k,0]-data[k+1,0])>2 or abs(data[k,1]-data[k+1,1])>2:
            return(k+1)
        
def curvature(xdata, ydata):
    difx = np.diff(xdata)
    dify = np.diff(ydata)
    dtime, difx = fit_spline(range(len(difx)), difx, 10)
    dtime, dify = fit_spline(range(len(dify)), dify, 10)
    #plt.plot(range(len(difx)), difx)
    #plt.plot(range(len(dify)), dify)
    ddify = np.diff(ydata, 2)
    dtime, ddify = fit_spline(range(len(ddify)), ddify, 9)
    ddifx = np.diff(xdata, 2)
    dtime, ddifx = fit_spline(range(len(ddifx)), ddifx, 9)
    #plt.plot(range(len(dify)),abs(np.asarray(difx)*np.asarray(ddify) - np.asarray(dify)*np.asarray(ddifx))/((np.asarray(difx)**2+np.asarray(dify)**2)**(3/2)))
    #plt.show()
    return sum(abs(np.asarray(difx)*np.asarray(ddify) - np.asarray(dify)*np.asarray(ddifx))/((np.asarray(difx)**2+np.asarray(dify)**2)**(3/2)))
import brute_dfs_it

def R_g(xs, ys):
    xavg = np.mean(xs)
    yavg = np.mean(ys)
    R = -(xavg**2+yavg**2)
    for item in range(len(xs)):
        R += (xs[item]**2+ys[item]**2)/len(xs)
    return np.sqrt(R)

#########################
def image_analysis_curvature(imagepath, start, ymin, ymax, xmin, xmax, plot=False):
    perimeters = []
    normperims = []
    times = []
    curvatures = []
    rg = []
    directory = os.path.split(imagepath)[0]
    prefix = os.path.split(imagepath)[1]
    frames = II.stack(directory, prefix, II.preprocess_sharpen)
    for k in range(len(frames)-start):
    #for k in range(200):
        ref = frames[k+start]
        #print(ref.shape)
        ref = ref[ymin:ymax, xmin:xmax]
        ref = (ref-ref.min())/ref.max()
        #edge_sobel = sobel(ref)
        val = filters.threshold_otsu(ref)
        mask = ref < val
        #edges1 = feature.canny(ref, sigma=2.7)edges1 = feature.canny(ref, sigma=2.7)
        from scipy import ndimage
        full = ndimage.morphology.binary_dilation(ndimage.binary_fill_holes(mask)).astype(int)
        #plt.imshow(full)
        edges = full - ndimage.morphology.binary_dilation(full)
        #print(full)
        #plt.imshow(edges, cmap = 'viridis')
        #fulledges = feature.canny(full)
        edgesloc = np.where(edges < 0)
        #print(edges)
        #plt.scatter(edgesloc[0],edgesloc[1], cmap = 'gray')
    
        #ref_img = plt.imshow(ref, cmap = 'viridis', alpha = 0.5)

    
        #plt.savefig(directory+'/filter/%i.png'%k)

        #plt.show()
    
        edges = []
        for i in range(len(edgesloc[0])):
            edges.append([edgesloc[0][i], edgesloc[1][i]])
        ordered_edges = brute_dfs_it.get_brute_list(edges)
    
    
        ordered_edges = np.asarray(ordered_edges)
        stop = len(ordered_edges)
        stop = clean_data(ordered_edges)
        print(stop)
        if k % 1 == 0:
            
            ra = np.r_[np.linspace(0, 1, stop)]
            c = plt.get_cmap("viridis")
            colors1 = c(ra)
            plt.scatter(ordered_edges[0:stop,1], ordered_edges[0:stop,0], c=ra, cmap = 'viridis')
            plt.imshow(edges, cmap = 'gray')
            plt.show()
        rg.append(R_g(edgesloc[0], edgesloc[1])*pixtomic)
        #splinedx, splinedy= fit_spline(ordered_edges[0:stop,0],ordered_edges[0:stop,1])
        #curve = curvature(ordered_edges[0:stop,0],ordered_edges[0:stop,1])
        #curvatures.append(curvature(ordered_edges[0:stop,0],ordered_edges[0:stop,1]))
   
        times.append(k)
    
    if plot==True:
        plt.imshow(edges1, cmap = 'gray')
        ref_img = plt.imshow(ref, cmap = 'viridis', alpha = 0.5)
        
        plt.show()
        ra = np.r_[np.linspace(0, 1, stop)]
        c = plt.get_cmap("viridis")
        colors1 = c(ra)
        plt.scatter(ordered_edges[0:stop,1], ordered_edges[0:stop,0], c=ra, cmap = 'viridis')
        plt.show()

    print("My program took", (time.time() - start_time)/60, "min to run")
    data = np.asarray([times, rg])
    np.savetxt(directory + '/rg.csv', data)
    plt.plot(times, rg)
    plt.show()


imagelist = ['/Volumes/My Passport/PRI bubbles clusters/19022021/300ms_fishingline_5%peg_1_5%sds_H20_air_p2_1_g_1/300ms_fishingline_5%peg_1_5%sds_H20_air_p2_1_g_1_MMStack.ome.tif','/Volumes/My Passport/PRI bubbles clusters/19022021/300ms_fishingline_5%peg_1_5%sds_H20_air_p2_2_g_1/300ms_fishingline_5%peg_1_5%sds_H20_air_p2_2_g_1_MMStack.ome.tif']
startlist = [83,87]
xminlist = [326, 110]
xmaxlist = [1076,1078]
yminlist = [418,374]
ymaxlist = [640, 580]


for j in range(len(imagelist)):
    
    image_analysis_curvature(imagelist[j],
                                 startlist[j], yminlist[j],
                                 ymaxlist[j],
                                 xminlist[j],
                                 xmaxlist[j], plot=False)
