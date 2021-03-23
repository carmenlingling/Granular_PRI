#Written for python 3.6
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rc('text', usetex = True) #this makes matplotlib use the latex renderer
import matplotlib.pyplot as plt
import numpy as np
import ImportImages as II

import scipy as sc
from skimage import feature, measure
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.segmentation import active_contour
import tkinter as tk
from tkinter import filedialog
import os 
import time
start_time = time.time()

########################################

#standard path selection code
root = tk.Tk()
root.withdraw()
root.update()
file_path = filedialog.askopenfilename() #asks which file you want to analyze and records the filepath and name
root.destroy()

start = 23
xmin = 474
xmax = 1024
ymin = 208
ymax = 328
#######################################
#List all of the image files
directory = os.path.split(file_path)[0]
prefix = os.path.split(file_path)[1]
frames = II.stack(directory, prefix, II.preprocess_sharpen, start)
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



def clean_data(data):
    for k in range(len(data)-1):
        if abs(data[k,0]-data[k+1,0])>2 or abs(data[k,1]-data[k+1,1])>2:
            return(k+1)
        
def curvature(xdata, ydata):
    difx = np.diff(xdata)
    dify = np.diff(ydata)
    return sum((difx[0:-1]*np.diff(ydata, 2) - dify[0:-1]*np.diff(xdata,2))/((difx[0:-1]**2+dify[0:-1]**2)**(3/2)))
import brute_dfs_it


#########################
perimeters = []
normperims = []
times = []
curvatures = []
for k in range(len(frames)-start):
#for k in range(200):
    ref = frames[k+start]
    #print(ref.shape)
    ref = ref[ymin:ymax, xmin:xmax]
    ref = (ref-ref.min())/ref.max()
    #edge_sobel = sobel(ref)

    edges1 = feature.canny(ref, sigma=3)
    edgesloc = np.where(edges1 == True)
    
    #plt.imshow(edges1, cmap = 'gray')
    
    #ref_img = plt.imshow(ref, cmap = 'viridis', alpha = 0.5)

    
    #plt.savefig(directory+'/filter/%i.png'%k)

    #plt.show()
    #plt.clf()
    #perim = measure.perimeter(edges1)
    #perimeters.append(perim)
    edges = []
    for i in range(len(edgesloc[0])):
        edges.append([edgesloc[0][i], edgesloc[1][i]])
    ordered_edges = brute_dfs_it.get_brute_list(edges)
    
    
    ordered_edges = np.asarray(ordered_edges)
    stop = clean_data(ordered_edges)
    curve = curvature(ordered_edges[0:stop,0],ordered_edges[0:stop,1])
    curvatures.append(curvature(ordered_edges[0:stop,0],ordered_edges[0:stop,1]))
    #print(curve)
    #ra = np.r_[np.linspace(0, 1, stop)]
    #c = plt.get_cmap("viridis")
    #colors1 = c(ra)
    #plt.scatter(ordered_edges[0:stop,1], ordered_edges[0:stop,0], c=ra, cmap = 'viridis')
    #plt.show()
    #print(ordered_edges[:,0])
    #normperims.append(perim/(2*np.pi*(7.42/2)*np.sqrt(276)))
    times.append(k)
    
    #print(perim)
#ref_img = plt.imshow(ref, cmap = 'viridis', alpha = 0.5)
#plt.imshow(edges1, cmap = 'gray')
#plt.show()
#ra = np.r_[np.linspace(0, 1, stop)]
#c = plt.get_cmap("viridis")
#colors1 = c(ra)
#plt.scatter(ordered_edges[0:stop,1], ordered_edges[0:stop,0], c=ra, cmap = 'viridis')
#plt.show()

print("My program took", (time.time() - start_time)/60, "min to run")
data = np.asarray([times, curvatures])
np.savetxt(directory + 'data.csv', data)
plt.plot(times, curvatures)
plt.show()
exit()
#data = np.asarray([perimeters,normperims, times])

print(directory)
'''
perimeters,times = np.genfromtxt('/Volumes/My Passport/PRI bubbles clusters/21012021/50ms_fishingline_bubble_column_05%sds_H20_air_p4_g1_3/images/1data.csv')
perimeters2,times2 = np.genfromtxt('/Volumes/My Passport/PRI bubbles clusters/21012021/50ms_fishingline_bubble_column_05%sds_H20_air_p4_g1_3/images/New Folder With Itemsdata.csv')
'''
figure = plt.figure(2) #Makes a matplotlib figure, the 1 indicates the plot number, plotting more figures at once requires changing that number

figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
axes = figure.add_subplot(111) # This allows for a subplot to be generated. More
axes.plot(times, perimeters,'*', color = 'k', label = r'$1$') #plot horizontal backwards
axes.plot(times, normperims, 'o', color = 'b', label = r'$normed$')

axes.set_xlabel(r'$t / \textrm{s}$', fontsize = 18) #x and y axis labels
axes.set_ylabel(r'$perimeter /\textrm{pix}$', fontsize = 18)
axes.tick_params(labelsize = 20)
axes.legend(loc = 3, fontsize = 16, frameon = False)
plt.savefig(directory+'/p_t.png')
plt.show()
figure = plt.figure(1) #Makes a matplotlib figure, the 1 indicates the plot number, plotting more figures at once requires changing that number

figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
axes = figure.add_subplot(111) # This allows for a subplot to be generated. More
axes.loglog(times, perimeters,'*', color = 'k', label = r'$1$')
#axes.plot(times2, perimeters2, 'o', color = 'b', label = r'$2$')

axes.set_xlabel(r'$t / \textrm{s}$', fontsize = 18) #x and y axis labels
axes.set_ylabel(r'$perimeter /\textrm{pix}$', fontsize = 18)
axes.tick_params(labelsize = 20)
axes.legend(loc = 3, fontsize = 16, frameon = False)
plt.savefig(directory+'/p_t_loglog.png')
plt.show()
