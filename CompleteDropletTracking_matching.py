# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 00:37:43 2019

@author: John
"""

'''
Load in images. Preform preprocessing
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from skimage.feature import canny
from skimage.filters import unsharp_mask
from skimage import filters, measure

import pandas as pd
import trackpy as tp

import time
start_time = time.time()


mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')



directory = '/Volumes/My Passport/PRI bubbles clusters/02032021/300ms_fishingline_5%peg_1_5%sds_H20_air_p1_2_4/'

prefix = '300ms_fishingline_5%peg_1_5%sds_H20_air_p1_2_4_MMStack.ome.tif'
datafile = 'stack'
start = 18
import ImportImages as II
if datafile == 'sequence':
    frames = II.ImportImg(directory, prefix, II.preprocess_sharpen)
elif datafile == 'stack':
    frames = II.stack(directory, prefix, II.preprocess_sharpen)
# frames = II.ImportImg(directory, prefix, II.preprocess_sharpen)
#rawframes = II.ImportImg(directory, prefix, II.crop)
#counter = II.ImportImg(directory, prefix, II.crop_drop)
#counter = counter[0:100]
#
#
#from DropletCounter import Counter
#
#droplets, droplet_template, droplet_counter, radius = Counter(counter)

#droplet_template = filters.unsharp_mask(droplet_template, radius = 2, amount = 5)
#pd.to_pickle(droplets,directory+'_droplet_data')

import MatchTemplate
#from MatchTemplate import GaussianMatch
#from MatchTemplate import PlotTemplate

## Use if counter does not work
droplet_template = frames[start][74:84,262:272]
threshold = 0.65
diameter =10
N =39
row =1
img_example = frames[-1]
plt.imshow(img_example)
plt.show()


'''
Test matching on example image
'''
def R_g(xs, ys):
    xavg = np.mean(xs)
    yavg = np.mean(ys)
    R = -(xavg**2+yavg**2)
    for item in range(len(xs)):
        R += (xs[item]**2+ys[item]**2)/len(xs)
    return np.sqrt(R)



match, peaks = MatchTemplate.MatchTemplate(img_example, droplet_template, threshold)
MatchTemplate.PlotTemplate(img_example, droplet_template, match, peaks)
print(len(peaks))




def labelimg(img):
    global features
    features = pd.DataFrame()
    
    match, peaks = MatchTemplate.MatchTemplate(img, droplet_template, threshold)
    print(len(peaks))
    Rtemp = R_g(peaks[:,0], peaks[:,1])
    return Rtemp, peaks


Rg = []
times = []
#for m in range(start,len(frames)-start):
for m in range(start,300+start):
    r, pek = labelimg(frames[m])
    if m == len(frames)-start-1:
        MatchTemplate.PlotTemplate(frames[m], droplet_template, match, pek)
    print(r)
    Rg.append(r)
    times.append(m)
    
fig ,[ax, axlog] = plt.subplots(2)  
ax.plot(range(len(Rg)), Rg, 'k.')
ax.set(xlabel='frame', ylabel = r'$R_g$')
#figlog, axlog = plt.subplots(2)
axlog.loglog(range(len(Rg)), Rg, 'k.')
plt.show()

data = np.asarray([Rg,Rg/(2*np.pi*(diameter/2)*np.sqrt(N)), times])
np.savetxt(directory+ '/Rg'+os.path.split(directory[0:3])[1]+'N%id%irow%i.csv'%(N,diameter,row) , data)


    

'''
exit()
#    
##Check to make sure there are no missing or additional droplets

plt.figure()   
plt.plot(result.set_index(['frame']).groupby(['frame']).x.count())
plt.show()


pd.to_pickle(result,directory+prefix+'_position_data')
'''

#%%

##Calculate Trajectories##

# pred = tp.predict.NearestVelocityPredict()
# t = pred.link_df(result, search_range = 200, adaptive_stop=5, adaptive_step=0.99, memory=2)
# #t = tp.link_df(data[data.frame<=100], search_range = 100, adaptive_stop=5, adaptive_step=0.99, memory=1)

# def calcTraj (t, item):
#     global data
#     data = pd.DataFrame()
#     sub = t[t.particle==item]
#     dvx = np.diff(sub.x)
#     dvy = np.diff(sub.y)
#     for x, y, dx, dy, frame in zip(sub.x[:-1], sub.y[:-1], dvx, dvy, sub.frame[:-1],):
#         data = data.append([{'dx': dx, 
#                              'dy': dy, 
#                              'x': x,
#                              'y': y,
#                              'frame': frame,
#                              'particle': item,
#                             }])
#     return data

# from joblib import Parallel, delayed
# Data = Parallel(n_jobs = 12)(delayed(calcTraj)(t, item) for item in set(t.particle))


# trajectories = pd.concat(Data)



#%%

#from TrackingThreshold import trackthresh
#test = trajectories
#trajectories = trackthresh(trajectories)

# trajectories = tp.filter_stubs(trajectories, threshold=3)

# '''
# Uncomment when ready
# '''
# trajectories.index.name = 'index'
# pd.to_pickle(trajectories,directory+'_all_data')
#plt.plot(trajectories.groupby(['frame']).particle.unique())
#diff = []
#for particle in trajectories.particle.unique():
#    plt.plot(trajectories[trajectories.particle==particle].y)
#for i in range(len(trajectories[trajectories.particle==3].y)):
#    diff.append(np.abs(trajectories[trajectories.particle==3].y.iloc[i]-trajectories[trajectories.particle==3].y.iloc[i-1]))
#%%
#from TrajectoryPlot import plotTraj
#from TrajectoryPlot import createFolder

#def createFolder(directory):
#    if not os.path.exists(directory+'/TrackingByFrame'):
#        os.mkdir(directory+'/TrackingByFrame')
##
#def plotTraj (traj, k, directory, frames):
#    plt.ion()
#    trajectories_fig = tp.plot_traj(traj[traj.frame<=k], colorby='particle', superimpose=frames[k])
#        
#    trajectories_fig.figure.savefig(directory+'/TrackingByFrame/trajectories' + str(k))
#    return
#
#createFolder(directory)
#
#plt.figure()
##
#from joblib import Parallel, delayed
#Parallel(n_jobs = 12)(delayed(plotTraj)(trajectories, k, directory, frames) for k in np.arange(0, 6000, 1))
##
##
print("My program took", (time.time() - start_time)/60, "min to run")
#

