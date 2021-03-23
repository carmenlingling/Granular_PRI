#Written for python 3.6
import matplotlib
matplotlib.use("TkAgg")
#matplotlib.rc('text', usetex = True) #this makes matplotlib use the latex renderer
import matplotlib.pyplot as plt
import numpy as np


import scipy as sc
from skimage import feature, measure
from skimage.filters import roberts, sobel, scharr, prewitt
import tkinter as tk
from tkinter import filedialog
import os 
matplotlib.rc('text', usetex = True) #this makes matplotlib use the latex renderer
import matplotlib.pyplot as plt
import numpy as np

#standard path selection code

pixtomic = 235/ 6.7 #len fishingline pix microscope * um/pixel/len camera pix

#######################################
#List all of the image files
directoryfeb19 = '/Volumes/My Passport/PRI bubbles clusters/glassy'
fileNamesfeb19 = []
for files in [f for f in os.listdir(directoryfeb19) if f.endswith('.csv')]:
    fileNamesfeb19.append(files)
fileNamesfeb19.sort()

directoryfeb19c1 = '/Volumes/My Passport/PRI bubbles clusters/19022021/1'
fileNamesfeb19c1 = []
for files in [f for f in os.listdir(directoryfeb19c1) if f.endswith('.csv')]:
    fileNamesfeb19c1.append(files)
fileNamesfeb19c1.sort()

#######################################
#List all of the image files
directorymar2r1 = '/Volumes/My Passport/PRI bubbles clusters/02032021/1'
fileNamesmar2r1 = []
for files in [f for f in os.listdir(directorymar2r1) if f.endswith('.csv')]:
    fileNamesmar2r1.append(files)
fileNamesmar2r1.sort()

directorymar2r2 = '/Volumes/My Passport/PRI bubbles clusters/02032021/2'
fileNamesmar2r2 = []
for files in [f for f in os.listdir(directorymar2r2) if f.endswith('.csv')]:
    fileNamesmar2r2.append(files)
fileNamesmar2r2.sort()
def rowCount(x):
    
    return x[-5]
def ncount(filename):
    st = filename.partition('N')
    sep = st[2].partition('d')
    N =int(sep[0])
    return(N)
fileNamesmar2r1.sort(key=ncount)
fileNamesmar2r1.sort(key=rowCount)
fileNamesmar2r2.sort(key=ncount)
fileNamesmar2r2.sort(key=rowCount)
fileNamesfeb19.sort(key=ncount)
print(fileNamesfeb19)
fileNamesfeb19c1.sort(key=ncount)
print(fileNamesfeb19c1)

figure = plt.figure(1) #Makes a matplotlib figure, the 1 indicates the plot number, plotting more figures at once requires changing that number

figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
axes = figure.add_subplot(111) # This allows for a subplot to be generated. More
#symbols = ['o', '^', '<', 's', 'd', 'p', '*']

offset1mar2 = [64,122,74,88,26,21,64,23,88,  60, 138, 57, 230, 217, 76, 297,185,222]
offset2mar2=[151,245, 142, 299, 155, 153, 200,182, 954,52]
ra = np.r_[np.linspace(0.5,1, len(offset1mar2))]
ra2 = np.r_[np.linspace(0.5,1, len(offset2mar2))]
ra3 = np.r_[np.linspace(0.5, 1, len(fileNamesfeb19))]
c = plt.get_cmap("Purples")
d = plt.get_cmap("Greens")
e = plt.get_cmap("Oranges")
f = plt.get_cmap('plasma')
colors1 = c(ra)
colors2 = d(ra2)
colors3 = e(ra3)
colors4 = f(ra2)

def colorpick(nrow, ind):
    if nrow ==1:
        return colors1[ind]
    elif nrow ==2:
        return colors2[ind]
    elif nrow ==3:
        return colors3[ind]
    elif nrow == 4:
        return colors3[ind]
offset1 = [566,139, 175,264,182, 246,351,148,154,215,200, 267,309 ]
offset1 =[139, 175,264,182, 246,351,148,154,215,200, 267,309,566 ]
offset2 = [450, 868, 655, 574, 91]
offsetboth = [139, 175,264,182, 246,351,148,154,215,200, 267,309,450, 868, 655, 574, 91]
offset1mar2 = [26,21,64,64,23,88,122,60,138, 57,230,74,88,217, 76, 231,222]
offset2mar2=[52,151,245, 142, 299, 155, 153, 200,182, 945]
offsetcrumple =[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 754]
offsetmar2both = [64,122,74,88,26,21,64,23,88,  60, 138, 57, 230, 217, 76, 297,222,151,245, 142, 299, 155, 153, 200,182, 945,52]
offsetmar2both = [26,21,64,64,23,88,122,60,138, 57,230,74,88,217, 76, 297,222,52,151,245, 142, 299, 155, 153, 200,182, 945]
offsetg = [96, 138,72,185]
def plot_colorbar(cmap, axes):
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    plt.colorbar(sm, cax = axes, orientation = 'horizontal', ticks = [0, 0.5, 1])
    
    return
'''cbaxes = axes.inset_axes([0.02, 0.77, 0.5, 0.05])
plot_colorbar(c, cbaxes)

daxes = axes.inset_axes([0.02, 0.55, 0.5, 0.05])
plot_colorbar(d, daxes)
eaxes = axes.inset_axes([0.02, 0.93, 0.5, 0.05])
plot_colorbar(e, eaxes)'''

#plt.text(0.23, 0.85, r'$\eta = 2000 \textrm{ cSt}$', fontsize = 16,transform=axes.transAxes)
#plt.text(0.23, 0.93, r'$\eta = 1000 \textrm{ cSt}$', fontsize = 16,transform=axes.transAxes)
#plt.text(0.23, 0.77, r'$\eta = 5000 \textrm{ cSt}$', fontsize = 16,transform=axes.transAxes)
gpixtomic= [235/ 6.7, 235/ 6.7,235/ 5.0, 235/ 5.0]
def plotthings(directory,fileNames, offsets, glassy = False):
    N = []
    row = []
    rg = []
    
    #print(len(fileNames), len(offsets))
    for m in range(len(fileNames)):
#for m in range(1):
        if glassy ==False:
            perimeter,normperim,  times = np.genfromtxt(directory+'/'+fileNames[m])
            perimeter=perimeter*pixtomic
        else:
            times, perimeter = np.genfromtxt(directory + '/'+fileNames[m])
        st = fileNames[m].partition('N')
        sep = st[2].partition('d')
        N.append(int(sep[0]))
        last = sep[2]
        #print(last)
        if glassy ==False:
            r = int(last.partition('row')[0])*pixtomic/2
        else:
            r = int(last.partition('row')[0])*gpixtomic[m]/2
        
        rwo = int((last.split('.'))[0].partition('row')[2])
        row.append(int((last.split('.'))[0].partition('row')[2]))
        rg.append(perimeter[-1]/r)
    #axes.plot((times*int(fileNames[m][0:3])/1000)-offsets[m], perimeter,'.', color = colorpick(int(fileNames[m][-11]), m), label = fileNames[m][-11])
        '''ind = np.where(times*300/1000 > offsets[m])
        locstart = np.where(perimeter/(np.sqrt(N[m])*d/2) < 75)

        if len(locstart[0]) == 0:
            pass
        elif locstart[0][0] > ind[0][0]:
            pass
        else:
            fit = np.polyfit(times[locstart[0][0]:ind[0][0]]*300/1000-offsets[m], perimeter[locstart[0][0]:ind[0][0]]/(np.sqrt(N[m])*d/2), 1)'''
            
        axes.plot((((times-times[0])*int(300)/1000)-offsets[m]), (perimeter-perimeter[-1])/(np.sqrt(N[m])*(r)),'.', color = colorpick(rwo, m), label = 'N = %i'%N[m])
            #axes.plot(((times[locstart[0][0]:ind[0][0]]*int(300)/1000)-offsets[m]), np.polyval(fit, (times[locstart[0][0]:ind[0][0]]*int(300)/1000)-offsets[m]))
    

    axes.set_xlabel(r'$t [\textrm{s}]$', fontsize = 24) #x and y axis labels
    axes.set_ylabel(r'$\frac {(R_\textrm{g}-R_\textrm{f})}{r\sqrt{N}}$', fontsize = 24)
    #axes.set_xlim([0,400])
    axes.tick_params(labelsize = 20)
    #axes.legend(fontsize = 16, frameon = False)
    return N, row, rg

Ng, rowg, rg = plotthings(directoryfeb19,fileNamesfeb19, offsetg, glassy = True)

Nmar1, rowmar1, rgmar1 = plotthings(directorymar2r1, fileNamesmar2r1, offset1mar2)
#Nfeb1, rowfeb1, rgfeb1 = plotthings(directoryfeb19c1, fileNamesfeb19c1, offset1mar2)
Nmar2, rowmar2, rgmar2 = plotthings(directorymar2r2, fileNamesmar2r2, offsetcrumple)
#ompile

#print(Nmar1+Nmar2)
plt.show()
'''
figure = plt.figure(2)
figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
axes = figure.add_subplot(111) 
'''

import rgforgran

figure = plt.figure(3)
figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
axes = figure.add_subplot(111) 
def linfit(x, a):
    return a*x
def plotrgn(fileNames, directory, N, row, rg, glassy = False):
    
    for m in range(len(rg)):
        
#for m in range(1):
        if glassy ==False:
            perimeter,normperim,  times = np.genfromtxt(directory+'/'+fileNames[m])
            perimeter = perimeter*pixtomic
        else:
            times, perimeter = np.genfromtxt(directory + '/'+fileNames[m])
        st = fileNames[m].partition('N')
        sep = st[2].partition('d')
        
        last = sep[2]
        #print(last)
        if glassy ==False:
            r = int(last.partition('row')[0])*pixtomic/2
        else:
            r = int(last.partition('row')[0])*gpixtomic[m]/2
    #axes.plot((times*int(fileNames[m][0:3])/1000)-offsets[m], perimeter,'.', color = colorpick(int(fileNames[m][-11]), m), label = fileNames[m][-11])

        axes.plot(np.sqrt(N[m]), (rg[m]),'.', color = colorpick(row[m], m), label = 'N = %i'%N[m], markersize = 20)
        
    


    axes.set_xlabel(r'$\sqrt{N}$', fontsize = 24) #x and y axis labels
    axes.set_ylabel(r'$(R_\textrm{g}/r)$', fontsize = 24)
    axes.tick_params(labelsize = 20)
    #axes.legend(loc = 3, fontsize = 16, frameon = False)
    axes.set_ylim(0,26.7)
fitg = sc.optimize.curve_fit(linfit,np.sqrt(np.asarray(Ng)), np.asarray(rg))
print(fitg)
#axes.plot(np.linspace(0, 20, 2),linfit((np.linspace(0, 20, 2)), fitg[0]), 'r-')
fitc = sc.optimize.curve_fit(linfit,np.sqrt(np.asarray(Nmar1+Nmar2)), np.asarray(rgmar1+rgmar2))
print(fitc)
axes.plot(np.linspace(0, 20, 2),linfit((np.linspace(0, 20, 2)), fitc[0]), 'b-')
axes.plot(np.sqrt(np.linspace(0, 400, 400)), (np.asarray(rgforgran.rgdata(rgforgran.placement_algo(400, (10/2)*235/6.7)))/(5*235/6.7)), 'k', label = r'$HCP')
#axes.plot(np.linspace(0, 400+1, 400)**0.5, rgforgran.rgdata(rgforgran.placement_algo(400, (10/2)*235/6.7)), 'k', label = r'$HCP')
#axes.plot(np.linspace(0, 400+1, 400)**0.5, rgforgran.rgcont(np.linspace(0, 400+1, 400), (10/2)*235/6.7))
#plotrgn(fileNamesfeb19, directoryfeb19, Ng, rowg, rg, glassy= True)
plotrgn(fileNamesmar2r1, directorymar2r1, Nmar1, rowmar1, rgmar1)
#plotrgn(fileNamesfeb19c1,directoryfeb19c1, Nfeb1, rowfeb1, rgfeb1)
plotrgn(fileNamesmar2r2, directorymar2r2, Nmar2, rowmar2, rgmar2)
plt.show()


figure = plt.figure(4)
figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
axes = figure.add_subplot(111)
slopes = []
slopesg = []
def linearfit(x, a, b):
    return(a*x+b)
def plotsloperow(fileNames, directory, N, row, offset, glassy = False):
    
    for m in range(len(fileNames)):
#for m in range(1):
        if glassy ==False:
            perimeter,normperim,  times = np.genfromtxt(directory+'/'+fileNames[m])
            perimeter = perimeter*pixtomic
        else:
            times, perimeter = np.genfromtxt(directory + '/'+fileNames[m])
        st = fileNames[m].partition('N')
        sep = st[2].partition('d')
        
        last = sep[2]
        if glassy ==False:
            r = int(last.partition('row')[0])*pixtomic/2
        else:
            r = int(last.partition('row')[0])*gpixtomic[m]/2
        ind = np.where(times*300/1000 >= offset[m])
        locstart = np.where((perimeter-perimeter[-1])/(np.sqrt(N[m])*r) < 1.95)
        
        if len(locstart[0]) == 0:
            pass
        elif locstart[0][0] > ind[0][0]:
            pass
        else:
            print(times[locstart[0][0]]*0.3, times[ind[0][0]]*0.3)
            fit = sc.optimize.curve_fit(linearfit,perimeter[locstart[0][0]:ind[0][0]]/(np.sqrt(N[m])*(r)),times[locstart[0][0]:ind[0][0]]*300/1000)
            #print(fit[1])
            if glassy == False:
                slopes.append(abs(fit[0][0]))
            else:
                slopesg.append(abs(fit[0][0]))
            aspect = (N[m]/row[m])/row[m]
            axes.errorbar(N[m], (abs(fit[0][0])), yerr = np.sqrt(fit[1][0,0]),color = colorpick(row[m], m), fmt = 'o')#,yerr = (fit[1][0]/abs(fit[0][0])**2)
            #axes.loglog(N[m], abs(fit[0]), '*', color = colorpick(row[m], m))
            #axes.plot(N[m], (perimeter[0]-perimeter[ind[0][0]])/(np.sqrt(N[m])*5)/((times[0]*int(300)/1000)-offset[m]),'.', color = colorpick(row[m], m), label = 'N = %i'%N[m])
            
    
    axes.set_xlabel(r'$N$', fontsize = 18) #x and y axis labels
    axes.set_ylabel(r'$\tau [\textrm{s}]$', fontsize = 24)
    axes.tick_params(labelsize = 20)
axes.set_ylim([0,134])
axes.set_xlim([0, 400])

#axes.legend(loc = 3, fontsize = 16, frameon = False)
plotsloperow(fileNamesfeb19, directoryfeb19, Ng, rowg, offsetg, glassy= True)
plotsloperow(fileNamesmar2r1, directorymar2r1, Nmar1, rowmar1, offset1mar2)
plotsloperow(fileNamesmar2r2, directorymar2r2, Nmar2, rowmar2, offset2mar2)
#plotsloperow(fileNamesfeb19c1,directoryfeb19c1, Nfeb1, rowfeb1, offset1)

#print(len(Nmar1+Nmar2+Nfeb1), len(slopes))
def oneover(x, a, b):
    return a/x+b
fitall = sc.optimize.curve_fit(linfit,np.asarray(Nmar1+Nmar2), np.asarray(slopes) )
fitg = sc.optimize.curve_fit(linfit,np.asarray(Ng), np.asarray(slopesg))
print(fitall, fitg)



axes.plot(np.linspace(0, 400, 400), (63/290)*np.linspace(0, 400, 400), 'k')
axes.fill_between(np.linspace(0, 400, 400),  (63/290)*np.linspace(0, 400, 400),134,'b', alpha =0.2)
axes.fill_between(np.linspace(0, 400, 400),  0, (63/290)*np.linspace(0, 400, 400), 'r', alpha =0.2)
#axes.plot(np.linspace(20, 400, 400), fitg[0][0]*np.linspace(20, 400, 400),'r')
plt.show()


###offset test
'''
offset1mar2 = [212,122,76,88,26,63,64,23, 216, 64, 178, 57, 234, 217, 84, 297,185,222,151]
offsetg = [116,29,62, 270]

fileNames= fileNamesfeb19
directory = directoryfeb19
for m in range(len(fileNames)):
    figure = plt.figure(1) #Makes a matplotlib figure, the 1 indicates the plot number, plotting more figures at once requires changing that number

    figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
    axes = figure.add_subplot(111)
#for m in range(1):
    perimeter,  times = np.genfromtxt(directory+'/'+fileNames[m])
    st = fileNames[m].partition('N')
    sep = st[2].partition('d')
    N =int(sep[0])
    last = sep[2]
    #print(last)
    d = int(last.partition('row')[0])
    rwo = int((last.split('.'))[0].partition('row')[2])

    #axes.plot((times*int(fileNames[m][0:3])/1000)-offsets[m], perimeter,'.', color = colorpick(int(fileNames[m][-11]), m), label = fileNames[m][-11])
    axes.plot(((times*int(300)/1000)), (perimeter)/np.sqrt(N)*(d/2),'.', label = 'N = %i'%N)
    plt.show()

offset2mar2=[151,245, 142, 299, 155, 153, 200, 954, 182, 52]
'''
'''def plotrgnn(directory, fileNames, N, row, glassy = False):
    for m in range(len(fileNames)):
#for m in range(1):
        if glassy ==False:
            perimeter,normperim,  times = np.genfromtxt(directory+'/'+fileNames[m])
        else:
            times, perimeter = np.genfromtxt(directory + '/'+fileNames[m])
        st = fileNames[m].partition('N')
        sep = st[2].partition('d')
        N.append(int(sep[0]))
        last = sep[2]
        print(last)
        d = int(last.partition('row')[0])
    #axes.plot((times*int(fileNames[m][0:3])/1000)-offsets[m], perimeter,'.', color = colorpick(int(fileNames[m][-11]), m), label = fileNames[m][-11])

        axes.plot(N[m], (perimeter[-1]/N[m]**0.5),'.', color = colorpick(row[m], m), label = 'N = %i'%N[m])
    axes.plot(N, np.asarray(N)**0.5*(10/2)*(12**0.25)/(2*np.pi**0.5*np.asarray(N)**0.5), 'k', label = r'$R_g = 12^(1/4)\sqrt(N/\pi)r/2$')

    axes.set_xlabel(r'$N$', fontsize = 18) #x and y axis labels
    axes.set_ylabel(r'$R_g/N^{1/2}$', fontsize = 18)
    axes.tick_params(labelsize = 20)
    axes.legend(fontsize = 16, frameon = False)
#plotrgnn(directoryfeb19, fileNamesfeb19, Nfeb, rowfeb)
#plotrgnn(directorymar2, fileNamesmar2, Nmar, rowmar)
plt.show()'''
