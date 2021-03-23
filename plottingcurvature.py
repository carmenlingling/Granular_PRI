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

#length_conversion = 170.76*1.38/ #len fishingline pix microscope * um/pixel
#######################################
#List all of the image files
directoryfeb19 = '/Volumes/My Passport/PRI bubbles clusters/19022021/both'
fileNamesfeb19 = []
for files in [f for f in os.listdir(directoryfeb19) if f.endswith('.csv')]:
    fileNamesfeb19.append(files)
fileNamesfeb19.sort()



#######################################
#List all of the image files
directorymar2 = '/Volumes/My Passport/PRI bubbles clusters/02032021/test'
fileNamesmar2 = []
for files in [f for f in os.listdir(directorymar2) if f.endswith('.csv')]:
    fileNamesmar2.append(files)
fileNamesmar2.sort()
def rowCount(x):
    
    return x[-5]
def ncount(filename):
    st = filename.partition('N')
    sep = st[2].partition('d')
    N =int(sep[0])
    return(N)
#fileNamesmar2.sort(key=ncount)
#fileNamesmar2.sort(key=rowCount)
#fileNamesfeb19.sort(key=rowCount)



figure = plt.figure(1) #Makes a matplotlib figure, the 1 indicates the plot number, plotting more figures at once requires changing that number

figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
axes = figure.add_subplot(111) # This allows for a subplot to be generated. More
#symbols = ['o', '^', '<', 's', 'd', 'p', '*']

offset1mar2 = [64,122,74,88,26,21,64,23,88,  60, 138, 57, 230, 217, 76, 297,185,222]
offset2mar2=[151,245, 142, 299, 155, 153, 200,182, 954,52]
ra = np.r_[np.linspace(0.5,1, len(fileNamesmar2))]
ra2 = np.r_[np.linspace(0.5, 1, len(fileNamesmar2))]
c = plt.get_cmap("Purples")
d = plt.get_cmap("Greens")
e = plt.get_cmap("Oranges")

colors1 = c(ra)
colors2 = d(ra2)
colors3 = e(ra)

def colorpick(nrow, ind):
    if nrow ==1:
        return colors1[ind]
    elif nrow ==2:
        return colors2[ind]
    elif nrow ==3:
        return colors3[ind]
offset1 = [566,139, 175,264,182, 246,351,148,154,215,200, 267,309 ]

offset2 = [450, 868, 655, 574, 91]
offsetboth = [139, 175,264,182, 246,351,148,154,215,200, 267,309,450, 868, 655, 574, 91]
offset1mar2 = [64,122,74,88,26,21,64,23,88,  60, 138, 57, 230, 217, 76, 297,185,222]
offset2mar2=[151,245, 142, 299, 155, 153, 200,182, 954,52]

offsetmar2both = [64,122,74,88,26,21,64,23,88,  60, 138, 57, 230, 217, 76, 297,222,151,245, 142, 299, 155, 153, 200,182, 945,52]
offsetmar2both = [26,21,64,64,23,88,122,60,138, 57,230,74,88,217, 76, 297,222,52,151,245, 142, 299, 155, 153, 200,182, 945]
row = []
rg = []

def plotthings(directory,fileNames, offsets):
    N = []
    row = []
    rg = []
    print(len(fileNames), len(offsets))
    for m in range(len(fileNames)):
#for m in range(1):
        times, curvature = np.genfromtxt(directory+'/'+fileNames[m])
        #st = fileNames[m].partition('N')
        #sep = st[2].partition('d')
        #N.append(int(sep[0]))
        #last = sep[2]
        #print(last)
        #d = int(last.partition('row')[0])
        #rwo = int((last.split('.'))[0].partition('row')[2])
        #row.append(int((last.split('.'))[0].partition('row')[2]))
        rg.append(curvature[-1])
    #axes.plot((times*int(fileNames[m][0:3])/1000)-offsets[m], perimeter,'.', color = colorpick(int(fileNames[m][-11]), m), label = fileNames[m][-11])
        axes.plot((times*int(300)/1000), curvature)
        
        #axes.plot(((times*int(300)/1000)-offsets[m]), curvature,'.', color = colorpick(rwo, m), label = 'N = %i'%N[m])
    

    axes.set_xlabel(r'$t /\textrm{s}$', fontsize = 18) #x and y axis labels
    axes.set_ylabel(r'$\frac {R_g}{(r/2)\sqrt{N}}$', fontsize = 18)
    axes.set_xlim([-200,200])
    axes.tick_params(labelsize = 20)
    #axes.legend(fontsize = 16, frameon = False)
    return N, row, rg

#Nfeb, rowfeb, rgfeb = plotthings(directoryfeb19,fileNamesfeb19, offsetboth)
Nmar, rowmar, rgmar = plotthings(directorymar2, fileNamesmar2, offsetmar2both)
print(Nmar)
plt.show()
exit()
figure = plt.figure(2)
figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
axes = figure.add_subplot(111) 

def plotrgnn(directory, fileNames, N, row):
    for m in range(len(fileNames)):
#for m in range(1):
        perimeter,normperim,  times = np.genfromtxt(directory+'/'+fileNames[m])
    #axes.plot((times*int(fileNames[m][0:3])/1000)-offsets[m], perimeter,'.', color = colorpick(int(fileNames[m][-11]), m), label = fileNames[m][-11])

        axes.plot(N[m], (perimeter[-1]/N[m]**0.5),'.', color = colorpick(row[m], m), label = 'N = %i'%N[m])
    axes.plot(N, np.asarray(N)**0.5*(10/2)*(12**0.25)/(2*np.pi**0.5*np.asarray(N)**0.5), 'k', label = r'$R_g = 12^(1/4)\sqrt(N/\pi)r/2$')

    axes.set_xlabel(r'$N$', fontsize = 18) #x and y axis labels
    axes.set_ylabel(r'$R_g/N^{1/2}$', fontsize = 18)
    axes.tick_params(labelsize = 20)
    axes.legend(fontsize = 16, frameon = False)
#plotrgnn(directoryfeb19, fileNamesfeb19, Nfeb, rowfeb)
plotrgnn(directorymar2, fileNamesmar2, Nmar, rowmar)
plt.show()

figure = plt.figure(3)
figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
axes = figure.add_subplot(111) 
def linfit(x, a):
    return a*x
def plotrgn(fileNames, directory, N, row, rg):
    for m in range(len(fileNames)):
#for m in range(1):
        perimeter,normperim,  times = np.genfromtxt(directory+'/'+fileNames[m])
    #axes.plot((times*int(fileNames[m][0:3])/1000)-offsets[m], perimeter,'.', color = colorpick(int(fileNames[m][-11]), m), label = fileNames[m][-11])

        axes.plot(N[m]**0.5, (perimeter[-1]),'.', color = colorpick(row[m], m), label = 'N = %i'%N[m])
        
    fit = sc.optimize.curve_fit(linfit,np.asarray(N)**0.5, np.asarray(rg))
    print(fit)
    axes.plot(np.linspace(0, 20, 2),linfit((np.linspace(0, 20, 2)), fit[0]), 'r-')
    axes.plot(np.linspace(0, 20, 2), np.sqrt(np.linspace(0, 20, 2))*(10/2)*(12**0.25)/(2*np.pi**0.5), 'k', label = r'$R_g = 12^(1/4)\sqrt(N/\pi)r/2$')


    axes.set_xlabel(r'$N^{1/2}$', fontsize = 18) #x and y axis labels
    axes.set_ylabel(r'$R_g$', fontsize = 18)
    axes.tick_params(labelsize = 20)
    axes.legend(loc = 3, fontsize = 16, frameon = False)


#plotrgn(fileNamesfeb19, directoryfeb19, Nfeb, rowfeb, rgfeb)
plotrgn(fileNamesmar2, directorymar2, Nmar, rowmar, rgmar)
plt.show()


figure = plt.figure(4)
figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
axes = figure.add_subplot(111)
def plotsloperow(directory, fileNames,offset, N, row):
    for m in range(len(fileNames)):
#for m in range(1):
        perimeter,normperim,  times = np.genfromtxt(directory+'/'+fileNames[m])
    #axes.plot((times*int(fileNames[m][0:3])/1000)-offsets[m], perimeter,'.', color = colorpick(int(fileNames[m][-11]), m), label = fileNames[m][-11])
        ind = np.where(times*300/1000 > offset[m])
        
        axes.plot(row[m], (perimeter[0]-perimeter[ind[0][0]])/((times[0]*int(300)/1000)-offset[m]),'.', color = colorpick(row[m], m), label = 'N = %i'%N[m])
    


    axes.set_xlabel(r'$row$', fontsize = 18) #x and y axis labels
    axes.set_ylabel(r'$slope$', fontsize = 18)
    axes.tick_params(labelsize = 20)
#axes.legend(loc = 3, fontsize = 16, frameon = False)
#plotsloperow(directoryfeb19,fileNamesfeb19, offsetboth, Nfeb, rowfeb)
plotsloperow(directorymar2,fileNamesmar2, offsetmar2both, Nmar, rowmar)
plt.show()

'''
###offset test

offset1mar2 = [212,122,76,88,26,63,64,23, 216, 64, 178, 57, 234, 217, 84, 297,185,222,151]

print(fileNames)
for m in range(len(fileNames)):
    figure = plt.figure(1) #Makes a matplotlib figure, the 1 indicates the plot number, plotting more figures at once requires changing that number

    figure.subplots_adjust(top=0.96, bottom=0.2, left=0.16, right=0.96) # adjusts the margins of the plot
    axes = figure.add_subplot(111)
#for m in range(1):
    perimeter,normperim,  times = np.genfromtxt(directory+'/'+fileNames[m])
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
