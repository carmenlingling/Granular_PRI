import numpy as np
import sys
import queue
import matplotlib.pyplot as plt


def dot(vec1, vec2):
    return(vec1[0]*vec2[0]+vec1[1]*vec2[1])


def angle_calc(p1, p2, midpoint):
    v1 = [p1[0]-midpoint[0],p1[1]-midpoint[1]]
    v2 = [p2[0]-midpoint[0], p2[1]-midpoint[1]]
    
    angle = np.arccos(dot(v1,v2)/ (np.sqrt(dot(v1,v1))*np.sqrt(dot(v2,v2))))
    return(angle)



def neighbours (mask, visited, p, midpoint): #This function takes the point in question, p and sees if there are any neighbours one square away, then it goes through to see which one is most clockwise if there are more than one new neighbour
    nlist = []
    for i in range(-1,2):
        for j in range(-1,2):
            if i==0 and j==0: continue
    
            px = [p[0]+i, p[1]+j]
            if px in mask and px not in visited: #checks if there are any neighbours that aren't already checked
                nlist.append(px)
    ##This deals with there being more than one neighbour. This won't be used if the shape is a perfectly smooth shape.            
    if len(nlist)>1: 
        minusangle = 2*np.pi
        plusangle = 0
        
        for k in range(len(nlist)):
            temp = angle_calc([midpoint[0],0], nlist[k] ,midpoint)
            if nlist[k][1] < midpoint[1]:
                if temp < minusangle:
                    minusangle = temp
                    cw = [nlist[k]]

            elif nlist[k][1] > midpoint[1]:
                
                if temp > plusangle:
                    plusangle = temp
                    cw = [nlist[k]]
            else:
                cw = nlist
            
        nlist = cw

    #This checks if there are no nearest neighbours and expands the search
    if len(nlist) == 0:
        for i in range(-2,3):
            for j in range(-2,3):
                if abs(i)<=1 and abs(j)<=1: continue
                px = [p[0]+i, p[1]+j]
                if px in mask and px not in visited:
                    nlist.append(px)
    
    return nlist

    

                

def get_dfs_list(mask, midpoint): #mask= a structured list of 0, 1 to indicate where there are edge points and where there are non-edge points, must be continuous. midpoint= the average of those values
   
    maskind = np.asarray(mask)
    
    startind = np.argmin(maskind[:,0])
    
    start = (maskind[startind,0], maskind[startind,1])
    visited = dfs_iterative(mask, start, midpoint)
    print('Done sorting!')
    return visited




def dfs_iterative(mask, start, midpoint): #this function is what orders the array. it takes the array of 1 and 0, the starting pixel, and the midpoint. This is an iterative process
    visited = []
    
    stack = neighbours(mask, visited, start, midpoint)
    
    res = []
    while len(stack) != 0:
        p = stack.pop()
        
        visited = visited + [p]
        n = neighbours(mask, visited+stack, p, midpoint)
        
        stack = stack + neighbours(mask, visited+stack, p, midpoint)
        
    return visited
