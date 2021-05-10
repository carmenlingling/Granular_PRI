import numpy as np
import sys
import queue
import matplotlib.pyplot as plt
sys.setrecursionlimit(1500)
iterations=0

def dot(vec1, vec2):
    return(vec1[0]*vec2[0]+vec1[1]*vec2[1])


def angle_calc(p1, p2, midpoint):
    v1 = [p1[0]-midpoint[0],p1[1]-midpoint[1]]
    v2 = [p2[0]-midpoint[0], p2[1]-midpoint[1]]
    
    angle = np.arccos(dot(v1,v2)/ (np.sqrt(dot(v1,v1))*np.sqrt(dot(v2,v2))))
    return(angle)



def neighbours (mask, visited, p, midpoint):
    nlist = []
    for i in range(-1,2):
        for j in range(-1,2):
            if i==0 and j==0: continue
            #print(p)
            #print(p[0]+i, p[1]+j)
            px = [p[0]+i, p[1]+j]
            if px in mask and px not in visited:
                nlist.append(px)
    if len(nlist)>1:
        minusangle = 2*np.pi
        plusangle = 0
        #print(nlist)
        for k in range(len(nlist)):
            '''plt.plot([midpoint[0],midpoint[0]], [0,midpoint[1]])
            #plt.plot(midpoint[0], midpoint[1], 's')
            plt.plot(nlist[k][0], nlist[k][1], 'o')
            plt.show()'''
            
            temp = angle_calc([midpoint[0],0], nlist[k] ,midpoint)
            
            
            if nlist[k][1] < midpoint[1]:
                
                if temp < minusangle:
                    minusangle = temp
                    #print(minusangle*360/(2*np.pi))
                    cw = [nlist[k]]
            elif nlist[k][1] > midpoint[1]:
                
                if temp > plusangle:
                    plusangle = temp
                    #print(plusangle*360/(2*np.pi))
                    cw = [nlist[k]]
                
                #print(nlist, 'angle', ccw)
            else:
                cw = nlist
            #print(ccw)
        nlist = cw
        #print(nlist)
    if len(nlist) == 0:
        for i in range(-2,3):
            for j in range(-2,3):
                if abs(i)<=1 and abs(j)<=1: continue
                px = [p[0]+i, p[1]+j]
                if px in mask and px not in visited:
                    nlist.append(px)
    #print(nlist)
    return nlist

    
def dfs(mask, visited, p, all_list):
    # get neighbours
    global iterations
    iterations+=1
    print(iterations)
    nlist = nn(mask, visited, p)
    visited = visited + [p] # add this to the visited list
    #print(visited)
    res = []
    if len(nlist) == 0:
        print(len(visited), len(all_list))
        # we are done,
        if set(visited) == all_list:
            iterations-=1
            return [visited]
        else:
            iterations-=1
            return []
    else:
        for pk in nlist:
            r = dfs(mask, visited, pk, all_list)
            res = res + r
    iterations-=1
    return res
                

def get_brute_list(mask, midpoint):
    #midpoint = [np.average(mask[:,0]),np.average(mask[:,1])]
    maskind = np.asarray(mask)
    #print(maskind[:,1])
    startind = np.argmin(maskind[:,0])
    print(maskind.shape, startind)
    start = (maskind[startind,0], maskind[startind,1])
    #all_list = set([(m[0], m[1]) for m in np.where(mask)])
    visited = dfs_iterative(mask, start, midpoint)
    print('Done sorting!')
    return visited

def dfs_iterative(mask, start, midpoint):
    visited = []
    #print(start)
    stack = neighbours(mask, visited, start, midpoint)
    
    res = []
    while len(stack) != 0:
        p = stack.pop()
        
        visited = visited + [p]
        n = neighbours(mask, visited+stack, p, midpoint)
        
        stack = stack + neighbours(mask, visited+stack, p, midpoint)
        
    return visited



def nn(mask, visited, p):
    nlist = []
    a = -1
    b = 2
    while len(nlist) ==0 and b<7:
        tempnlist = []
        for i in range(-a,b):
            for j in range(-a,b):
                if i==0 and j==0: continue
                px = [p[0]+i, p[1]+j]
                if px in mask and px not in visited:
                    tempnlist.append(px)
        if len(tempnlist) == 1:
            nlist.append(tempnlist[0])
        elif len(tempnlist) > 1:
            dist = 100000
            minpos = [0,0]
            for k in range(len(tempnlist)):
                tdist = (p[0]-tempnlist[k][0])**2 + (p[1]-tempnlist[k][1])**2
                if tdist < dist:
                    dist = tdist
                    minpos = tempnlist[k]
            nlist.append(minpos)
        else:
            a-=1
            b+=1
