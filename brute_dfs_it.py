import numpy as np
import sys
import queue
sys.setrecursionlimit(1500)
iterations=0
def neighbours (mask, visited, p):
    nlist = []
    for i in range(-1,2):
        for j in range(-1,2):
            if i==0 and j==0: continue
            px = [p[0]+i, p[1]+j]
            if px in mask and px not in visited:
                nlist.append(px)
    if len(nlist) ==0:
        for i in range(-2,3):
            for j in range(-2,3):
                if i==0 and j==0: continue
                px = [p[0]+i, p[1]+j]
                if px in mask and px not in visited:
                    nlist.append(px)
    #print(nlist)
    return nlist
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
                

def get_brute_list(mask):
    
    start = (mask[0][0], mask[0][1])
    all_list = set([(m[0], m[1]) for m in np.where(mask)])
    visited = dfs_iterative(mask, start)
    return visited

def dfs_iterative(mask, start):
    visited = []
    #print(start)
    stack = neighbours(mask, visited, start)
    res = []
    while len(stack) != 0:
        p = stack.pop()
        #print(p)
        visited = visited + [p]
        stack = stack + neighbours(mask, visited+stack, p)
        
    return visited
