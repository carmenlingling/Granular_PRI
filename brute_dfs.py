import numpy as np
import sys
sys.setrecursionlimit(1500)
def neighbours (mask, visited, p):
    nlist = []
    for i in range(-1,2):
        for j in range(-1,2):
            if i==0 and j==0: continue
            px = [p[0]+i, p[1]+j]
            if px in mask and px not in visited:
                nlist.append(px)
    #print(nlist)
    return nlist
    
def dfs(mask, visited, p, all_list):
    # get neighbours
    nlist = neighbours(mask, visited, p)
    visited = visited + [p] # add this to the visited list
    #print(visited)
    res = []
    if len(nlist) == 0:
        print(len(visited), len(all_list))
        return visited
        break
    
    # we are done,
        '''if set(visited) == all_list:
            return [visited]
        else:
            return []'''
    else:
        for pk in nlist:
            r = dfs(mask, visited, pk, all_list)
            res = res + r
                
    return res
                

def get_brute_list(mask):

    start = (mask[0][0], mask[0][1])
    all_list = set([(m[0], m[1]) for m in np.where(mask)])
    visited = dfs(mask, [], start, mask)
    return visited
