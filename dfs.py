import numpy as np
def neighbours (mask, visited, p):
    
    nlist = []
    for i in range(-1,2):
        for j in range(-1,2):
            if i==0 and j==0: continue
            px = [p[0]+i, p[1]+j]
            if px in mask and px not in visited:
                if len(nlist)>0 and abs(i+j)>1:
                    pass
                
    print('nlist',nlist)
    return nlist
    
def dfs(mask, visited, p, start = True):
    # get neighbours #data, list of places you've gone, currentlocation)
    
    nlist = neighbours(mask, visited, p)
    visited.append(p) # add this to the visited list
    print('visited',visited)
    if len(nlist) == 0:
        # we are done,
        return
    elif len(nlist) == 1:
        # easy case, only 1 neighbour

        dfs(mask, visited, nlist[0], False)

    elif len(nlist) == 2 and start:
        # we are starting, pick the rightmost one.
        if nlist[0][1] < nlist[1][1]:
            dfs(mask, visited, nlist[1], False)
        elif nlist[0][1] > nlist[1][1]:
            dfs(mask, visited, nlist[0], False)
        else:
            if nlist[0][0] > nlist[1][0]:
                dfs(mask, visited, nlist[0], False)
            elif nlist[0][0] < nlist[1][0]:
                dfs(mask, visited, nlist[1], False)
            else:
                
                print('assumptions violated: starting order')
                assert(False)
                
    elif len(nlist) == 2:
        # this is the hard part, need to determine which one to visit first
        pxlist = [neighbours(mask, visited, px) for px in nlist]
        print('wrongway',pxlist)
        # check to see if each of these points have a neighbour in common
        vaild_assumption = (px[1] in pxlist[0]) and (px[0] in pxlist[1])
        if not valid_assumption:
            print('assumption violated, the pixels dont have a common neighbour.')
            assert(False)
        # pick the one with the least amount of neighbours
        if len(pxlist[0]) > len(pxlist[1]):
            dfs(mask, visited, nlist[1], False)
        else:
            dfs(mask, visited, nlist[0], False)
        
    else:
        print('assumptions violated')
        assert(False)

def get_list(mask):
    visited = []
    start = (mask[0])
    dfs(mask, visited, start, True)
    return visited
