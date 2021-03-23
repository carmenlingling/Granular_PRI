import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

def R_g(xs, ys):
    xavg = np.mean(xs)
    yavg = np.mean(ys)
    R = -(xavg**2+yavg**2)
    for item in range(len(xs)):
        R += (xs[item]**2+ys[item]**2)/len(xs)
    return np.sqrt(R)

def Rsquared(pos):
    x = pos[0]
    y = pos[1]
    R2 = x**2 + y**2
    return R2

def cycle(radius, N):
    a = radius
    b = -radius
    c = 0
    N-=1
    hexes = [[a,b,c]]
    
    while a >0 and N>0:
        a -=1
        c +=1
        hexes.append([a,b,c])
        N-=1
    while b <0 and N>0:
        a-=1
        b+=1
        hexes.append([a,b,c])
        N-=1
    while c>0 and N>0:
        c-=1
        b+=1
        hexes.append([a,b,c])
        N-=1
    while a<0 and N>0:
        a+=1
        c-=1
        hexes.append([a,b,c])
        N-=1
    while b>0 and N>0:
        b-=1
        a+=1
        hexes.append([a,b,c])
        N-=1
    while c<-1 and N>0:
        c+=1
        b-=1
        hexes.append([a,b,c])
        N-=1
    return hexes
def hex_to_cart(hex, r):
    x = (2*r)*hex[0]
    y = 2.*(2*r) * np.sin(np.pi/3) * (hex[1] - hex[2])/3
    return(x, y)

#print(cycle(2, 10))
def placement_algo(N, r):
    hexpositions = [[0,0,0]]
    current_rad = 1
    N-=1
    while N > 0:
        if N > current_rad*6:
            pos = cycle(current_rad, current_rad*6)
            hexpositions = hexpositions+pos
            N-= current_rad*6
            current_rad+=1
        else:
            hexpositions =hexpositions+cycle(current_rad, N)
            N-=N
    cart=[]
    for n in hexpositions:
        cart.append(hex_to_cart(n, r))
    
    return(np.asarray(cart))
    
#N  = 400
##positions = placement_algo(N, 10*235/6.7)
def rgdata(positions):
    rgs =[0]
    for n in range(1,len(positions)):
        rgs.append(R_g(positions[0:n+1,0], positions[0:n+1,1]))
    return rgs
def rgcont(N, r):
    return (N**0.5)*(r/2)
#ra = np.r_[np.linspace(0.2,1, len(positions[0]))]
c = plt.get_cmap("Purples")
#colors1 = c(ra)
fig, ax = plt.subplots(figsize = [6,6])

'''post = placement_algo(91,0.35)
plt.plot(post[:,0], post[:,1], 'o', markersize = 20)
ax.set(xlim = [-6, 6], ylim = [-6,6])
plt.show();'''
