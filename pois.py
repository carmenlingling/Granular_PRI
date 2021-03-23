
import numpy as np
x = np.linspace(0, 31, 1)
def pois(number):
    a = np.exp(-5)
    b = 5**number
    c = np.math.factorial(number)
    return(a*b/c)

