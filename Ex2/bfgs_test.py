import numpy as np
from scipy.optimize import fmin_bfgs

def f(xvec, coef):
    return coef[0]*xvec[0]**2 + coef[1]*xvec[0]*xvec[1] + coef[2]*xvec[1]**2

def df(xvec, coef):
    d0 = 2*coef[0]*xvec[0] + coef[1]*xvec[1]
    d1 = coef[1]*xvec[0] + 2*coef[2]*xvec[1]
    return np.asarray([d0, d1])

inix = np.asarray([1, -2])
c = np.asarray([1, -2, 1])
minim = fmin_bfgs(f, inix, df, args=(c,))

print(minim)
