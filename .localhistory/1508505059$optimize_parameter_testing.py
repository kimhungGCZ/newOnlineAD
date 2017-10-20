import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fitFunc(x, a, b, c, d):
    return a + b*x[0] + c*x[1] + d*x[0]*x[1]

x_3d = np.array([[1,2,3,4,6],[4,5,6,7,8]])

p0 = [5.11, 3.9, 5.3, 2]

fitParams, fitCovariances = curve_fit(fitFunc, x_3d, x_3d[1,:], p0)
print ' fit coefficients:\n', fitParams