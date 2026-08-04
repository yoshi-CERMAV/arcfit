import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import bisplrep, bisplev
from numpy.linalg import lstsq

gauss_cof = np.log(2) * 4

def arg(x, pos, sigma):
    x1 = x - pos
    rsigma = 1. / sigma
    x1 *= rsigma
    x1 *= x1
    return x1

def arg_fac(x, pos, sigma):
    x1 = x - pos
    rsigma = 1. / sigma
    x1 *= rsigma
    fac = 2. * rsigma * x1
    return x1, fac

def gaussian(x, pos, sigma):
    return np.exp(-gauss_cof * arg(x, pos, sigma))

def dgaussian(x, pos, sigma):
    arg_val, fac = arg_fac(x, pos, sigma)
    y = np.exp(-gauss_cof * arg_val * arg_val)
    dpos = gauss_cof * y * fac
    return [y, dpos, dpos * arg_val]

def gaussian2d(x, y, I, pos_x, pos_y, sigma_x, sigma_y):
    return I * np.exp(-gauss_cof * arg(x, pos_x, sigma_x)) * np.exp(-gauss_cof * arg(y, pos_y, sigma_y))

def dgaussian2d(x, y, I, pos_x, pos_y, sigma_x, sigma_y):
    dx = dgaussian(x, pos_x, sigma_x)
    dy = dgaussian(y, pos_y, sigma_y)
    return [
        dx[0] * dy[0],
        I * dx[1] * dy[0],
        I * dx[0] * dy[1],
        I * dx[2] * dy[0],
        I * dx[0] * dy[2],
    ]

def func(var, *data):
    return (
        gaussian2d(data[0], data[1], var[0], var[1], var[2], var[3], var[4])
        + var[5]
        + var[6] * data[1]
        - data[2]
    )

def jacobian(var, *data):
    t = dgaussian2d(data[0], data[1], var[0], var[1], var[2], var[3], var[4])
    t.append(np.ones(data[0].size))
    t.append(data[1])
    return np.array(t).transpose()
