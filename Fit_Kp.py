# -*- coding: utf-8 -*-

import math
import numpy as np

from scipy.optimize import curve_fit

Tgas = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
logKp = [-47.959, -20.020, -10.704, -6.055, -3.274, -1.427,
         -0.112, 0.870, 1.630, 2.236]


def func(x, A, n, E):
    return np.log10(A * x ** n * np.exp(-E / x))

popt, pcov = curve_fit(func, Tgas, logKp)
