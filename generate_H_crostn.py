#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08:16 18-10-25

@author:    Liu Jinbao    
@mail:      liu.jinbao@outlook.com  
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import math
import pandas as pd
from math import log10, log, sqrt, exp
import numpy as np
from matplotlib import pyplot as plt


def generate_energy(_from, _to, _middle):
    energy = np.hstack((np.arange(_from, _middle, step=1e-2),
                        np.logspace(log10(_middle), log10(_to))))
    assert np.diff(energy).all()

    return energy


def func_1_to_2(E):
    dE = 10.2
    # a, b, c = 0.114, 0.0575, 0.1795   1s /to 2s
    # A_seq = [0.0, 0.88606, -2.7990, 5.9451, -7.6948, 4.4152]    # 1s /to 2s
    
    a, b, c = 0.228, 0.1865, 0.5025  # 1s /to n=2
    A_seq = [4.4979, 1.4182, -20.877, 49.735, -46.249, 17.442]  # 1s /to n=2
    
    if 10.2 < E < 11.56:
        return a + b * (E - dE)
    elif 11.56 <= E < 12.23:
        return c
    elif E >= 12.23:
        x = E / dE
        return 5.984 / E * (A_seq[0] * log(x) +
                            A_seq[1] / x ** 0 +
                            A_seq[2] / x ** 1 +
                            A_seq[3] / x ** 2 +
                            A_seq[4] / x ** 3 +
                            A_seq[5] / x ** 4)
    else:
        return 0.0


def func_1_to_3_4_5(E, n, m):
    if n == 1 and m == 3:
        dE = 12.09
        a = 0.38277
        A_seq = [0.75448, 0.42956, -0.58288, 1.0693, 0.00]
    elif n == 1 and m == 4:
        dE = 12.75
        a = 0.41844
        A_seq = [0.24300, 0.24846, 0.19701, 0.00, 0.00]
    elif n == 1 and m == 5:
        dE = 13.06
        a = 0.45929
        A_seq = [0.11508, 0.13092, 0.23581, 0.00, 0.00]
    elif n == 2 and m == 3:
        dE = 12.0875051 - 10.1988357
        a = 1.3196
        A_seq = [38.906, 5.2373, 119.25, -595.39, 816.71]
    else:
        raise Exception("n is not 3, 4 or 5.")
    
    x = E / dE
    if E < dE:
        return 0.0
    else:
        return 5.984 / E * (1 - 1 / x) ** a * (A_seq[0] * log(x) +
                                               A_seq[1] / x ** 0 +
                                               A_seq[2] / x ** 1 +
                                               A_seq[3] / x ** 2 +
                                               A_seq[4] / x ** 3)


def func_n_m(E, n, m):
    def fnm(n, m):
        if n == 2:
            g0, g1, g2 = 1.0785, -0.2319, 0.02947
        elif n >= 3:
            g0 = 0.9935 + 0.2328 / n - 0.1296 / n ** 2
            g1 = -1 / n * (0.6282 - 0.5598 / n + 0.5299 / n ** 2)
            g2 = 1 / n ** 2 * (0.3887 - 1.181 / n + 1.470 / n ** 2)
        else:
            raise Exception("n is error.")
        ynm = 1 - n ** 2 / m ** 2
        gnm = g0 + g1 / ynm + g2 / ynm ** 2
        fnm = gnm / ynm ** 3 * n / m ** 3 * 32 / 3 / sqrt(3) / math.pi
        return fnm
    
    ynm = 1 - n ** 2 / m ** 2
    bn = 1 / n * (4.0 - 18.63 / n + 36.24 / n ** 2 - 28.09 / n ** 3)
    rn = 1.94 / n ** 1.57
    Anm = 2 * n ** 2 * fnm(n, m) / ynm
    Bnm = 4 * n ** 4 / m ** 3 / ynm ** 2 * (1 + 4 / 3 / ynm + bn / ynm ** 2)
    dEnm = 13.6*(1/n**2 - 1/m**2)
    xnm = E/dEnm
    if E<dEnm:
        return 0.0
    else:
        return 1.76*n**2/ynm/xnm*(1-exp(-rn*ynm*xnm))*(Anm*(log(xnm)+1/2/xnm) +
                                                       (Bnm-Anm*log(2*n**2/ynm))*(1-1/xnm))
    print(fnm(n, m))


# energy = generate_energy(10, 1e4, 1e2)
# crostn_1_2 = np.array([func_1_to_2(_) for _ in energy])
# plt.loglog(energy, crostn_1_2)
# crostn_1_3 = np.array([func_1_to_3_4_5(_, 1, 3) for _ in energy])
# plt.loglog(energy, crostn_1_3)
# crostn_1_4 = np.array([func_1_to_3_4_5(_, 1, 4) for _ in energy])
# plt.loglog(energy, crostn_1_4)
# crostn_1_5 = np.array([func_1_to_3_4_5(_, 1, 5) for _ in energy])
# plt.loglog(energy, crostn_1_5)
energy_2_3 = generate_energy(1, 1e4, 1e2)
crostn_2_3 = np.array([func_1_to_3_4_5(_, 2, 3) for _ in energy_2_3])
plt.loglog(energy_2_3, crostn_2_3)
energy_2_4 = generate_energy(1, 1e4, 1e2)
crostn_2_4 = np.array([func_n_m(_, 2, 4) for _ in energy_2_4])
plt.loglog(energy_2_4, crostn_2_4)
crostn_2_5 = np.array([func_n_m(_, 2, 5) for _ in energy_2_4])
plt.loglog(energy_2_4, crostn_2_5)
crostn_2_6 = np.array([func_n_m(_, 2, 6) for _ in energy_2_4])
plt.loglog(energy_2_4, crostn_2_6)

# for _ in (crostn_1_2, crostn_1_3, crostn_1_4, crostn_1_5, crostn_2_3):
#     plt.loglog(energy, _)

r"""
output = np.vstack((energy, crostn_1_2)).transpose()
np.savetxt('H(1)_to_H(2).csv', output, delimiter='\t')
output = np.vstack((energy, crostn_1_3)).transpose()
np.savetxt('H(1)_to_H(3).csv', output, delimiter='\t')
output = np.vstack((energy, crostn_1_4)).transpose()
np.savetxt('H(1)_to_H(4).csv', output, delimiter='\t')
output = np.vstack((energy, crostn_1_5)).transpose()
np.savetxt('H(1)_to_H(5).csv', output, delimiter='\t')
"""
output = np.vstack((energy_2_3, crostn_2_3)).transpose()
np.savetxt('H(2)_to_H(3).csv', output, delimiter='\t')
output = np.vstack((energy_2_4, crostn_2_4)).transpose()
np.savetxt('H(2)_to_H(4).csv', output, delimiter='\t')
output = np.vstack((energy_2_4, crostn_2_5)).transpose()
np.savetxt('H(2)_to_H(5).csv', output, delimiter='\t')
output = np.vstack((energy_2_4, crostn_2_6)).transpose()
np.savetxt('H(2)_to_H(6).csv', output, delimiter='\t')
