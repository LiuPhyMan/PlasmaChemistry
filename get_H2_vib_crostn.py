#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15:02 2018/10/16

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

import numpy as np
from scipy.integrate import trapz, simps
from scipy.interpolate import interp1d
from math import sqrt, exp, log
from matplotlib import pyplot as plt
from plasmistry.electron import get_maxwell_eepf
from plasmistry import constants as const
from plasmistry.molecule import get_vib_energy


def get_cs01():
    dE = 0.516
    Ev = 1
    energy_eV = np.logspace(-4, 3, num=10000)
    x = energy_eV[energy_eV > dE] / dE
    crostn_01 = np.zeros_like(energy_eV)
    crostn_01[energy_eV > dE] = 5.78 * Ev / dE ** 4 / x ** 2 * (1 - 1 / x) ** (
            6.11 / sqrt(dE)) * 1e-20
    # crostn_01 = interp1d(energy_eV, crostn_01,
    #                      bounds_error=False, fill_value=0.0)(np.logspace(-3, 3, num=1000000))
    # return np.logspace(-3, 3, num=1000000), crostn_01
    return energy_eV, crostn_01


def get_rate_const(energy_eV, crostn, Te_eV):
    eepf = get_maxwell_eepf(energy_eV * const.eV2J, Te_eV=Te_eV)
    k = simps(eepf * energy_eV * const.eV2J * crostn, energy_eV * const.eV2J) * sqrt(2 / const.m_e)
    return k * 1e6


def shift(energy_eV, crostn, d_e):
    energy_new = energy_eV + d_e
    crostn_new = crostn[energy_new > 0.0]
    energy_new = energy_new[energy_new > 0.0]
    return energy_new, crostn_new


def scale_v(v0):
    r"""Fit the reference result."""
    return 1 + 0.65504 * v0 - 0.03523 * v0 ** 2 + 0.02801 * v0 ** 3


def scale_dv(dv):
    r"""
    References
    ----------
        Shakhatov, V. A. and Y. A. Lebedev (2011).
        "Collisional-radiative model of hydrogen low-temperature plasma:
        Processes and cross sections of electron-molecule collisions."
        High Temperature 49(2): 257-302.
    """
    a = 8.23718
    b = 5.79598
    c = -4.76075
    d = 1.15
    f = 2
    temp = exp(a * dv + c * dv ** d) / (1 + dv ** f) ** b
    return temp / (exp(a + c) / 2 ** b)


def get_crostn(v0, v1):
    energy_eV, crostn_01 = get_cs01()
    dv = v1 - v0
    cs_v0_v1 = crostn_01 * scale_v(v0) * scale_dv(dv)
    d_e = get_vib_energy('H2', quantum_number=v1) - get_vib_energy('H2', quantum_number=v0) - \
          get_vib_energy('H2', quantum_number=1) + get_vib_energy('H2', quantum_number=0)
    return shift(energy_eV, cs_v0_v1, d_e)


# for v0, v1 in ((0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (3, 6)):
# for v0, v1 in ((3, 4), (3, 5), (3, 6)):
for v0 in range(15):
    for v1 in range(15):
        if v1 > v0:
            x = get_crostn(v0, v1)[0]
            y = get_crostn(v0, v1)[1]
            output = np.vstack((x, y)).transpose()
        elif v1 < v0:
            e_thres = get_vib_energy('H2', quantum_number=v0) - \
                      get_vib_energy('H2', quantum_number=v1)
            x = get_crostn(v1, v0)[0]
            y = get_crostn(v1, v0)[1]
            _chosen = x > e_thres
            _x = (x - e_thres)[_chosen]
            _y = y[_chosen] * x[_chosen] / (x[_chosen] - e_thres)
            output = np.vstack((_x, _y)).transpose()
        else:
            continue
        _file_path = 'H2(v{0})_to_H2(v{1}).csv'.format(v0, v1)
        _file_path = _file_path.replace('(v0)', '')
        np.savetxt(_file_path, output, delimiter='\t')

    # plt.loglog(get_crostn(v0, v1)[0], get_crostn(v0, v1)[1] * 1e20,
    #            label='{0}{1}'.format(v0, v1))
# plt.legend()
# x = []
# y = []
# for _ in np.logspace(-2, 3, num=50):
#     x.append(_)
#     y.append(get_rate_const(energy_eV, crostn, Te_eV=_))

# x0 = np.loadtxt('D:/Desktop/Untitled-2.txt')[:, 0]
# y0 = np.loadtxt('D:/Desktop/Untitled-2.txt')[:, 1]

# plt.plot(energy_eV, crostn)
# plt.loglog(x, y)
# plt.loglog(x0, y0)


r"""
plt.plot(np.arange(15), Fvv)
# coefs = np.loadtxt(r'D:/Desktop/Untitled-1.txt')
from plasmistry import constants as const


def plot_crostn(coefs):
    energy = np.logspace(-2, 3, num=50)
    crostn = np.zeros_like(energy)
    for i in range(9):
        crostn = crostn + coefs[i] * np.log(energy) ** i
    crostn = np.exp(crostn)
    crostn[crostn < 0] = 0.0
    return energy, crostn


energy, crostn = plot_crostn([-2.019865E+01,
                              9.563689E-01,
                              -6.930433E-01,
                              1.672170E-01,
                              -3.218185E-02,
                              5.798138E-03,
                              -8.494786E-04,
                              7.361712E-05,
                              -2.624614E-06
                              ])
"""
