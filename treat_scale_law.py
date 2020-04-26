#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  20:17 2019/5/29

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

from plasmistry import constants as const
from math import exp
import math
import numpy as np


def G(Tgas, mu_mass, we, xe, L, v):
    return v / (1 - xe * v) * F(Lambda(Tgas, mu_mass, we, xe, L, v))


def F(x):
    print(f"lambda={x}")
    return 0.5 * (3 - exp(-2 * x / 3)) * exp(-2 * x / 3)


def Lambda(Tgas, mu_mass, we, xe, L, v):
    # v => v-1
    theta_ij = 16 * math.pi ** 4 * mu_mass * const.c ** 2 * (
                we * 1e2) ** 2 * L ** 2 / const.k
    theta_i = we * const.WNcm2K
    dE = we * (1 - 2 * v * xe) * const.WNcm2K
    return 2 ** (-1.5) * math.sqrt(theta_ij / Tgas) * dE / theta_i


if __name__ == "__main__":
    # we = 2170
    # xe = 13.3 / we
    # Tgas = 1000
    # mu_mass = 14 * const.atomic_mass
    # L = 0.2e-10
    # v = 10
    # _L = Lambda(Tgas, mu_mass, we, xe, L, v)
    # print(G(Tgas, mu_mass, we, xe, 0.2e-10, 10))
    _data = np.loadtxt("_cs_list/CO2_dis/CO2v0_ele_diss_Polak_7.5_eV.dat")
    _a = ["a", 'b', 'c', 'd'] + list(range(1, 22))
    for _v, dE in zip(_a, ([0.08, 0.17, 0.25, 0.33,
                            0.29, 0.58, 0.87, 1.15, 1.43,
                            1.70, 1.98, 2.25, 2.51, 2.78,
                            3.04, 3.29, 3.55, 3.80, 4.05,
                            4.29, 4.53, 4.77, 5.01, 5.24, 5.47])):
        _new_data = np.vstack((_data[:, 0] - dE, _data[:, 1])).transpose()
        np.savetxt(f"_cs_list/CO2_dis/CO2v{_v}_ele_diss_Polak_7.5_eV.dat",
                   _new_data)

    # print(_L)
    # print(F(Lambda(Tgas, mu_mass, we, xe, 0.2e-10, v)))
    # print(F(Lambda(Tgas, mu_mass, we, xe, 0.16e-10, v)))
