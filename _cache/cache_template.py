#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19:32 2019/11/19

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

from math import exp, sqrt
import numpy as np
import numba
from numba import float64, void



@numba.jit("float64(float64, float64, float64, float64)", nopython=True, nogil=False, parallel=True, fastmath=True)
def LT(A, B, C, Tgas):
    return A * exp(B*Tgas**(-1/3)+C*Tgas**(-2/3))

@numba.jit("float64(float64, float64, float64, float64)", nopython=True, nogil=False, parallel=True, fastmath=True)
def Lij(dE, a, mu, Tgas):
    return 0.32*dE*11604.5/a*sqrt(mu/Tgas)

@numba.jit("float64(float64)", nopython=True, nogil=False, parallel=True, fastmath=True)
def F(_Lij):
    return 0.5*(3-exp(-2/3*_Lij))*exp(-2/3*_Lij)

@numba.jit("float64[:](float64)", nopython=True, nogil=False, parallel=True, fastmath=True)
def test(Tgas):
    """__REPLACE__"""
    return value

