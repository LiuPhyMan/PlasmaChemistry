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




@numba.jit("void(float64[:], float64)", nopython=True, nogil=False, parallel=True, fastmath=True)
def test(value, Tgas):
    """__REPLACE__"""

