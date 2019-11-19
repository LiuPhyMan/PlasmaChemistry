#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19:32 2019/11/19

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

from math import exp
import numpy as np
import numba
from numba import float32


@numba.jit(nopython=True)
def test(Tgas, value):
    """__REPLACE__"""
