#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21:00 2017/7/13

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import
from .distri_func import *
from .cal_eedf import *
from scipy.interpolate import interp1d
from scipy.integrate import simps
from .distri_func import get_maxwell_eedf
__all__ = [s for s in dir() if not s.startswith('_')]

def get_rate_const_from_crostn(*, Te_eV, crostn):
    _energy = crostn[0]
    _crostn = crostn[1]
    _f = get_maxwell_eedf(_energy, Te_eV)




from numpy.testing import Tester

test = Tester().test
