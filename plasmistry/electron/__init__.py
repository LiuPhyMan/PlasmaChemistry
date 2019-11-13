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
from scipy.integrate import simps, trapz
from .. import constants as const
from .distri_func import get_maxwell_eedf

__all__ = [s for s in dir() if not s.startswith('_')]


def get_rate_const_from_crostn(*, Te_eV, crostn):
    assert crostn.shape[0] == 2
    _energy = crostn[0] * const.eV2J
    _crostn = crostn[1]
    _f = get_maxwell_eedf(_energy, Te_eV=Te_eV)
    # k = np.sqrt(2 / const.m_e) * simps(np.sqrt(_energy) * _crostn * _f, _energy)
    k = np.sqrt(2 / const.m_e) * trapz(np.sqrt(_energy) * _crostn * _f, _energy)
    return k


from numpy.testing import Tester

test = Tester().test
