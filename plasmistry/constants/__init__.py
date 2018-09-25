#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16:58 2017/7/3

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm

Add some unit conversion factor to module scipy.constants.

"""
from scipy import constants
from scipy.constants import *

# %%--------------------------------------------------------------------------------------------- #
for _ in ('kB',
          'eV2J', 'J2eV', 'kJ2eV',
          'eV2K', 'K2eV',
          'J2K', 'K2J',
          'cal2J', 'J2cal',
          'Kcal2J', 'J2Kcal',
          'cm2eV', 'eV2cm'):
    assert _ not in constants.__dict__
del constants
# %%--------------------------------------------------------------------------------------------- #
kB = k

eV2J = e
J2eV = 1 / eV2J
kJ2eV = 1e3 * J2eV

K2J = kB
J2K = 1 / K2J

eV2K = eV2J * J2K
K2eV = 1 / eV2K

cal2J = calorie_th
J2cal = 1 / calorie_th

Kcal2J = 1e3 * cal2J
J2Kcal = 1 / Kcal2J

WNcm2J = h * c * 1e2
J2WNcm = 1 / WNcm2J

WNcm2eV = WNcm2J * J2eV
eV2WNcm = 1 / WNcm2eV

WNcm2K = WNcm2eV * eV2K
K2WNcm = 1 / WNcm2K

# %%--------------------------------------------------------------------------------------------- #
# STP, standard temperature and pressure.
# %%--------------------------------------------------------------------------------------------- #
temperature_STP = convert_temperature(0.0, 'Celsius', 'Kelvin')
pressure_STP = bar  # 100000

# %%--------------------------------------------------------------------------------------------- #
# NTP, normal temperature and pressure.
# %%--------------------------------------------------------------------------------------------- #
temperature_NTP = convert_temperature(20.0, 'Celsius', 'Kelvin')
pressure_NTP = atmosphere  # 101325

# %%--------------------------------------------------------------------------------------------- #
__all__ = [_ for _ in dir() if not _.startswith('_')]
from numpy.testing import Tester

test = Tester().test
