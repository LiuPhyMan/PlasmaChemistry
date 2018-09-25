#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1:36 2017/7/14

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import
import math
import numpy as np
from scipy.integrate import simps
from plasmistry.electron import get_maxwell_eedf, get_maxwell_eepf
from plasmistry import constants as const
from numpy.testing import assert_approx_equal, assert_array_almost_equal, assert_allclose


# %%--------------------------------------------------------------------------------------------- #
def test_get_maxwell_eedf_eepf():
    elec_energy = np.logspace(-2, 2, num=100) * const.eV2J
    eedf = get_maxwell_eedf(elec_energy, Te_eV=1.0)
    eepf = get_maxwell_eepf(elec_energy, Te_eV=1.0)

    assert_allclose(eedf, eepf * np.sqrt(elec_energy))
    assert_approx_equal(simps(eedf, elec_energy), 1.0, significant=3)
    assert_approx_equal(eedf.max() / const.J2eV,
                        2 * math.sqrt(1 / 2 / math.pi) * math.exp(-1 / 2), significant=3)
    assert_approx_equal(simps(eepf * np.sqrt(elec_energy), elec_energy), 1.0, significant=3)
    assert_approx_equal(simps(eedf * elec_energy, elec_energy), 1.5 * const.eV2J, significant=3)

# %%--------------------------------------------------------------------------------------------- #
