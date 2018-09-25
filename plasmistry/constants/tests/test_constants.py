#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15:38 2017/9/25

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm Community Edition
"""

from numpy.testing import TestCase,assert_allclose
from plasmistry import constants as const


# %%--------------------------------------------------------------------------------------------- #
class TestConstants(TestCase):
    def setUp(self):
        self.cases = ((const.eV2J, const.eV2K * const.K2J),
                      (const.eV2K * const.K2eV, 1),
                      (const.kB, const.k),
                      (19224.464 * const.WNcm2eV, 2.3835297),# N(2D) energy level
                      (const.pressure_NTP, 101325),
                      (const.pressure_STP, 100000))

    def test_0(self):
        for actual, desired in self.cases:
            assert_allclose(actual,desired)
