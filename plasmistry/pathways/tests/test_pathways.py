#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22:27 2018/7/9

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

from __future__ import division, print_function, absolute_import
from plasmistry.reactions import CoefReactions
from plasmistry.pathways import Pathways
import numpy as np
from scipy import sparse as spr
from numpy.testing import (assert_allclose,
                           TestCase)

import warnings

warnings.filterwarnings('ignore', module='scipy.sparse')


class test_Pathways(TestCase):

    def setUp(self):
        rctn_str = ['O3=O+O2',
                    'O2=O+O',
                    'O+O3=O2+O2',
                    'O2+O2=O+O3',
                    'O+O=O2']
        rcnt = [_.split('=')[0] for _ in rctn_str]
        prdt = [_.split('=')[1] for _ in rctn_str]
        rctn = CoefReactions(reactant=rcnt, product=prdt)
        rctn.rate = np.array([3, 3, 5, 7, 9])
        self.case_0 = Pathways(reactions=rctn)
        # p.rate_prdc_brspc('O')
        # p.set_crrnt_brspc('O')
        # p.update()

    def test_case_0(self):
        cases = []
        cases.append((self.case_0.index_spc('O'), 0))
        cases.append((self.case_0.index_spc('O2'), 1))
        cases.append((self.case_0.index_spc('O3'), 2))
        cases.append((self.case_0.n_spcs, self.case_0.sij.shape[0]))
        cases.append((self.case_0.sij.shape[1], self.case_0.xjk.shape[0]))
        cases.append((self.case_0.xjk.shape[1], self.case_0.mik.shape[1]))
        cases.append((self.case_0.xjk.shape[1], self.case_0.f1k.shape[0]))
        for actual, desired in cases:
            assert actual == desired

    def test_case_1(self):
        cases = []
        cases.append((self.case_0.index_pthwys_cnsm_brspc(spc='O'), np.array([2, 4])))
        cases.append((self.case_0.index_pthwys_cnsm_brspc(spc='O2'), np.array([1, 3])))
        cases.append((self.case_0.index_pthwys_cnsm_brspc(spc='O3'), np.array([0, 2])))
        cases.append((self.case_0.index_pthwys_prdc_brspc(spc='O'), np.array([0, 1, 3])))
        cases.append((self.case_0.index_pthwys_prdc_brspc(spc='O2'), np.array([0, 2, 4])))
        cases.append((self.case_0.index_pthwys_prdc_brspc(spc='O3'), np.array([3])))
        cases.append((self.case_0.index_pthwys_zero_brspc(spc='O'), np.array([])))
        cases.append((self.case_0.index_pthwys_zero_brspc(spc='O2'), np.array([])))
        cases.append((self.case_0.index_pthwys_zero_brspc(spc='O3'), np.array([1, 4])))
        cases.append((self.case_0.spcs_cnsm_rate, np.array([23, 17, 8])))
        cases.append((self.case_0.spcs_prdc_rate, np.array([16, 22, 7])))
        cases.append((self.case_0.spcs_net_rate, np.array([-7, 5, -1])))
        cases.append((self.case_0.Di, np.array([23, 22, 8])))
        for actual, desired in cases:
            assert_allclose(actual, desired)

    def test_update(self):
        self.case_0.set_crrnt_brspc('O')
        temp = self.case_0.connect_two_pathways(1, 4)
        assert_allclose(temp[0].toarray(), np.array([[0, 1, 0, 0, 1]]).transpose())
        assert_allclose(temp[1], self.case_0.f1k[1] * self.case_0.f1k[4] /
                        self.case_0.Di[self.case_0.index_crrnt_brspc] * 2)
        # gcd
