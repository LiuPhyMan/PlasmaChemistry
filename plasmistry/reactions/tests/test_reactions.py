#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14:46 2017/7/10

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import
import sys
import math
import re
import numpy as np
import pandas as pd

from scipy.integrate import odeint
from numpy.testing import (assert_allclose,
                           run_module_suite,
                           TestCase)
from pandas.util.testing import assert_series_equal
from scipy.integrate import simps
from plasmistry import constants as const
from plasmistry.electron import get_maxwell_eedf
from plasmistry.reactions import (Reactions,
                                  CrosReactions,
                                  CoefReactions,
                                  MixReactions,
                                  get_reverse_crostn)
from plasmistry.io import (read_cross_section_to_frame,
                           instance_reactionList)


# %%--------------------------------------------------------------------------------------------- #
def test_Reactions_basic():
    r"""
    Check :
        fortran2python
        format_cmpnds
        _regexp_check
        specie_regexp
        cmpnds_regexp

    """
    cases = []
    cases.append(([Reactions.fortran2python(_) for _ in ('1d2', '1.0d-2', '-3.d+2')],
                  ['1e2', '1.0e-2', '-3.e+2']))
    cases.append((Reactions.format_cmpnds(
            pd.Series(['A^+ + b+a  ', '2A()+3C', 'OH  +  OH', '3A^+D+D'])).tolist(),
                  ['A^+ + b + a', '2A() + 3C', 'OH + OH', '3A^+D + D']))
    for actual, desired in cases:
        assert actual == desired
    assert Reactions._regexp_check(pd.Series(['A', 'B(V2)', 'CH3OH(3H)^+']),
                                   Reactions.specie_regexp)
    assert Reactions._regexp_check(pd.Series(['A + B', '', 'C + C(20)^+', 'W^+D + 2W + C']),
                                   Reactions.cmpnds_regexp)


# %%--------------------------------------------------------------------------------------------- #
class test_Reactions(TestCase):
    def setUp(self):
        reactions = ['2A + 3B =>C+D',
                     '2B =>3C+A',
                     '=>2D+3A',
                     'A =>2C+D',
                     ' C =>']
        reactant = [re.split(r"=>", _)[0] for _ in reactions]
        product = [re.split(r"=>", _)[1] for _ in reactions]
        self.case_0 = Reactions(reactant=pd.Series(reactant),
                                product=pd.Series(product),
                                k_str=pd.Series(['Tgas*2',
                                                 'Te',
                                                 'EN',
                                                 '2.0d2',
                                                 '2.0']))
        # %%------------------------------------------------------------------------------------- #
        replc_A = ['N2^+', 'O2^+', 'O4^+', 'NO^+', 'NO2^+', 'O2^+N2']
        replc_B = ['N+N', 'O+O', 'O2+O2', 'N+O', 'N+O2', 'O2+N2']
        reactions = ["O^- + {A} => O + {B}".format(A=A, B=B) for A, B in zip(replc_A, replc_B)]
        reactant = [re.split(r"=>", _)[0] for _ in reactions]
        product = [re.split(r"=>", _)[1] for _ in reactions]
        self.case_1 = Reactions(reactant=pd.Series(reactant),
                                product=pd.Series(product),
                                k_str=pd.Series(['kVT01_N2N2 * 1.0d0',
                                                 'kVT01_N2O * 2.0d0',
                                                 '3.0d-2',
                                                 '2.0d-7 * (300.0d0/Te)**0.5',
                                                 '2.7d-10 * (Tef0d0)**0.5 * exp(-5590.0d0/TeffN2)',
                                                 'max( 220.0d0/Tgas) , 1.0d-33 * (Tgas)**0.41 )']))
        # %%------------------------------------------------------------------------------------- #
        reactions = ['E + CO => E + CO(V{i})'.format(i=_) for _ in range(1, 5)]
        reactant = pd.Series([_.split(sep='=>')[0] for _ in reactions])
        product = pd.Series([_.split(sep='=>')[1] for _ in reactions])
        self.case_2 = Reactions(reactant=reactant,
                                product=product,
                                k_str=pd.Series(['4', '3', '2', '1']))

    # %%----------------------------------------------------------------------------------------- #
    def test_case_0(self):
        r"""
        Check :
            __init__
            _get_sparse_paras
            _get_rcnt_index_expnt

        """
        cases = []
        cases.append((self.case_0.n_species, 4))
        cases.append((self.case_0.n_reactions, 5))
        cases.append((self.case_0.species.tolist(), ['A', 'B', 'C', 'D']))
        cases.append((self.case_0.reactant.tolist(), ['2A + 3B', '2B', '', 'A', 'C']))
        cases.append((self.case_0.product.tolist(), ['C + D', '3C + A', '2D + 3A', '2C + D', '']))
        cases.append((self.case_0._Reactions__sij.toarray().tolist(), [[-2, 1, 3, -1, 0],
                                                                       [-3, -2, 0, 0, 0],
                                                                       [1, 3, 0, 2, -1],
                                                                       [1, 0, 2, 1, 0]]))
        cases.append((self.case_0._Reactions__rcntsij.toarray().tolist(), [[2, 0, 0, 1, 0],
                                                                           [3, 2, 0, 0, 0],
                                                                           [0, 0, 0, 0, 1],
                                                                           [0, 0, 0, 0, 0]]))
        cases.append((self.case_0._Reactions__prdtsij.toarray().tolist(), [[0, 1, 3, 0, 0],
                                                                           [0, 0, 0, 0, 0],
                                                                           [1, 3, 0, 2, 0],
                                                                           [1, 0, 2, 1, 0]]))
        cases.append((self.case_0._Reactions__rcnt_index.tolist(), [[0, 1],
                                                                    [1, -1],
                                                                    [-1, -1],
                                                                    [0, -1],
                                                                    [2, -1]]))
        cases.append((self.case_0._Reactions__rcnt_expnt.tolist(), [[2, 3],
                                                                    [2, 0],
                                                                    [0, 0],
                                                                    [1, 0],
                                                                    [1, 0]]))
        # %%------------------------------------------------------------------------------------- #
        for actual, desired in cases:
            assert actual == desired

    # %%----------------------------------------------------------------------------------------- #
    def test_case_1(self):
        cases = []
        cases.append((self.case_1.species.tolist(), ['N', 'N2', 'N2^+', 'NO2^+', 'NO^+', 'O',
                                                     'O2', 'O2^+', 'O2^+N2', 'O4^+', 'O^-']))
        cases.append((self.case_1.reactant.tolist(), ['O^- + N2^+',
                                                      'O^- + O2^+',
                                                      'O^- + O4^+',
                                                      'O^- + NO^+',
                                                      'O^- + NO2^+',
                                                      'O^- + O2^+N2']))
        cases.append((self.case_1.product.tolist(), ['O + N + N',
                                                     'O + O + O',
                                                     'O + O2 + O2',
                                                     'O + N + O',
                                                     'O + N + O2',
                                                     'O + O2 + N2'])),
        cases.append((self.case_1._Reactions__rcntsij.toarray().tolist(), [[0, 0, 0, 0, 0, 0],
                                                                           [0, 0, 0, 0, 0, 0],
                                                                           [1, 0, 0, 0, 0, 0],
                                                                           [0, 0, 0, 0, 1, 0],
                                                                           [0, 0, 0, 1, 0, 0],
                                                                           [0, 0, 0, 0, 0, 0],
                                                                           [0, 0, 0, 0, 0, 0],
                                                                           [0, 1, 0, 0, 0, 0],
                                                                           [0, 0, 0, 0, 0, 1],
                                                                           [0, 0, 1, 0, 0, 0],
                                                                           [1, 1, 1, 1, 1, 1]]))
        cases.append((self.case_1._Reactions__prdtsij.toarray().tolist(), [[2, 0, 0, 1, 1, 0],
                                                                           [0, 0, 0, 0, 0, 1],
                                                                           [0, 0, 0, 0, 0, 0],
                                                                           [0, 0, 0, 0, 0, 0],
                                                                           [0, 0, 0, 0, 0, 0],
                                                                           [1, 3, 1, 2, 1, 1],
                                                                           [0, 0, 2, 0, 1, 1],
                                                                           [0, 0, 0, 0, 0, 0],
                                                                           [0, 0, 0, 0, 0, 0],
                                                                           [0, 0, 0, 0, 0, 0],
                                                                           [0, 0, 0, 0, 0, 0]]))
        cases.append((self.case_1._Reactions__rcnt_index.tolist(), [[2, 10],
                                                                    [7, 10],
                                                                    [9, 10],
                                                                    [4, 10],
                                                                    [3, 10],
                                                                    [8, 10]]))
        cases.append((self.case_1._Reactions__rcnt_expnt.tolist(), [[1, 1],
                                                                    [1, 1],
                                                                    [1, 1],
                                                                    [1, 1],
                                                                    [1, 1],
                                                                    [1, 1]]))
        for actual, desired in cases:
            assert actual == desired

    # %%----------------------------------------------------------------------------------------- #
    def test_case_2(self):
        r"""
        Check :
            set_rate
            get_dn
            get_dH_e
            get_dH_g
            get_initial_density
            time_evolution

        """
        a = self.case_2
        density_0 = a.get_initial_density(density_dict={'CO': 4.0, 'E': 1.0})
        a.rate_const = np.array([4, 3, 2, 1])
        a.set_rate(density=density_0)

        # %%------------------------------------------------------------------------------------- #
        def deriv_func(t, y, rctn_instance):
            rctn_instance.set_rate(density=y)
            return rctn_instance.get_dn()

        output = a.time_evolution(deriv_func=deriv_func,
                                  y_0=density_0,
                                  time_span=(0.0, 10.0),
                                  rtol=1e-6,
                                  atol=1e-6,
                                  output_index=a.species,
                                  solver_args=(a,))
        time_seq = output.index.tolist()
        exact_values_dict = {'CO': 4 * np.exp(-10 * np.array(time_seq)),
                             'CO(V1)': -1.6 * np.exp(-10 * np.array(time_seq)) + 1.6,
                             'CO(V2)': -1.2 * np.exp(-10 * np.array(time_seq)) + 1.2,
                             'CO(V3)': -0.8 * np.exp(-10 * np.array(time_seq)) + 0.8,
                             'CO(V4)': -0.4 * np.exp(-10 * np.array(time_seq)) + 0.4,
                             'E': np.ones_like(time_seq)}
        for _ in a.species:
            assert_allclose(output[_].values + 1, exact_values_dict[_] + 1, atol=1e-6, rtol=1e-4)


# %%--------------------------------------------------------------------------------------------- #
class test_CrosReactions(TestCase):
    def setUp(self):
        reactions = ['E + CO => E + CO(V{v}) + {E}_eV! BOLSIG CO -> CO(V{v})'.format(v=_, E=_E)
                     for _, _E in zip(range(1, 5), (0.5, 1.0, 1.5, 2.0))]
        self.case_0 = instance_reactionList(reaction_list=reactions, class_name=CrosReactions)
        cs_frame = pd.DataFrame({'type': 'EXCITATION',
                                 'energy': [np.array([0.9, 1.0, 2.0, 3.0, 4.0]) + i for i in
                                            range(4)],
                                 'crostn': [np.array([0.0] + [i] * 4) for i in range(4)],
                                 'thres_info': ['1', '2', '3', '4']},
                                index=['CO->CO(V{v})'.format(v=_) for _ in range(1, 5)])
        self.case_0.set_cross_sections(cs_frame=cs_frame,
                                       electron_energy_grid=np.linspace(0.0, 10, num=101)[
                                                            1:] * const.eV2J)
        # %%------------------------------------------------------------------------------------- #
        reactions = ['E + CO => E + CO(V1) + 0.1_eV     !   kVT01*1.0d0',
                     'E + CO => E + CO(V2) + 0.2d0_eV   !   kVT02*2.0d0',
                     'E + CO(V1) => E + CO + 0.4e1_eV   !   kVT10*1.0d0',
                     'E + CO(V2) => E + CO + 1e-1_eV    !   kVT20*2.0d0']
        self.case_1 = instance_reactionList(reaction_list=reactions, class_name=CoefReactions)
        self.case_1.set_pre_exec_list(['QvibCO = exp(-11605/Tgas)*0.0 + max(Tgas,100)*0.0 + 2.0',
                                       'kVT10 = 7.8d-12*Tgas/(1.0-QvibCO) * 0.0 + 1.0',
                                       'kVT01 = kVT10 * QvibCO',
                                       'kVT20 = 1.20d-13 * exp(-30.0/Tgas**(1.0/3.0)) * 0.0 + 2.0',
                                       'kVT02 = kVT20 * QvibCO'])
        self.case_1.compile_k_str()

    def test_case_0(self):
        cases = []
        cases.append((self.case_0.species.tolist(),
                      ['CO', 'CO(V1)', 'CO(V2)', 'CO(V3)', 'CO(V4)', 'E']))
        cases.append((self.case_0.reactant.tolist(),
                      ['E + CO'] * 4))
        cases.append((self.case_0.product.tolist(),
                      ['E + CO(V1)', 'E + CO(V2)', 'E + CO(V3)', 'E + CO(V4)']))
        cases.append((self.case_0.dH_g.tolist(),
                      [0.0] * 4))
        cases.append((self.case_0.dH_e.tolist(),
                      [-0.5, -1.0, -1.5, -2.0]))
        cases.append((self.case_0.k_str.tolist(),
                      ['CO->CO(V1)', 'CO->CO(V2)', 'CO->CO(V3)', 'CO->CO(V4)']))
        for actual, desired in cases:
            assert actual == desired
        # %%------------------------------------------------------------------------------------- #
        #   cs_type, cs_thres, cs_electron_energy_grid, cs_crostn
        # %%------------------------------------------------------------------------------------- #
        assert_series_equal(self.case_0.cs_type,
                            pd.Series(['EXCITATION'] * 4))
        assert_series_equal(self.case_0.cs_thres,
                            pd.Series(['1', '2', '3', '4']))
        assert_allclose(self.case_0.cs_electron_energy_grid,
                        np.linspace(0, 10, num=101)[1:] * const.eV2J)
        # %%------------------------------------------------------------------------------------- #
        #   set_rate_const
        # %%------------------------------------------------------------------------------------- #
        self.case_0.set_rate_const(
                eedf=np.sqrt(const.m_e / 2) / np.sqrt(self.case_0.cs_electron_energy_grid))
        assert_allclose(self.case_0.rate_const * const.J2eV,
                        np.array([(10 - 1) * 0,
                                  (10 - 2) * 1,
                                  (10 - 3) * 2,
                                  (10 - 4) * 3]), rtol=1e-2)
        self.case_0.set_rate_const(
                eedf=np.sqrt(const.m_e / 2) * np.sqrt(self.case_0.cs_electron_energy_grid))
        assert_allclose(self.case_0.rate_const * const.J2eV ** 2,
                        np.array([0 / 2 * (10 ** 2 - 1 ** 2),
                                  1 / 2 * (10 ** 2 - 2 ** 2),
                                  2 / 2 * (10 ** 2 - 3 ** 2),
                                  3 / 2 * (10 ** 2 - 4 ** 2)]), rtol=1e-2)

    def test_case_1(self):
        r"""
        Check :
            __init__
            compile_k_str
            set_pre_exec_list
            set_rate_const

        """
        a = self.case_1
        density_0 = a.get_initial_density(density_dict={'CO': 10.0, 'E': 1.0})

        def deriv_func(t, y, rctn_instance):
            rctn_instance.set_rate_const(Tgas_K=1e3, Te_eV=1.0, EN_Td=1.0)
            rctn_instance.set_rate(density=y)
            return rctn_instance.get_dn()

        output = a.time_evolution(deriv_func=deriv_func,
                                  y_0=density_0,
                                  time_span=(0.0, 10.0),
                                  rtol=1e-6,
                                  atol=1e-6,
                                  output_index=a.species,
                                  solver_args=(a,))
        time_seq = np.array(output.index)

        # %%------------------------------------------------------------------------------------- #
        #   Exact values
        # %%------------------------------------------------------------------------------------- #
        def dy(y, t):
            n0, n1, n2 = y
            dydt = [n1 + 4 * n2 - 10 * n0, 2 * n0 - n1, 8 * n0 - 4 * n2]
            return dydt

        y0 = [10, 0, 0]
        exact_values = odeint(dy, y0, time_seq)
        # %%------------------------------------------------------------------------------------- #
        #   assert
        # %%------------------------------------------------------------------------------------- #
        assert_allclose(output['CO'].values, exact_values[:, 0], atol=1e-6, rtol=1e-4)
        assert_allclose(output['CO(V1)'].values, exact_values[:, 1], atol=1e-6, rtol=1e-4)
        assert_allclose(output['CO(V2)'].values, exact_values[:, 2], atol=1e-6, rtol=1e-4)
        assert a.mid_variables == {'EN': 1.0, 'QvibCO': 2.0, 'Te': 1.0, 'Tgas': 1e3, 'kVT01': 2.0,
                                   'kVT02': 4.0, 'kVT10': 1.0, 'kVT20': 2.0}
        assert_allclose(a.rate, np.array([4.0, 16.0, 4.0, 16.0]), atol=1e-4, rtol=1e-4)

    def test_case_2(self):
        a = MixReactions(cros_instance=self.case_0, coef_instance=self.case_1)
        cases = []
        cases.append((a.species.tolist(),
                      ['CO', 'CO(V1)', 'CO(V2)', 'CO(V3)', 'CO(V4)', 'E']))
        cases.append((a.reactant.tolist(),
                      ['E + CO'] * 6 + ['E + CO(V1)', 'E + CO(V2)']))
        cases.append((a.product.tolist(),
                      ['E + CO(V1)', 'E + CO(V2)', 'E + CO(V3)', 'E + CO(V4)',
                       'E + CO(V1)', 'E + CO(V2)', 'E + CO', 'E + CO']))
        cases.append((a.dH_g.tolist(),
                      [0.0] * 4 + [-0.1, -0.2, -4.0, -0.1]))
        cases.append((a.dH_e.tolist(),
                      [-0.5, -1.0, -1.5, -2.0] + [0.0] * 4))
        cases.append((a.k_str.tolist(),
                      ['CO->CO(V1)', 'CO->CO(V2)', 'CO->CO(V3)', 'CO->CO(V4)',
                       'kVT01*1.0e0', 'kVT02*2.0e0', 'kVT10*1.0e0', 'kVT20*2.0e0']))
        a.set_rate_const(
                eedf=np.sqrt(const.m_e / 2) / np.sqrt(
                        self.case_0.cs_electron_energy_grid) * const.J2eV,
                Tgas_K=1000.0,
                Te_eV=1.0,
                EN_Td=1.0)
        assert_allclose(a.rate_const,
                        np.array([0, 8.05, 14.10, 18.15, 2, 8, 1, 4]), rtol=1e-2)
        a.set_rate(
                density=a.get_initial_density(density_dict={'E': 1.0, 'CO': 2.0, 'CO(V1)': 3.0}))
        assert_allclose(a.get_dn(), np.array([-97.6, 1.0, 32.1, 28.2, 36.3, 0.0]))
        assert a.get_dH_g() == -15.6
        assert a.get_dH_e() == -131.0


# %%--------------------------------------------------------------------------------------------- #
class Test_get_reverse_crostn(TestCase):
    def setUp(self):
        cs_file_path = sys.path[0] + r"\cross_section_e.g._dat"
        a = read_cross_section_to_frame(cs_file_path)
        self.case_0 = get_reverse_crostn(a.loc['N2->N2(V1)'], stcs_wght_ratio_dict={'N2': 2})
        cs_file_path = sys.path[0] + r"\cross_section_e.g._dat"
        self.case_1 = read_cross_section_to_frame(cs_file_path).loc['CO2->CO2(Va)']

    # %%----------------------------------------------------------------------------------------- #
    def test_case_0(self):
        cases = []
        cases.append((self.case_0['type'], 'EXCITATION'))
        cases.append((self.case_0.name, 'N2(V1)->N2'))
        cases.append((self.case_0['thres_info'], '-1.00000e+0'))
        for actual, desired in cases:
            assert actual == desired
        assert_allclose(self.case_0['energy'], np.array([0.1, 1.0, 2.0, 3.0]))
        assert_allclose(self.case_0['crostn'], np.array([22.0, 4.0, 3.0, 8 / 3]))

    # %%----------------------------------------------------------------------------------------- #
    def test_case_1(self):
        a = self.case_1
        Te_eV = 0.5
        eedf_forward = get_maxwell_eedf(electron_energy_grid=a['energy'] * const.eV2J, Te_eV=Te_eV)

        def get_k(_energy, _crostn, _eedf):
            y = np.sqrt(_energy) * _crostn * _eedf
            return math.sqrt(2 / const.m_e) * simps(y=y, x=_energy)

        b = get_reverse_crostn(a, stcs_wght_ratio_dict={'N2': 2})
        eedf_backward = get_maxwell_eedf(electron_energy_grid=b['energy'] * const.eV2J,
                                         Te_eV=Te_eV)
        k_forward = get_k(a['energy'], a['crostn'], eedf_forward)
        k_backward = get_k(b['energy'], b['crostn'], eedf_backward)

        assert_allclose(-math.log(k_forward / k_backward) * Te_eV, 0.083, rtol=1e-1, atol=0)
