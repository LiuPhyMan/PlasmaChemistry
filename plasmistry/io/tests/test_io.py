#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14:43 2017/7/12

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import
import sys
import numpy as np
import pandas as pd
from plasmistry.io.io_reactions import (instance_reactionList,
                                        instance_reactionFile,
                                        _read_rcnt_prdt_dH_kStr,
                                        _read_reactionlist_block,
                                        read_reactionFile,
                                        read_reactionList)
from plasmistry.io.io_cross_section import read_cross_section_to_frame
from plasmistry.reactions import (Reactions,
                                  MixReactions,
                                  CoefReactions,
                                  CrosReactions)
from numpy.testing import (assert_allclose,
                           assert_array_equal,
                           run_module_suite, )
from pandas.util.testing import assert_series_equal


# %%--------------------------------------------------------------------------------------------- #
def test_read_rcnt_prdt_dH_kStr():
    test_cases = [(' A + B => C + D',
                   'A + B', 'C + D', 0.0, ''),
                  (' 2A + C=> ',
                   '2A + C', '', 0.0, ''),
                  (' A <=> 3a(v1) ! 3a + b',
                   'A', '3a(v1)', 0.0, '3a + b'),
                  ('A + 43_eV=> B + C ! 32.0d3',
                   'A', 'B + C', 43.0, '32.0e3'),
                  ('20_eV=>!abc',
                   '', '', 20.0, 'abc'),
                  ('10_eV=>',
                   '', '', 10.0, '')]
    for case in test_cases:
        assert _read_rcnt_prdt_dH_kStr(case[0]) == case[1:]


# %%--------------------------------------------------------------------------------------------- #
def test_read_reactionlist_block():
    # %%----------------------------------------------------------------------------------------- #
    #   case 1
    # %%----------------------------------------------------------------------------------------- #
    rcntM, prdtM, dHM, k_strM = _read_reactionlist_block(line='A + @B => @C',
                                                         replc_input=['@B = 1 2 3',
                                                                      '@C = a b c'])
    assert rcntM == ['A + 1', 'A + 2', 'A + 3']
    assert prdtM == ['a', 'b', 'c']
    assert dHM == [0.0, 0.0, 0.0]
    assert k_strM == ["", "", ""]
    # %%----------------------------------------------------------------------------------------- #
    #   case 2
    # %%----------------------------------------------------------------------------------------- #
    rcntM, prdtM, dHM, k_strM = _read_reactionlist_block(line='A + @B => @C + @A_eV ! @C -> A ',
                                                         replc_input=['@A = 1 2 3',
                                                                      '@C = a b c',
                                                                      '@B = CO CO2 C'])
    assert rcntM == ['A + CO', 'A + CO2', 'A + C']
    assert prdtM == ['a', 'b', 'c']
    assert dHM == [-1, -2, -3]
    assert k_strM == ["a -> A", "b -> A", "c -> A"]
    # %%----------------------------------------------------------------------------------------- #
    #   case 3
    # %%----------------------------------------------------------------------------------------- #
    rcntM, prdtM, dHM, k_strM = _read_reactionlist_block(line='A + @A@ => @B@ ! @A@ -> A',
                                                         replc_input=['@A@ = a b c',
                                                                      '@B@ = x y z'])
    assert rcntM == ['A + {}'.format(_) for _i in range(3) for _ in 'abc']
    assert prdtM == ['{}'.format(_) for _ in 'xyz' for _i in range(3)]
    assert dHM == [0.0] * 9
    assert k_strM == ['{} -> A'.format(_) for _i in range(3) for _ in 'abc']


# %%--------------------------------------------------------------------------------------------- #
def test_read_reactionfile():
    rcntM, prdtM, dHM, k_strM, pre_exec_list = read_reactionFile(
            sys.path[0] + r'\reactionList_e.g._.inp', start_line=42, end_line=53)
    assert_series_equal(rcntM, pd.Series(['e + N2',
                                          'e + N2',
                                          'e + N2',
                                          'N2(v7) + N2',
                                          'N2(v8) + N2',
                                          'N2     + N2',
                                          'N2(v1) + N2',
                                          'N2(v2) + N2']))
    assert_series_equal(prdtM, pd.Series(['e+N2(v1)',
                                          'e+N2(v1)',
                                          'e+N2(v2)',
                                          'N2(v6)+N2',
                                          'N2(v7) + N2',
                                          'N2(v1) + N2',
                                          'N2(v2) + N2',
                                          'N2(v3) + N2']))
    assert_series_equal(k_strM, pd.Series(['BOLSIG N2 -> N2(v1res)',
                                           'BOLSIG N2 -> N2(v1)',
                                           'BOLSIG N2 -> N2(v2)',
                                           'kVT10_N2N2 * 7.0e0',
                                           'kVT10_N2N2 * 8.0e0',
                                           'kVT01_N2N2',
                                           'kVT01_N2N2 * 2.0e0',
                                           'kVT01_N2N2 * 3.0e0']))
    assert_series_equal(dHM, pd.Series([-0.1,
                                        -0.1,
                                        -0.3,
                                        -200.0,
                                        -200.0,
                                        200.0,
                                        -200.0,
                                        -200.0]))
    assert pre_exec_list == ['QvibO2 = exp(-11605/Tgas)',
                             'kVT10_N2N2 = 4.50d-15 * Tgas',
                             'kVT01_N2N2 = kVT10_N2N2 * QvibO2',
                             'kVT01_N2N2 = kVT01_N2N2 * max(Tgas,3000)']


def test_read_reactionList():
    reaction_list = ["E + {x} => 2B + {y} ! {x} -> {y}".format(x=x, y=y) for x in 'ABC'
                     for y in 'XYZ']
    rcntM, prdtM, dHM, k_strM = read_reactionList(reaction_list)
    assert_series_equal(rcntM, pd.Series(['E + A', 'E + A', 'E + A',
                                          'E + B', 'E + B', 'E + B',
                                          'E + C', 'E + C', 'E + C']))
    assert_series_equal(prdtM, pd.Series(['2B + X', '2B + Y', '2B + Z',
                                          '2B + X', '2B + Y', '2B + Z',
                                          '2B + X', '2B + Y', '2B + Z']))
    assert_series_equal(dHM, pd.Series([0.0] * 9))
    assert_series_equal(k_strM,
                        pd.Series(['{x} -> {y}'.format(x=x, y=y) for x in 'ABC' for y in 'XYZ']))


# %%--------------------------------------------------------------------------------------------- #
def test_instance_reactionsFile_0():
    file_path = sys.path[0] + r"\reactionList_e.g._.inp"
    a = instance_reactionFile(file_path=file_path,
                              class_name=CoefReactions,
                              start_line=21,
                              end_line=22)
    assert_series_equal(a.species,
                        pd.Series(['C',
                                   'CO',
                                   'CO(C3)',
                                   'CO(V1)',
                                   'CO2',
                                   'CO2(V1)']))
    assert_series_equal(a.reactant,
                        pd.Series(['CO + CO2',
                                   'C + CO + CO(V1)']))
    assert_series_equal(a.product,
                        pd.Series(['CO2(V1) + CO(C3)',
                                   'CO + C']))
    assert_array_equal(a._Reactions__sij.toarray(), np.array([[0, 0],
                                                              [-1, 0],
                                                              [1, 0],
                                                              [0, -1],
                                                              [-1, 0],
                                                              [1, 0]]))
    assert_allclose(a._Reactions__rcnt_index, np.array([[1, 4, -1],
                                                        [0, 1, 3]]))
    assert_allclose(a._Reactions__rcnt_expnt, np.array([[1, 1, 0],
                                                        [1, 1, 1]]))


# %%--------------------------------------------------------------------------------------------- #
def test_instance_reactionsFile_1():
    file_path = sys.path[0] + r"\reactionList_e.g._.inp"
    a = instance_reactionFile(file_path=file_path,
                              class_name=CoefReactions,
                              start_line=1,
                              end_line=9)
    assert_series_equal(a.species,
                        pd.Series(['A', 'B', 'B^+', 'C', 'CO2(v32)', 'D', 'c', 'd(fd)']))
    assert_series_equal(a.reactant,
                        pd.Series(['2A + 3B',
                                   '2B',
                                   '',
                                   '',
                                   '2A',
                                   'A',
                                   'A + B^+ + c + d(fd)',
                                   'C',
                                   '']))
    assert_series_equal(a.product,
                        pd.Series(['C + D',
                                   '3C + A',
                                   '2D + 4A',
                                   '3D',
                                   '',
                                   '2C + D',
                                   '2C + CO2(v32)',
                                   '',
                                   'D']))
    assert_array_equal(a._Reactions__sij.toarray(),
                       np.array([[-2, 1, 4, 0, -2, -1, -1, 0, 0],
                                 [-3, -2, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, -1, 0, 0],
                                 [1, 3, 0, 0, 0, 2, 2, -1, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                 [1, 0, 2, 3, 0, 1, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0, -1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, -1, 0, 0]]))
    assert_array_equal(a._Reactions__rcnt_index,
                       np.array([[0, 1, -1, -1],
                                 [1, -1, -1, -1],
                                 [-1, -1, -1, -1],
                                 [-1, -1, -1, -1],
                                 [0, -1, -1, -1],
                                 [0, -1, -1, -1],
                                 [0, 2, 6, 7],
                                 [3, -1, -1, -1],
                                 [-1, -1, -1, -1]]))
    assert_array_equal(a._Reactions__rcnt_expnt,
                       np.array([[2, 3, 0, 0],
                                 [2, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [2, 0, 0, 0],
                                 [1, 0, 0, 0],
                                 [1, 1, 1, 1],
                                 [1, 0, 0, 0],
                                 [0, 0, 0, 0]]))


# %%--------------------------------------------------------------------------------------------- #
def test_instance_reactionsFile_2():
    file_path = sys.path[0] + r"\reactionList_e.g._.inp"
    a = instance_reactionFile(file_path=file_path,
                              class_name=CoefReactions,
                              start_line=11,
                              end_line=14)
    assert_series_equal(a.species,
                        pd.Series(['C',
                                   'CO2',
                                   'H',
                                   'H2O',
                                   'N',
                                   'N2',
                                   'O',
                                   'O2']))
    assert_series_equal(a.reactant,
                        pd.Series(['CO2 + 3CO2',
                                   'CO2 + 3H2O',
                                   'CO2 + 3N2',
                                   'CO2 + 3O2']))
    assert_series_equal(a.product,
                        pd.Series(['2C + O',
                                   '3H + O',
                                   '2N + O',
                                   '1O + O']))
    assert_array_equal(a._Reactions__sij.toarray(),
                       np.array([[2, 0, 0, 0],
                                 [-4, -1, -1, -1],
                                 [0, 3, 0, 0],
                                 [0, -3, 0, 0],
                                 [0, 0, 2, 0],
                                 [0, 0, -3, 0],
                                 [1, 1, 1, 2],
                                 [0, 0, 0, -3]]))
    assert_array_equal(a._Reactions__rcnt_index,
                       np.array([[1, -1],
                                 [1, 3],
                                 [1, 5],
                                 [1, 7]]))
    assert_array_equal(a._Reactions__rcnt_expnt,
                       np.array([[4, 0],
                                 [1, 3],
                                 [1, 3],
                                 [1, 3]]))


# %%--------------------------------------------------------------------------------------------- #
def test_instance_reactionsFile_3():
    file_path = sys.path[0] + r"\reactionList_e.g._.inp"
    a = instance_reactionFile(file_path=file_path,
                              class_name=CoefReactions,
                              start_line=16,
                              end_line=18)
    assert_series_equal(a.species,
                        pd.Series(['E',
                                   'H2(Va)',
                                   'H2(Vb)',
                                   'H2(Vc)',
                                   'H2O(V1)',
                                   'H2O(V2)',
                                   'H2O(V3)']))
    assert_series_equal(a.reactant,
                        pd.Series(['E + H2(Va)',
                                   'E + H2(Vb)',
                                   'E + H2(Vc)',
                                   'E + H2(Va)',
                                   'E + H2(Vb)',
                                   'E + H2(Vc)',
                                   'E + H2(Va)',
                                   'E + H2(Vb)',
                                   'E + H2(Vc)']))
    assert_series_equal(a.product,
                        pd.Series(['E + H2O(V1)',
                                   'E + H2O(V1)',
                                   'E + H2O(V1)',
                                   'E + H2O(V2)',
                                   'E + H2O(V2)',
                                   'E + H2O(V2)',
                                   'E + H2O(V3)',
                                   'E + H2O(V3)',
                                   'E + H2O(V3)']))
    assert_array_equal(a._Reactions__sij.toarray(),
                       np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [-1, 0, 0, -1, 0, 0, -1, 0, 0],
                                 [0, -1, 0, 0, -1, 0, 0, -1, 0],
                                 [0, 0, -1, 0, 0, -1, 0, 0, -1],
                                 [1, 1, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 1, 1]]))
    assert_array_equal(a._Reactions__rcnt_index,
                       np.array([[0, 1],
                                 [0, 2],
                                 [0, 3],
                                 [0, 1],
                                 [0, 2],
                                 [0, 3],
                                 [0, 1],
                                 [0, 2],
                                 [0, 3]]))
    assert_array_equal(a._Reactions__rcnt_expnt,
                       np.array([[1, 1],
                                 [1, 1],
                                 [1, 1],
                                 [1, 1],
                                 [1, 1],
                                 [1, 1],
                                 [1, 1],
                                 [1, 1],
                                 [1, 1]]))


# %%--------------------------------------------------------------------------------------------- #
def test_instance_reactionsFile_4():
    file_path = sys.path[0] + r"\reactionList_e.g._.inp"
    a = instance_reactionFile(file_path=file_path,
                              class_name=MixReactions,
                              start_line=41,
                              end_line=53)
    cros_reactions = a['cros_reactions']
    coef_reactions = a['coef_reactions']
    # %%----------------------------------------------------------------------------------------- #
    assert cros_reactions.reaction_type == 'cross_sections related'
    assert (cros_reactions.n_species, cros_reactions.n_reactions) == (4, 3)
    assert_series_equal(cros_reactions.reactant,
                        pd.Series(['e + N2', 'e + N2', 'e + N2']))
    assert_series_equal(cros_reactions.product,
                        pd.Series(['e + N2(v1)', 'e + N2(v1)', 'e + N2(v2)']))
    assert_series_equal(cros_reactions.k_str,
                        pd.Series(['N2->N2(v1res)',
                                   'N2->N2(v1)',
                                   'N2->N2(v2)']))
    assert_series_equal(cros_reactions.dH_e,
                        pd.Series([-0.1, -0.1, -0.3]))
    assert_series_equal(cros_reactions.dH_g,
                        pd.Series([0.0, 0.0, 0.0]))

    # %%----------------------------------------------------------------------------------------- #
    assert coef_reactions.reaction_type == 'k_coefficients related'
    assert (coef_reactions.n_species, coef_reactions.n_reactions) == (7, 5)
    assert_series_equal(coef_reactions.reactant,
                        pd.Series(['N2(v7) + N2',
                                   'N2(v8) + N2',
                                   'N2 + N2',
                                   'N2(v1) + N2',
                                   'N2(v2) + N2']))
    assert_series_equal(coef_reactions.product,
                        pd.Series(['N2(v6) + N2',
                                   'N2(v7) + N2',
                                   'N2(v1) + N2',
                                   'N2(v2) + N2',
                                   'N2(v3) + N2']))
    assert_series_equal(coef_reactions.k_str,
                        pd.Series(['kVT10_N2N2*7.0e0',
                                   'kVT10_N2N2*8.0e0',
                                   'kVT01_N2N2',
                                   'kVT01_N2N2*2.0e0',
                                   'kVT01_N2N2*3.0e0']))
    assert_series_equal(coef_reactions.dH_e,
                        pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]))
    assert_series_equal(coef_reactions.dH_g,
                        pd.Series([-200.0, -200.0, 200.0, -200.0, -200.0]))
    assert coef_reactions.pre_exec_list == ['QvibO2 = exp(-11605/Tgas)',
                                            'kVT10_N2N2 = 4.50e-15 * Tgas',
                                            'kVT01_N2N2 = kVT10_N2N2 * QvibO2',
                                            'kVT01_N2N2 = kVT01_N2N2 * max(Tgas,3000)']


# %%--------------------------------------------------------------------------------------------- #
def test_instance_reactionsList_0():
    reactions = ['e + N2 + {E}_eV => e + N2(v{v}) ! N2->N2(v{v})'.format(v=v, E=E)
                 for v, E in zip(range(1, 5), [0.1, 0.2, 0.3, 0.4])]
    a = instance_reactionList(reaction_list=reactions, class_name=CoefReactions)
    assert_series_equal(a.reactant, pd.Series(['e + N2'] * 4))
    assert_series_equal(a.product,
                        pd.Series(['e + N2(v1)', 'e + N2(v2)', 'e + N2(v3)', 'e + N2(v4)']))
    assert_series_equal(a.dH_g, pd.Series([0.1, 0.2, 0.3, 0.4]))
    assert_series_equal(a.dH_e, pd.Series([0.0, 0.0, 0.0, 0.0]))
    assert_series_equal(a.k_str,
                        pd.Series(['N2->N2(v1)', 'N2->N2(v2)', 'N2->N2(v3)', 'N2->N2(v4)']))


# %%--------------------------------------------------------------------------------------------- #
def test_instance_reactionsList_1():
    reactions = ['e + N2 + {E}_eV => e + N2(v{v}) ! BOLSIG N2 -> N2(v{v})'.format(v=v, E=E)
                 for v, E in zip(range(1, 5), [0.1, 0.2, 0.3, 0.4])]
    a = instance_reactionList(reaction_list=reactions, class_name=CrosReactions)
    assert_series_equal(a.reactant, pd.Series(['e + N2'] * 4))
    assert_series_equal(a.product,
                        pd.Series(['e + N2(v1)', 'e + N2(v2)', 'e + N2(v3)', 'e + N2(v4)']))
    assert_series_equal(a.dH_e, pd.Series([0.1, 0.2, 0.3, 0.4]))
    assert_series_equal(a.dH_g, pd.Series([0.0, 0.0, 0.0, 0.0]))
    assert_series_equal(a.k_str,
                        pd.Series(['N2->N2(v1)', 'N2->N2(v2)', 'N2->N2(v3)', 'N2->N2(v4)']))


# %%--------------------------------------------------------------------------------------------- #
def test_read_cross_section_to_frame():
    cs_file_path = sys.path[0] + r"\cross_section_e.g._.dat"
    a = read_cross_section_to_frame(cs_file_path)
    b = a.loc['TEST_CS']
    assert b['type'] == 'IONIZATION'
    assert b.name == 'TEST_CS'
    assert b['info_dict'] == {'SPECIES': 'e / N2',
                              'PROCESS': 'E + N2 -> E + E + N2+, Ionization',
                              'PARAM.': 'E = 15.6 eV, complete set',
                              'UPDATED': '2010-03-02 08:39:45',
                              'COLUMNS': 'Energy (eV) | Cross section (m2)'}
    assert_allclose(np.array([1.56e+1, 1.6e+1, 2.0e+1, 2.1e+1]),
                    b['energy'])
    assert_allclose(np.array([0.0e+0, 2.1e-22, 2.2e-21, 0.0e+0]),
                    b['crostn'])
    assert b['energy_range'] == (15.6, 21.0)


# %%--------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_module_suite()
