#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9:46 2017/10/12

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm Community Edition
"""

from __future__ import division, print_function, absolute_import
import math
import numpy as np

from scipy.special import erf
from matplotlib import pyplot as plt
from plasmistry.electron import EEDF, get_maxwell_eedf

from plasmistry import constants as const
from numpy.testing import (assert_allclose,
                           TestCase)


# %%--------------------------------------------------------------------------------------------- #
class test_CAL_EEDF(TestCase):
    def setUp(self):
        case_0 = EEDF(max_energy_J=10 * const.eV2J,
                      grid_number=100)
        case_0.density_in_J = get_maxwell_eedf(case_0.energy_point, Te_eV=1.0)
        case_0.pre_set_ee_colli_flux()
        case_0.set_ee_colli_flux()
        self.case_0 = case_0
    
    # %%----------------------------------------------------------------------------------------- #
    def test_get_electron_properties(self):
        eedf = self.case_0
        assert_allclose(eedf.energy_point,
                        np.linspace(0.0, eedf.max_energy_bound, num=eedf.grid_number + 1)[1:] -
                        eedf.energy_intvl / 2)
        assert_allclose(eedf.energy_nodes,
                        np.linspace(0.0, eedf.max_energy_bound, num=eedf.grid_number + 1))
        assert_allclose(eedf.energy_intvl,
                        eedf.max_energy_bound / eedf.grid_number)
        assert_allclose(eedf.electron_temperature(), 1.0 * const.eV2K, rtol=1e-1)
        assert_allclose(eedf.electron_density(), 1.0, rtol=1e-1)
        assert_allclose(eedf.electron_mean_energy(), 1.5 * const.eV2J, rtol=1e-1)
    
    # %%----------------------------------------------------------------------------------------- #
    def test_alpha(self):
        eedf = self.case_0
        alpha_desired = 2 * math.pi / 3 * math.sqrt(2 / const.m_e) * \
                        (const.fine_structure * const.hbar * const.c) ** 2 * \
                        math.log(8 * math.pi / const.e ** 3 *
                                 math.sqrt(2 * (
                                         const.epsilon_0 * eedf.get_electron_mean_energy()) ** 3
                                           / 3 / eedf.get_electron_density()))
        assert_allclose(eedf.ee_alpha, alpha_desired)
    
    def test_P1(self):
        eedf = self.case_0
        Te = 1.0 * const.eV2J
        _energy = eedf.energy_nodes[1:]
        P1_desired = 4 / math.sqrt(math.pi) / np.sqrt(_energy) / Te ** 1.5 * \
                     (-_energy ** (3 / 2) * Te * np.exp(-_energy / Te) -
                      3 / 2 * np.sqrt(_energy) * Te ** 2 * np.exp(-_energy / Te) +
                      3 / 4 * math.sqrt(math.pi) * Te ** (5 / 2) * erf(np.sqrt(_energy / Te)))
        P1_desired = np.hstack((0.0, P1_desired))
        P1_evalued = eedf._op_P1.dot(eedf.density_in_J)
        assert_allclose(P1_evalued[1], P1_desired[1], rtol=3e-2)
        assert_allclose(P1_evalued[2:], P1_desired[2:], rtol=1e-3)
        # plot
        # plt.figure()
        # plt.plot(eedf.energy_nodes, P1_desired)
        # plt.plot(eedf.energy_nodes, P1_evalued, 'o')
        # plt.xscale('log')
        # plt.yscale('log')
    
    # %%----------------------------------------------------------------------------------------- #
    def test_P2(self):
        eedf = self.case_0
        Te = 1.0 * const.eV2J
        _energy = eedf.energy_nodes
        P2_desired = 4 * _energy / math.sqrt(math.pi) * Te ** (-0.5) * np.exp(-_energy / Te)
        P2_evalued = eedf._op_P2.dot(eedf.density_in_J)
        assert_allclose(P2_evalued[1:], P2_desired[1:], rtol=1e-3, atol=1e-12)
        # plot
        # plt.figure()
        # plt.plot(eedf.energy_nodes, P2_desired)
        # plt.plot(eedf.energy_nodes, P2_evalued, 'o')
        # plt.xscale('log')
        # plt.yscale('log')
    
    def test_Q(self):
        eedf = self.case_0
        Te = 1.0 * const.eV2J
        _energy = eedf.energy_nodes[1:]
        Q_desired = 6 / math.sqrt(math.pi) / np.sqrt(_energy) * Te ** (-1.5) * \
                    (-np.sqrt(_energy) * Te * np.exp(-_energy / Te) +
                     math.sqrt(math.pi) * Te ** 1.5 / 2 * erf(np.sqrt(_energy / Te)))
        Q_desired = np.hstack((0.0, Q_desired))
        Q_evalued = eedf._op_Q.dot(eedf.density_in_J)
        assert_allclose(Q_evalued[1:10], Q_desired[1:10], rtol=1e-2)
        assert_allclose(Q_evalued[10:], Q_desired[10:], rtol=2e-3)
        
        # plt.figure()
        # plt.plot(eedf.energy_nodes, Q_desired)
        # plt.plot(eedf.energy_nodes, Q_evalued, 'o')
        # plt.xscale('log')
        # plt.yscale('log')
    
    def test_electric_field_elastic_colli_flux(self):
        eedf = self.case_0
        eedf.set_elastic_crostn(bg_molecule_tuple=('CO', 'N2', 'CO2'))
        # %%------------------------------------------------------------------------------------- #
        eedf.crostn_el = np.vstack((2e-20 * np.ones_like(eedf.energy_nodes),
                                    3e-20 * np.ones_like(eedf.energy_nodes),
                                    5e-20 * np.ones_like(eedf.energy_nodes)))
        eedf.crostn_ef = eedf.crostn_el
        # %%------------------------------------------------------------------------------------- #
        eedf.set_electric_field_flux(electric_field=10.0,
                                     density_total=10.0,
                                     bg_molecule_mole_frac=np.array([0.1, 0.2, 0.7]))
        _const = 1.2678562882894629e-23 * np.sqrt(eedf.energy_nodes[1:-1])
        _variable = 10 / (np.ones_like(eedf.energy_nodes[1:-1]) * 4.3e-20)
        ef_D_k_desired = np.hstack((0.0, _const * _variable / eedf.energy_intvl, 0.0))
        ef_F_k_desired = np.hstack((0.0, _const * _variable / (2 * eedf.energy_nodes[1:-1]), 0.0))
        assert_allclose(ef_D_k_desired, eedf.ef_D_k)
        assert_allclose(ef_F_k_desired, eedf.ef_F_k)
        eedf.set_elastic_colli_flux(Tgas=1e3,
                                    density_total=10.0,
                                    bg_molecule_mole_frac=np.array([0.1, 0.2, 0.7]))
        _const = 3.727113511227553e-38 * eedf.energy_nodes[1:-1] ** 1.5
        _variable = 1e4 * (np.array([0.1, 0.2, 0.7]) / const.atomic_mass / np.array(
                [28.0101, 28.0134, 44.0095])).dot(eedf.crostn_el[:, 1:-1])
        el_D_k_desired = np.hstack((0.0, _const * _variable / eedf.energy_intvl, 0.0))
        el_F_k_desired = np.hstack((0.0, _const * _variable * (.5 / eedf.energy_nodes[1:-1]
                                                               - 1e-3 / const.kB), 0.0))
        assert_allclose(el_D_k_desired, eedf.el_D_k)
        assert_allclose(el_F_k_desired, eedf.el_F_k)
    
