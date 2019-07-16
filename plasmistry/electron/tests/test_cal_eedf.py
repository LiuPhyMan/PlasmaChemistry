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
from copy import deepcopy
import sys
import math
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from plasmistry.electron import EEDF, get_maxwell_eedf
from plasmistry.solvers import ode_ivp
from plasmistry import constants as const
from numpy.testing import (assert_allclose, TestCase)


# %%--------------------------------------------------------------------------------------------- #
class test_CAL_EEDF(TestCase):
    def setUp(self):
        # case 0
        case_0 = EEDF(max_energy_J=10 * const.eV2J, grid_number=100)
        case_0.density_in_J = get_maxwell_eedf(case_0.energy_point, Te_eV=1.0)
        case_0._pre_set_flux_ee_colli()
        case_0._set_flux_ee_colli()
        self.case_0 = case_0
        # case 1
        case_1 = EEDF(max_energy_J=10 * const.eV2J, grid_number=100)
        case_1.density_in_J = get_maxwell_eedf(case_1.energy_point, Te_eV=1.0)
        species = ['CO2', 'CO2(v0)', 'CO2(v1)', 'CO', 'O2']
        case_1._set_crostn_elastic(total_species=species)
        _inelas_dataframe = pd.read_pickle(sys.path[0] + r'\e.g._inelas_colli_dataframe.pkl')
        case_1._set_crostn_inelastic(inelas_reaction_dataframe=_inelas_dataframe)
        case_1._set_index_bg_molecule(total_species=species)
        self.case_1 = case_1
        # case 2
        case_2 = EEDF(max_energy_J=10 * const.eV2J, grid_number=100)
        case_2.density_in_J = get_maxwell_eedf(case_1.energy_point, Te_eV=1.0)
        species = ['CO2', 'CO2(v0)', 'CO2(v1)', 'CO2(vc)', 'CO', 'O2']
        case_2._set_crostn_elastic(total_species=species)
        case_3 = deepcopy(case_2)
        case_4 = deepcopy(case_2)
        case_5 = deepcopy(case_2)

        case_2._set_crostn_inelastic(inelas_reaction_dataframe=pd.read_pickle(
            sys.path[0] + r'\e.g._cs_CO2(v0)_vibexc_CO2(v1)_0.291.pkl'))
        case_2._set_index_bg_molecule(total_species=species)

        case_3._set_crostn_inelastic(inelas_reaction_dataframe=pd.read_pickle(
            sys.path[0] + r'\e.g._cs_CO2(vc)_vibexc_CO2(v1)_0.0459.pkl'))
        case_3._set_index_bg_molecule(total_species=species)

        case_4._set_crostn_inelastic(inelas_reaction_dataframe=pd.read_pickle(
            sys.path[0] + r'\e.g._cs_CO2(v1)_vibexc_CO2(v0)_-0.291.pkl'))
        case_4._set_index_bg_molecule(total_species=species)

        case_5._set_crostn_inelastic(inelas_reaction_dataframe=pd.read_pickle(
            sys.path[0] + r'\e.g._cs_CO2(v1)_vibexc_CO2(vc)_-0.0459.pkl'))
        case_5._set_index_bg_molecule(total_species=species)
        self.case_2 = case_2
        self.case_3 = case_3
        self.case_4 = case_4
        self.case_5 = case_5

    # %%----------------------------------------------------------------------------------------- #
    def test_set_parameters(self):
        eedf = self.case_0
        eedf.set_parameters(E=10.0, Tgas=100.0, N=1e21)
        assert_allclose(eedf.electric_field, 10.0)
        assert_allclose(eedf.gas_temperature, 100.0)
        assert_allclose(eedf.total_bg_molecule_density, 1e21)

    # %%----------------------------------------------------------------------------------------- #
    def test_get_electron_properties(self):
        eedf = self.case_0
        assert_allclose(eedf.energy_max_bound, 10 * const.eV2J)
        assert_allclose(eedf.grid_number, 100)
        assert_allclose(eedf.energy_point, np.linspace(0.05, 9.95, num=100) * const.eV2J)
        assert_allclose(eedf.energy_nodes, np.linspace(0.0, 10, num=101) * const.eV2J)
        assert_allclose(eedf.energy_intvl, 0.1 * const.eV2J)
        assert_allclose(eedf.electron_temperature, 1.0 * const.eV2K, rtol=1e-1)
        assert_allclose(eedf.electron_density, 1.0, rtol=1e-1)
        assert_allclose(eedf.electron_mean_energy, 1.5 * const.eV2J, rtol=1e-1)

    # %%----------------------------------------------------------------------------------------- #
    def test_alpha(self):
        eedf = self.case_0
        alpha_desired = 2 * math.pi / 3 * math.sqrt(2 / const.m_e) * \
                        (const.fine_structure * const.hbar * const.c) ** 2 * \
                        math.log(8 * math.pi / const.e ** 3 *
                                 math.sqrt(2 * (
                                         const.epsilon_0 * eedf.electron_mean_energy) ** 3
                                           / 3 / eedf.electron_density))
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

    # %%----------------------------------------------------------------------------------------- #
    def test_set_crostn_elas(self):
        eedf = self.case_1
        assert eedf.bg_molecule_elas == ['CO2', 'CO', 'O2']
        assert_allclose(eedf.bg_molecule_mass_elas, 1.66053904e-27 * np.array([44, 28, 32]),
                        rtol=1e-2)

    def test_set_crostn_inelas(self):
        eedf = self.case_1
        assert eedf.bg_molecule_inelas == ['CO2(v0)', 'CO2(v1)']

    # %%----------------------------------------------------------------------------------------- #
    def test_index_bg_molecule(self):
        eedf = self.case_1
        assert_allclose(eedf._index_bg_molecule_elas.toarray(),
                        np.array([[1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1]]))
        assert_allclose(eedf._index_bg_molecule_inelas.toarray(),
                        np.array([[0, 1, 0, 0, 0],
                                  [0, 0, 1, 0, 0]]))

    def test_set_flux_electric_field(self):
        eedf = self.case_1
        eedf.set_parameters(E=1e2, Tgas=1e3, N=1e25)
        eedf._set_flux_electric_field(_density=np.array([1, 2, 3, 4, 5]))

        def dndt(t, y, _eedf):
            _eedf.density_in_J = y
            _eedf.set_flux_electric_field(_density=np.array([1, 2, 3, 4, 5]))
            return -(_eedf.J_flux_ef[1:] - _eedf.J_flux_ef[:-1]) / _eedf.energy_intvl

        # species conservation
        _, a = ode_ivp(deriv_func=dndt,
                       func_args=(eedf,),
                       time_span=(0, 1e4),
                       y_0=get_maxwell_eedf(eedf.energy_point, Te_eV=1.0),
                       rtol=1e-6,
                       atol=1e-12,
                       show_time=False)
        eedf.density_in_J = a[-1]
        assert_allclose(eedf.electron_density, 1.0, rtol=1e-4)
        steady_state_desired = 3 / 2 * eedf.energy_max_bound ** (-1.5) * np.sqrt(eedf.energy_point)
        steady_state_evalued = a[-1]
        assert_allclose(steady_state_evalued[0], steady_state_desired[0], rtol=1e-1)
        assert_allclose(steady_state_evalued[1:9], steady_state_desired[1:9], rtol=2e-2)
        assert_allclose(steady_state_evalued[9:], steady_state_desired[9:], rtol=2e-3)

    def test_set_flux_elastic_colli(self):
        eedf = self.case_1
        eedf.set_parameters(E=1e2, Tgas=1.5 * const.eV2K, N=1e25)
        eedf._set_flux_elastic_colli(_density=np.array([1, 2, 3, 4, 5]))

        # species conservation
        def dndt(t, y, _eedf):
            _eedf.density_in_J = y
            _eedf.set_flux_elastic_colli(_density=np.array([1, 2, 3, 4, 5]))
            return -(_eedf.J_flux_el[1:] - _eedf.J_flux_el[:-1]) / _eedf.energy_intvl

        _, a = ode_ivp(deriv_func=dndt,
                       func_args=(eedf,),
                       time_span=(0, 1e5),
                       y_0=get_maxwell_eedf(eedf.energy_point, Te_eV=1.0),
                       rtol=1e-6,
                       atol=1e-12,
                       show_time=False)
        eedf.density_in_J = a[-1]
        assert_allclose(eedf.electron_density, 1.0, rtol=1e-3)
        assert_allclose(eedf.electron_temperature, 1.5 * const.eV2K, rtol=2e-2)
        steady_state_desired = get_maxwell_eedf(eedf.energy_point, Te_eV=1.5)
        steady_state_evalued = a[-1]
        assert_allclose(steady_state_evalued[0], steady_state_desired[0], rtol=1e-1)
        assert_allclose(steady_state_evalued[1:4], steady_state_desired[1:4], rtol=2e-2)
        assert_allclose(steady_state_evalued[4:], steady_state_desired[4:], rtol=1e-2)

    def test_set_flux_ee_colli(self):
        eedf = self.case_0

        def dndt(t, y, _eedf):
            _eedf.density_in_J = y
            _eedf._set_flux_ee_colli()
            return -(_eedf.J_flux_ee[1:] - _eedf.J_flux_ee[:-1]) / _eedf.energy_intvl

        y_0 = np.zeros_like(eedf.energy_point)
        y_0[(eedf.energy_point < 5.5 * const.eV2J) & (eedf.energy_point > 0.5 * const.eV2J)] = \
            1e16 / 5.0 / const.eV2J
        eedf.density_in_J = y_0
        density_0 = eedf.electron_density
        Te_0 = eedf.electron_temperature
        sol = ode_ivp(deriv_func=dndt,
                       func_args=(eedf,),
                       time_span=(0, 1e5),
                       y_0=y_0,
                       rtol=1e-6,
                       atol=1e-12,
                       show_time=False)
        # plt.plot(eedf.energy_point, a.transpose())

        # eedf.density_in_J = a[-1]
        eedf.density_in_J = sol.y
        density_1 = eedf.electron_density
        Te_1 = eedf.electron_temperature
        # species conservation and energy conservation
        assert_allclose(density_0, density_1, rtol=2e-3)
        assert_allclose(Te_0, Te_1, rtol=1e-3)

    def test_set_rate_const_matrix(self):
        threshold_tuple = ((2, 0.91),
                           (0, 0.4594),
                           (2, 0.91),
                           (0, 0.4594))
        for i_case, eedf in enumerate([self.case_2, self.case_3, self.case_4, self.case_5]):
            eedf._set_rate_const_matrix_e_inelas_electron()
            eedf._set_rate_const_matrix_e_inelas_molecule()
            crostn_eV_m2 = eedf.inelas_reaction_dataframe['cross_section'][0]
            _energy = np.hstack((0.0, crostn_eV_m2[0], np.inf))
            _crostn = np.hstack((0.0, crostn_eV_m2[1], 0.0))
            _energy_discretized = deepcopy(eedf.energy_point)
            _n, _phi = threshold_tuple[i_case]
            _energy_discretized[_n] = _energy_discretized[_n] + 0.5 * _phi * eedf.energy_intvl
            _crostn_discretized = interp1d(_energy, _crostn)(_energy_discretized * const.J2eV)
            _gamma = math.sqrt(2 / const.m_e)
            _temp = _gamma * _crostn_discretized * np.sqrt(eedf.energy_point)
            electron_rate_const_matrix = eedf.rate_const_matrix_e_inelas_electron.toarray()
            molecule_rate_const_matrix = eedf.rate_const_matrix_e_inelas_molecule[0]
            if i_case == 0:
                assert_allclose(electron_rate_const_matrix.sum(axis=0),
                                np.zeros_like(eedf.energy_point), atol=1e-29)
                assert_allclose(electron_rate_const_matrix.diagonal()[_n],
                                -0.09 * _temp[_n])
                assert_allclose(electron_rate_const_matrix.diagonal()[_n + 1:],
                                -1 * _temp[_n + 1:])
                assert_allclose(electron_rate_const_matrix.diagonal(offset=_n),
                                0.09 * _temp[_n:])
                assert_allclose(electron_rate_const_matrix.diagonal(offset=_n + 1),
                                0.91 * _temp[_n + 1:])
                assert_allclose(molecule_rate_const_matrix[:_n],
                                0.0 * _temp[:_n] * eedf.energy_intvl)
                assert_allclose(molecule_rate_const_matrix[_n],
                                2 * 0.09 / 2.91 * _temp[_n] * eedf.energy_intvl)
                assert_allclose(molecule_rate_const_matrix[_n + 1:],
                                1 * _temp[_n + 1:] * eedf.energy_intvl)
                assert_allclose(-molecule_rate_const_matrix * 0.291 * const.eV2J,
                                (eedf.energy_point[np.newaxis].transpose() *
                                 electron_rate_const_matrix *
                                 eedf.energy_intvl).sum(axis=0))
            elif i_case == 1:
                assert_allclose(electron_rate_const_matrix.sum(axis=0),
                                np.zeros_like(eedf.energy_point), atol=1e-29)
                assert_allclose(electron_rate_const_matrix.diagonal()[0],
                                _phi * (1 - _phi) * _temp[0])
                assert_allclose(electron_rate_const_matrix.diagonal()[1:],
                                -_phi * _temp[1:])
                assert_allclose(electron_rate_const_matrix.diagonal(offset=1),
                                _phi * _temp[1:])
                assert_allclose(electron_rate_const_matrix[1, 0],
                                -_phi * (1 - _phi) * _temp[0])
                assert_allclose(molecule_rate_const_matrix[0],
                                1 * (1 - _phi) * _temp[0] * eedf.energy_intvl)
                assert_allclose(molecule_rate_const_matrix[1:],
                                1 * _temp[1:] * eedf.energy_intvl)
                assert_allclose(-molecule_rate_const_matrix * 0.04594 * const.eV2J,
                                (eedf.energy_point[np.newaxis].transpose() *
                                 electron_rate_const_matrix *
                                 eedf.energy_intvl).sum(axis=0))
            elif i_case == 2:
                assert_allclose(electron_rate_const_matrix.sum(axis=0),
                                np.zeros_like(eedf.energy_point), atol=1e-29)
                assert_allclose(electron_rate_const_matrix.diagonal()[:eedf.grid_number - _n - 1],
                                -1 * _temp[:eedf.grid_number - _n - 1])
                assert_allclose(electron_rate_const_matrix.diagonal()[eedf.grid_number - _n - 1:],
                                0 * _temp[eedf.grid_number - _n - 1:])
                assert_allclose(electron_rate_const_matrix.diagonal(offset=-_n)
                                [:eedf.grid_number - _n - 1],
                                (1 - _phi) * _temp[:eedf.grid_number - _n - 1])
                assert_allclose(electron_rate_const_matrix.diagonal(offset=-_n - 1)
                                [:eedf.grid_number - _n - 1],
                                _phi * _temp[:eedf.grid_number - _n - 1])
                assert_allclose(molecule_rate_const_matrix[:eedf.grid_number - _n - 1],
                                1 * _temp[:eedf.grid_number - _n - 1] * eedf.energy_intvl)
                assert_allclose(molecule_rate_const_matrix[eedf.grid_number - _n - 1:],
                                0 * _temp[eedf.grid_number - _n - 1:] * eedf.energy_intvl)
                assert_allclose(-molecule_rate_const_matrix * (-0.291) * const.eV2J,
                                (eedf.energy_point[np.newaxis].transpose() *
                                 electron_rate_const_matrix *
                                 eedf.energy_intvl).sum(axis=0))
            elif i_case == 3:
                assert_allclose(electron_rate_const_matrix.sum(axis=0),
                                np.zeros_like(eedf.energy_point), atol=1e-29)
                assert_allclose(electron_rate_const_matrix.diagonal()[:-1],
                                -_phi * _temp[:-1])
                assert_allclose(electron_rate_const_matrix.diagonal(offset=-1),
                                _phi * _temp[:-1])
                assert_allclose(molecule_rate_const_matrix[:- 1],
                                1 * _temp[:-1] * eedf.energy_intvl)
                assert_allclose(molecule_rate_const_matrix[-1],
                                0 * _temp[-1] * eedf.energy_intvl)
                assert_allclose(-molecule_rate_const_matrix * (-0.04594) * const.eV2J,
                                (eedf.energy_point[np.newaxis].transpose() *
                                 electron_rate_const_matrix *
                                 eedf.energy_intvl).sum(axis=0))
            else:
                pass

    '''
        def dndt(t, y, eedf, density):
            eedf.density_in_J = y
            return eedf.get_electron_rate_e_inelas(density=density)

        y_0 = 1e16 * get_maxwell_eedf(eedf.energy_point, Te_eV=1.0)
        density = 1e25 * np.array([1, 2, 3, 4, 5])
        print(eedf.get_molecule_rate_e_inelas(density=density) * 0.291 * const.eV2J)
        print((eedf.get_electron_rate_e_inelas(
                density=density) * eedf.energy_point * eedf.energy_intvl).sum())
        print(eedf.inelas_reaction_dataframe)
        _, a = ode_ivp(deriv_func=dndt,
                       func_args=(eedf, density),
                       time_span=(0, 1e-12),
                       y_0=y_0,
                       rtol=1e-6,
                       atol=1e-12,
                       show_time=False)
        plt.plot(eedf.energy_point, a.transpose())
        eedf.density_in_J = a[-1]




    def test_electric_field_elastic_colli_flux(self):
        eedf = self.case_0
        eedf.set_crostn_elastic(bg_molecule_elas=['CO', 'N2', 'CO2'])
        # %%------------------------------------------------------------------------------------- #
        eedf.crostn_elas = np.vstack((2e-20 * np.ones_like(eedf.energy_nodes),
                                      3e-20 * np.ones_like(eedf.energy_nodes),
                                      5e-20 * np.ones_like(eedf.energy_nodes)))
        # %%------------------------------------------------------------------------------------- #
        eedf.set_parameters(E=10.0, N=10.0, Tgas=1e3)
        eedf.set_flux_electric_field()
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
    '''
