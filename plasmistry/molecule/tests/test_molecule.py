#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9:26 2017/7/6

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

from __future__ import division, print_function, absolute_import

from numpy.testing import (assert_allclose,
                           assert_approx_equal,
                           run_module_suite, )

from plasmistry import constants as const
from plasmistry.molecule import species_thermal_data, atomic_rela_mass, molecular_const
from plasmistry.molecule import (get_vib_energy,
                                 get_ideal_gas_density,
                                 get_enthalpy,
                                 get_entropy,
                                 get_C_p,
                                 get_C_v,
                                 get_free_energy,
                                 get_reaction_enthalpy, )


# %%--------------------------------------------------------------------------------------------- #
def test_read_data():
    assert_allclose([atomic_rela_mass[m] for m in ('E', 'H', 'C', 'S')],
                    [5.4858E-4, 1.00794, 12.0107, 32.065], atol=1e-4)
    assert_allclose([molecular_const['N2'].loc['Te', state] for state in ('X', 'A', 'B', 'C')],
                    [0.0, 50203.6, 59619.35, 89136.88],
                    atol=1e-2)
    assert_allclose([molecular_const['N2'].loc['we', 'X'],
                     molecular_const['O2'].loc['we', 'X'],
                     molecular_const['CO'].loc['we', 'X'],
                     molecular_const['H2'].loc['we', 'X']],
                    [2358.57, 1580.19, 2169.81358, 4401.21],
                    atol=1e-2)
    assert_allclose([molecular_const['N2'].loc['Be', 'X'],
                     molecular_const['O2'].loc['Be', 'X'],
                     molecular_const['CO'].loc['Be', 'X'],
                     molecular_const['H2'].loc['Be', 'X']],
                    [1.99824, 1.4376766, 1.93128087, 60.853],
                    atol=1e-3)


# %%--------------------------------------------------------------------------------------------- #
def test_get_ideal_gas_density():
    assert_approx_equal(get_ideal_gas_density(p_Pa=const.pressure_NTP,
                                              Tgas_K=const.temperature_NTP), 2.5034769E+25)
    assert_approx_equal(get_ideal_gas_density(p_Pa=const.pressure_STP,
                                              Tgas_K=const.temperature_STP), 2.6516467E+25)


# %%--------------------------------------------------------------------------------------------- #
def test_thermal_calculation():
    assert_allclose([get_enthalpy(_, tmpr_K=298.15) * const.J2Kcal
                     for _ in ('H2', 'O2', 'N2', 'E', 'Ar', 'Rn', 'Xe')],
                    [0.0] * 7, atol=1e-7)
    assert_allclose([get_enthalpy(_, tmpr_K=298.15) * const.J2Kcal
                     for _ in ('CO', 'CO2', 'H2O', 'HO', 'CH4')],
                    [-26.4, -94.01, -57.77, 9.31, -17.89], atol=1e-2)
    assert_allclose([get_enthalpy(_, tmpr_K=298.15) * const.J2Kcal
                     for _ in ('C', 'H', 'O', 'N')],
                    [171.21, 52.08, 59.52, 112.92], atol=1e-2)

    # test H0, S0, G0
    for molecule in species_thermal_data.index:
        H0 = species_thermal_data.loc[molecule, 'H0']
        S0 = species_thermal_data.loc[molecule, 'S0']
        G0 = H0 - 298.15 * S0 * 1e-3
        a_H0 = 0 if abs(H0) > 1 else 2
        a_G0 = 0 if abs(G0) > 1 else 2
        assert_allclose([get_enthalpy(molecule, tmpr_K=298.15) * const.J2Kcal + a_H0],
                        [H0 + a_H0], rtol=1e-2)
        assert_allclose([get_entropy(molecule, tmpr_K=298.15) * const.J2cal],
                        [S0], rtol=1e-2)
        assert_allclose([get_free_energy(molecule, tmpr_K=298.15) * const.J2Kcal + a_G0],
                        [G0 + a_G0], rtol=1e-2)

    # test C_p
    assert_allclose([get_C_p(m, tmpr_K=300.0) for m in ('He', 'Ne', 'Ar', 'Kr', 'Xe')],
                    [20.776338] * 5,
                    rtol=1e-7)
    assert_allclose([get_C_p(m, tmpr_K=300.0) for m in ('C', 'H', 'O', 'N')],
                    [20.84, 20.78, 21.92, 20.78],
                    rtol=1e-2)
    assert_allclose([get_C_p('Ar', tmpr_K=T) for T in (300.0, 700.0, 1200.0, 3000.0, 6000.0)],
                    [20.776338] * 5,
                    rtol=1e-7)
    assert_allclose([get_C_p(m, tmpr_K=300.0) for m in ('H2', 'N2', 'CO', 'O2')],
                    [28.71, 29.07, 29.09, 29.42],
                    rtol=1e-2)
    assert_allclose([get_C_p(m, tmpr_K=300.0) for m in ('CO2', 'H2O', 'NO2', 'CH4')],
                    [37.16, 33.63, 37.09, 35.86],
                    rtol=1e-2)
    assert_allclose([get_C_v(m, tmpr_K=300.0) for m in ('He', 'Ne', 'Ar', 'Kr', 'Xe')],
                    [20.776338 - const.R] * 5,
                    rtol=1e-7)


# %%--------------------------------------------------------------------------------------------- #
def test_get_vib_energy():
    assert_allclose([0.0, 0.2889, 0.5742, 0.8559, 1.1342, 1.4088, 1.6801, 1.9475, 2.2115],
                    [get_vib_energy('N2', quantum_number=v, minimum_is_zero=True) for v in
                     range(9)],
                    rtol=1e-3)


# %%--------------------------------------------------------------------------------------------- #
def test_reaction_enthalpy():
    assert_approx_equal(get_reaction_enthalpy(reaction='A + B => C + D',
                                              particle_enthalpy=dict(A=1, B=2, C=3, D=4)), 4)
    assert_approx_equal(get_reaction_enthalpy(reaction='=> C + D',
                                              particle_enthalpy=dict(A=1, B=2, C=3, D=4)), 7)


# %%--------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_module_suite()
