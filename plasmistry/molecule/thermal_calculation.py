#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14:47 2017/7/5

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import

import re
from math import log
from .. import constants as const
from .read_data import species_thermal_data, molecular_const


# ----------------------------------------------------------------------------------------------- #
class MoleculeError(Exception):
    """pass"""
    pass


# ----------------------------------------------------------------------------------------------- #
def get_ideal_gas_density(*, p_Pa, Tgas_K):
    """
    Calculate the gas density based on the ideal gas law.

    Parameters
    ----------
    p_Pa : float
        Pressure in Pa.
    Tgas_K : float
        Temperature of gas in Kelvin.

    Returns
    -------
    density : float
         Gas number density in unit of m^-3

    Examples
    --------
    >>> from plasmistry.molecule import get_ideal_gas_density
    >>> from plasmistry import constants as const
    >>> get_ideal_gas_density(p_Pa=const.pressure_STP,Tgas_K=const.temperature_STP)
    2.6516467463592656e+25
    >>> get_ideal_gas_density(p_Pa=const.pressure_NTP,Tgas_K=const.temperature_NTP)
    2.5034768825147869e+25

    """
    assert isinstance(p_Pa, float) or isinstance(p_Pa, int)
    assert isinstance(Tgas_K, float) or isinstance(Tgas_K, int)
    assert p_Pa > 0
    assert Tgas_K > 0
    return p_Pa / const.R / Tgas_K * const.N_A


def get_reaction_enthalpy(*, reaction, particle_enthalpy):
    """
    Calculate reaction enthalpy based on enthalpy_dict.

    Parameters
    ----------
    reaction : str
        Reaction string like 'A + B => C + D'
    particle_enthalpy : dict
        {particle[string] : enthalpy[float]}

    Returns
    -------
    reaction_enthalpy : float
        Reaction enthalpy in unit same with particle_enthalpy value

    """
    assert isinstance(reaction, str)
    assert '=>' in reaction
    assert isinstance(particle_enthalpy, dict)

    rcnt_str, prdt_str = re.split(r'\s*[<]?[=][>]\s*', reaction.strip())
    sum_H = lambda _list: sum(
            particle_enthalpy[particle] for particle in _list) if _list else 0.0
    get_H = lambda _str: sum_H(re.split(r'\s+[+]\s+', _str.strip()) if _str.strip() else [])
    return get_H(prdt_str) - get_H(rcnt_str)


# ----------------------------------------------------------------------------------------------- #
def get_vib_energy(molecule, *, quantum_number, state='X', minimum_is_zero=False):
    """
    Calculate the vibrational energy of molecule.

    Parameters
    ----------
    molecule : str
        The molecule name.
    quantum_number : int or tuple of int
        Quantum number. int for diatomic, (int,int,int) for triatomic.
    state : str, optional
        Electric state.
    minimum_is_zero : bool, optional
        Whether the minimum vibrational energy is set at zero.

    Returns
    -------
    vibrational energy : float
        Vibrational energy in unit of eV.

    """
    assert molecule in molecular_const
    if molecule in ('CO', 'O2', 'N2', 'H2'):
        assert isinstance(quantum_number, int), quantum_number
    elif molecule in ('H2O', 'CO2'):
        assert isinstance(quantum_number, tuple), quantum_number
        assert len(quantum_number) == 3

    spe_const = molecular_const[molecule][state]
    lamb_diatomic = lambda c, v: const.WNcm2eV * (c['we'] * (v + 0.5) - c['wexe'] * (v + 0.5)**2 +
                                                c['weye'] * (v + 0.5)**3)
    lamb_triatomic = lambda c, v, n: const.WNcm2eV * sum([c['w1'] * (v[0] + n[0]),
                                                        c['w2'] * (v[1] + n[1]),
                                                        c['w3'] * (v[2] + n[2]),
                                                        c['X11'] * (v[0] + n[0])**2,
                                                        c['X22'] * (v[1] + n[1])**2,
                                                        c['X33'] * (v[2] + n[2])**2,
                                                        c['X12'] * (v[0] + n[0]) * (v[1] + n[1]),
                                                        c['X13'] * (v[0] + n[0]) * (v[2] + n[2]),
                                                        c['X23'] * (v[1] + n[1]) * (v[2] + n[2]),
                                                        c['Xll'] * (v[1]**2 - 1)])

    def _get_vib_energy(mole, v):
        if mole in ('CO', 'O2', 'N2', 'H2'):
            return lamb_diatomic(spe_const, v), lamb_diatomic(spe_const, 0)
        elif mole == 'CO2':
            n = (1 / 2, 1, 1 / 2)
            return (lamb_triatomic(spe_const, v, n),
                    lamb_triatomic(spe_const, (0, 0, 0), n))
        elif mole == 'H2O':
            n = (1 / 2, 1 / 2, 1 / 2)
            return (lamb_triatomic(spe_const, v, n),
                    lamb_triatomic(spe_const, (0, 0, 0), n))
        else:
            raise MoleculeError("{}'s data is not imported.".format(mole))

    if minimum_is_zero and (quantum_number == 0 or quantum_number == (0, 0, 0)):
        return 0.0
    if minimum_is_zero:
        return _get_vib_energy(molecule, quantum_number)[0] - \
               _get_vib_energy(molecule, quantum_number)[1]
    else:
        return _get_vib_energy(molecule, quantum_number)[0]


# ----------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    pass
