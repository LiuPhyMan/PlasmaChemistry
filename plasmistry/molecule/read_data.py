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
import math
from math import log
import numpy as np
import pandas as pd
from .. import constants as const
from ._data import atomic_rela_mass_file_path, molecular_const_file_path, thermo_data_file_path


# ----------------------------------------------------------------------------------------------- #
class MoleculeError(Exception):
    """pass"""
    pass


# ----------------------------------------------------------------------------------------------- #
def _read_atomic_rela_mass(file_path):
    """
    Read the atomic relative mass into a dict.

    Parameters
    ----------
    file_path : str
        The atomic mass file path.

    Returns
    -------
    atomic_relative_mass : dict
        {atom_str : atom_mass_value}

    """
    with open(file_path) as f:
        result = {atom: float(mass_str) for atom, mass_str in
                  [line.strip().split() for line in f
                   if line.strip() and not line.strip().startswith('#')]}
    return result


# ----------------------------------------------------------------------------------------------- #
def _read_molecule_const(file_path):
    """
    Read the molecule constants from the file in file_path.

    Parameters
    ----------
    file_path : str
        The molecule constants file path.

    Returns
    -------
    molecular_const : dict of pd.DataFrame

    Examples
    --------
    >>> from plasmistry import molecule
    >>> molecule.molecular_const['CO']

                       X             A             B
    Te          0.000000  65075.770000  86945.200000
    we       2169.813580   1518.240000   2112.700000
    wexe       13.288310     19.400000     15.220000
    weye        0.000000      0.000000      0.000000
    Be          1.931281      1.611500      1.961200
    alpha_e     0.017504      0.023250      0.026100
    gama_e      0.000000      0.000000      0.000000
    De          0.000006      0.000007      0.000007
    beta_e      0.000000      0.000000      0.000000
    re          1.128323      1.235300      1.119700

    """
    molecule_const = dict()
    with open(file_path) as f:
        for line in f:
            if not line.strip().startswith('@'):
                continue
            else:
                molecule_name = line.strip()[1:]
                state = [_[1:] for _ in f.readline().strip().split()]
                data = []
                const_name = []
                while True:
                    subline = f.readline().strip()
                    if subline == 'END':
                        break
                    else:
                        const_name.append(subline.split()[0])
                        sub_data = subline.split()[1:]
                        data.append([float(_) if _ != 'NA' else 0.0 for _ in sub_data])
                molecule_const[molecule_name] = pd.DataFrame(data, index=const_name, columns=state)
    return molecule_const


# ----------------------------------------------------------------------------------------------- #
def _read_particle_data(file_path, _atomic_rela_mass):
    """
    Read the thermal data from therm file to a DataFrame, the therm_file is from CHEMKIN file--
    therm.data V.4.0 March 2004

    Parameters
    ----------
    file_path : str
        The thermal data file path.
    _atomic_rela_mass : dict
        {atom : atom_relative_mass}

    Returns
    -------
    particle_data : DataFrame
        Index
            formula
        Columns
            name
            source
            H0
            S0
            atomic_symbols
            relative_mass
            absolute_mass
            phase
            T_range
            coef_lower_T
            coef_upper_T

    Notes
    -----
    particle_data includes :
        ----                            format  position
        Species name:                   16A1    1-16
        Atomic symbols and formula:     6A1     25-44
        Phase of species (S,L,or G):    A1      45
        Low temperature:                E10.0   46-55
        High temperature:               E10.0   56-65
        Common temperature:             E8.0    66-73
        A mandatory element(integer_1): I1      80
        Coefficients_line_2:            5E15.8  1-75    #   a1-a5 for upper temperature interval
        Coefficients_line_3:            5E15.8  1-75    #   a6-a7 for upper & a1-a3 for lower
        Coefficients_line_4:            4E15.8  1-75    #   a4-a7 for lower

    Examples
    --------
    !
    ! Species: CO2              CAS Number: 124-38-9
    ! Name:    Carbon Dioxide
    ! Source:  ReactionDesign fit to 1986 JANAF tables, J155
    ! H0(298K) =      -94.0100 (Kcal/mole),  S0(298K) =       51.0700 (cal/mole-K)
    CO2               J155  C   1O   2          G   200.000  6000.000 1400.00      1
     5.02582135e+00 2.21159005e-03-7.60738632e-07 1.19929044e-10-6.95997162e-15    2
    -4.92013983e+04-4.21727446e+00 2.47951421e+00 8.22287212e-03-5.84723958e-06    3
     1.89723502e-09-2.09186513e-13-4.83645720e+04 9.36630565e+00                   4

    """
    particle_data = pd.DataFrame(columns=['name', 'source', 'H0', 'S0',
                                          'atomic_symbols',
                                          'relative_mass', 'absolute_mass',
                                          'phase', 'T_range',
                                          'coef_lower_T', 'coef_upper_T'])
    with open(file_path) as f:
        for line in f:
            if line[2:10] == 'Species:':
                formula = line[11:11 + 16].strip()
                assert formula not in particle_data.index
                particle_data.loc[formula, 'name'] = f.readline()[7:].strip()
                particle_data.loc[formula, 'source'] = f.readline()[9:].strip()
                temp = f.readline().strip()
                particle_data.loc[formula, 'H0'] = float(temp[12:27].strip())
                particle_data.loc[formula, 'S0'] = float(temp[51:66].strip())
                temp = f.readline()
                particle_data.loc[formula, 'atomic_symbols'] = temp[24:44].strip()
                particle_mass = sum(_atomic_rela_mass[s[:2].strip().upper()] * int(s[2:].strip())
                                    for s in (temp[24:29], temp[29:34], temp[34:39], temp[39:44])
                                    if s.strip())
                particle_data.loc[formula, 'relative_mass'] = particle_mass
                particle_data.loc[formula, 'absolute_mass'] = particle_mass * const.atomic_mass
                particle_data.loc[formula, 'phase'] = temp[44:45].strip()
                particle_data.loc[formula, 'T_range'] = (float(temp[45:55].strip()),
                                                         float(temp[65:73].strip()),
                                                         float(temp[55:65].strip()))
                coef_str = f.readline()[0:75] + f.readline()[0:75] + f.readline()[0:75]
                coef_T = [float(coef_str[15 * i:15 * (i + 1)].strip()) for i in range(14)]
                particle_data.loc[formula, 'coef_upper_T'] = np.array(coef_T[0:7],
                                                                      dtype=np.float64)
                particle_data.loc[formula, 'coef_lower_T'] = np.array(coef_T[7:14],
                                                                      dtype=np.float64)
            else:
                continue
    return particle_data


class THERMO(object):

    def __init__(self, file_path):
        self.__func_enthalpy = lambda c, T: (c[0] + c[1] / 2 * T + c[2] / 3 * T ** 2 +
                                             c[3] / 4 * T ** 3 + c[4] / 5 * T ** 4 + c[
                                                 5] / T) * const.R * T
        self.__func_entropy = lambda c, T: (c[0] * log(T) + c[1] * T + c[2] / 2 * T ** 2 +
                                            c[3] / 3 * T ** 3 + c[4] / 4 * T ** 4 + c[6]) * const.R
        self.__func_C_p = lambda c, T: (c[0] + c[1] * T + c[2] * T ** 2 +
                                        c[3] * T ** 3 + c[4] * T ** 4) * const.R
        self.__set_particle_data(file_path)

    def __set_particle_data(self, file_path):
        r"""
        Set thermochemistry data of particles.
                phase   T_range     coef_lower_T    coef_upper_T
        INDEX
        formula ...

        """
        with open(file_path) as f:
            data_list = [_.rstrip() for _ in f if (not _.startswith('!')) and (_.strip() != '')]
        assert data_list[0].startswith('THERMO')
        data_list = data_list[2:-1]
        assert math.fmod(len(data_list), 4) == 0
        particle_data = pd.DataFrame(columns=['phase', 'T_range', 'coef_lower_T', 'coef_upper_T'])
        for i in range(0, len(data_list), 4):
            formula = data_list[i][:16].strip().split()[0]
            # assert formula not in particle_data.index, formula
            if formula in particle_data.index:
                print(formula)
                continue
            particle_data.loc[formula, 'phase'] = data_list[i][44].strip()
            particle_data.loc[formula, 'T_range'] = (float(data_list[i][45:53].strip()),
                                                     float(data_list[i][65:73].strip()),
                                                     float(data_list[i][55:63].strip()))
            coef_str = data_list[i + 1][:75] + data_list[i + 2][:75] + data_list[i + 3][:60]
            coef_T = [float(coef_str[15 * i:15 * (i + 1)].strip()) for i in range(14)]
            particle_data.loc[formula, 'coef_upper_T'] = np.array(coef_T[:7])
            particle_data.loc[formula, 'coef_lower_T'] = np.array(coef_T[7:14])
        self.particle_data = particle_data
        self.formulas = particle_data.index.tolist()

    def therm_data_dict(self):
        return self.particle_data.transpose().to_dict()
    
    def __get_thermal_coefs(self, name, *, tmpr_K):
        r"""
        Get the thermal coefficients at some pressure.

        Parameters
        ----------
        name : str
            Species name.
        tmpr_K : float
            Temperature in Kelvin.

        Returns
        -------
        thermal coefficients : tuple of float
            Tuple of seven.

        """
        assert name in self.particle_data.index
        assert isinstance(tmpr_K, float) or isinstance(tmpr_K, int)
        assert tmpr_K >= self.particle_data.loc[name, 'T_range'][0]
        assert tmpr_K <= self.particle_data.loc[name, 'T_range'][2]

        mid_temperature = self.particle_data.loc[name, 'T_range'][1]
        temperature_range = 'coef_lower_T' if tmpr_K < mid_temperature else 'coef_upper_T'
        coefs = self.particle_data.loc[name, temperature_range]
        return coefs

    def enthalpy(self, name, *, tmpr_K):
        """
        Calculate the enthalpy of species in some temperature.

        Parameters
        ----------
        name : str
            Species name.
        tmpr_K : float
            Temperature in Kelvin.

        Returns
        -------
        enthalpy : float
            The enthalpy in unit of J/mol.

        """
        coefs = self.__get_thermal_coefs(name, tmpr_K=tmpr_K)
        return self.__func_enthalpy(coefs, tmpr_K)

    def entropy(self, name, *, tmpr_K):
        """
        Calculate the entropy of species in some temperature.

        Parameters
        ----------
        name : str
            Species name.
        tmpr_K : float
            Temperature in Kelvin.

        Returns
        -------
        entropy : float
            The entropy in unit of J/mol/K.

        """
        coefs = self.__get_thermal_coefs(name, tmpr_K=tmpr_K)
        return self.__func_entropy(coefs, tmpr_K)

    def free_energy(self, name, *, tmpr_K):
        """
        Calculate the free energy of species in some temperature.

        Parameters
        ----------
        name : str
            Species name.
        tmpr_K : float
            Temperature in Kelvin.

        Returns
        -------
        free energy : float
            Free energy in unit of J mol^-1.

        """
        return self.enthalpy(name, tmpr_K=tmpr_K) - tmpr_K * self.entropy(name, tmpr_K=tmpr_K)

    def C_p(self, name, *, tmpr_K):
        """
        Calculate the specific heat capacity at constant pressure of species
        in some temperature.

        Parameters
        ----------
        name : str
            Species name.
        tmpr_K : float
            Temperature in Kelvin.

        Returns
        -------
        C_p : float
            The enthalpy in unit of J/mol/K.

        """
        coefs = self.__get_thermal_coefs(name, tmpr_K=tmpr_K)
        return self.__func_C_p(coefs, tmpr_K)

    def C_v(self, name, *, tmpr_K):
        """
        Calculate the specific heat capacity at constant volume of species
        in some temperature.

        See Also
        --------
        get_C_p(name,*,tmpr_K)

        """
        return self.C_p(name, tmpr_K=tmpr_K) - const.R

    def equilibrium_const(self, *, reaction_str, tmpr_seq):
        rcnt_str, prdt_str = re.split(r'\s*[<]?[=][>]\s*', reaction_str)
        rcnt_list = re.split(r'\s*\+\s*', rcnt_str)
        prdt_list = re.split(r'\s*\+\s*', prdt_str)
        K_p_seq = []
        for tmpr in tmpr_seq:
            d_G = np.sum([self.free_energy(_, tmpr_K=tmpr) for _ in prdt_list]) - \
                  np.sum([self.free_energy(_, tmpr_K=tmpr) for _ in rcnt_list])
            d_v = len(prdt_list) - len(rcnt_list)
            K_n = np.exp(-d_G / const.R / tmpr)
            K_p = K_n * (const.atm / const.R / tmpr) ** d_v
            K_p_seq.append(K_p)
        return K_p_seq


molecular_const = _read_molecule_const(molecular_const_file_path)
atomic_rela_mass = _read_atomic_rela_mass(atomic_rela_mass_file_path)
species_thermal_data = _read_particle_data(thermo_data_file_path, atomic_rela_mass)
