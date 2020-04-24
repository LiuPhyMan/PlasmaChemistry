#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:58:46 2016

@author: liujinbao
"""
import os
import math
from math import log, pi, sqrt

import re
import numpy as np
import pandas as pd
import scipy.sparse as sprs
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from .. import constants as const
from ..molecule import species_thermal_data
# from ..io.io_cross_section import read_cross_section_to_frame
from pandas.core.frame import DataFrame as DataFrame_type


# ---------------------------------------------------------------------------- #
class EEDFerror(Exception):
    pass


# ---------------------------------------------------------------------------- #
class EEDF(object):
    __slots__ = [
        # ----
        # parameters
        'electric_field',  # E
        'gas_temperature',  # Tgas
        'total_bg_molecule_density',  # N
        # ----
        # discretization
        'energy_point',
        'energy_intvl',
        'energy_nodes',
        'energy_max_bound',
        'grid_number',
        # ----
        # distribution
        'density_per_J',
        # ----
        # elastic collisions
        '_index_bg_molecule_elas',
        'crostn_elas',  # ndarray (bg_molecule_elas number x grid_number)
        'bg_molecule_elas',  # list (bg_molecule_elas number)
        'bg_molecule_mass_elas',  # list (bg_molecule_elas number)
        # ----
        # inelas collisions
        '_index_bg_molecule_inelas',
        '_index_bg_molecule_inelas_in_order',
        'inelas_reaction_dataframe',
        'bg_molecule_inelas',
        'bg_molecule_inelas_in_order',
        'rate_const_matrix_e_inelas_electron',
        'rate_const_matrix_e_inelas_molecule',
        # ----
        # electric field
        'D_k_ef', 'F_k_ef',
        'J_flux_ef',
        # ----
        # elastic collisions
        'D_k_el', 'F_k_el',
        'J_flux_el',
        # ----
        # electron-electron collisions
        '_op_Q', '_op_P1', '_op_P2', '_op_P',
        'ee_alpha', 'ee_op_a', 'ee_op_b',
        'J_flux_ee',
    ]

    # ------------------------------------------------------------------------ #
    def __init__(self, *, max_energy_eV: float,
                 grid_number: int):
        r"""

        Parameters
        ----------
        max_energy_eV
        grid_number

        Notes
        -----
                                        k      k+1
                                   k-1      k      k+1
        energy_nodes
                1       2       3       4       5       6       7       8
                |---*---|---*---|---*---|---*---|---*---|---*---|---*---|
        energy_point
                    1       2       3       4       5       6       7
                |---*---|---*---|---*---|---*---|---*---|---*---|---*---|
                                    W       P       E
                                        w       e

        """
        self.energy_max_bound = max_energy_eV * const.eV2J
        self.grid_number = grid_number
        self.energy_nodes = np.linspace(0.0, self.energy_max_bound,
                                        num=self.grid_number + 1)
        self.energy_point = 0.5 * (
                self.energy_nodes[:-1] + self.energy_nodes[1:])
        self.energy_nodes.setflags(write=False)
        self.energy_point.setflags(write=False)
        self.energy_intvl = self.energy_nodes[1] - self.energy_nodes[0]
        self.bg_molecule_elas = None
        self.bg_molecule_inelas = None
        self.set_density_per_J(np.zeros_like(self.energy_point))
        self._pre_set_flux_ee_colli()

    def __setattr__(self, key, value):
        r"""
        Gas_temperature and total_density can not be negative.
        """
        if key in ('gas_temperature', 'total_bg_molecule_density'):
            assert value > 0.0, key
        object.__setattr__(self, key, value)

    # ------------------------------------------------------------------------ #
    #   properties
    # ------------------------------------------------------------------------ #
    @property
    def density_per_eV(self):
        return self.density_per_J / const.J2eV

    @property
    def energy_point_eV(self):
        return self.energy_point * const.J2eV

    @property
    def energy_nodes_eV(self):
        return self.energy_nodes * const.J2eV

    @property
    def electron_density(self):
        r"""Calculate the electron density in m^-3."""
        return trapz(y=np.hstack((0.0, self.density_per_J, 0.0)),
                     x=np.hstack(
                         (0.0, self.energy_point, self.energy_max_bound)))

    @property
    def reduced_electric_field_in_Td(self):
        r"""Calculate reduced electric field in Td."""
        return self.electric_field / self.total_bg_molecule_density * 1e21

    @property
    def electron_mean_energy_in_J(self):
        r"""Calculate electron mean energy in J."""
        _density = self.electron_density
        return simps(
            y=np.hstack((0.0, self.energy_point * self.density_per_J, 0.0)),
            x=np.hstack(
                (0.0, self.energy_point, self.energy_max_bound))) / _density

    @property
    def electron_mean_energy_in_eV(self):
        return self.electron_mean_energy_in_J * const.J2eV

    @property
    def electron_temperature_in_eV(self):
        r"""Calculate electron temperature in K."""
        return self.electron_mean_energy_in_J * const.J2K * 2 / 3 * const.K2eV

    @property
    def electron_temperature_in_K(self):
        r"""Calculate electron temperature in K."""
        return self.electron_temperature_in_eV * const.eV2K

    @property
    def eepf_J(self):
        r"""Electron energy probability function in J^{-3/2}."""
        return self.density_per_J / np.sqrt(self.energy_point)

    @property
    def eepf_eV(self):
        r"""Electron energy probability function in eV^{-3/2}."""
        return self.eepf_J * const.J2eV ** (-3 / 2)

    @property
    def normalized_eepf_J(self):
        return self.eepf_J / self.electron_density

    @property
    def normalized_eepf_eV(self):
        return self.eepf_eV / self.electron_density

    @property
    def normalized_eedf_J(self):
        return self.density_per_J / self.electron_density

    @property
    def normalized_eedf_eV(self):
        return self.density_per_eV / self.electron_density

    @property
    def n_inelas_rctn(self):
        return self.inelas_reaction_dataframe.shape[0]

    @property
    def n_bg_molecule_elas(self):
        return len(self.bg_molecule_elas)

    @property
    def n_bg_molecule_inelas(self):
        return len(self.bg_molecule_inelas)

    @property
    def inelas_rctn_info(self):
        _df = self.inelas_reaction_dataframe[
            ['formula', 'type', 'threshold_eV']].copy()
        _df['rate const'] = self._get_molecule_rate_const_e_inelas()
        _df['energy lose'] = _df['threshold_eV'] * _df['rate const']
        return _df

    # ------------------------------------------------------------------------ #
    #   public functions
    # ------------------------------------------------------------------------ #
    def initialize(self, *, rctn_with_crostn_df: DataFrame_type,
                   species_list: np.ndarray):
        r"""
        Set elastic/inelastic cross section for species.
        index of background molecule
        """
        self._set_crostn_elastic(species_list=species_list)
        self._set_crostn_inelastic(
            inelas_reaction_dataframe=rctn_with_crostn_df)
        self._set_index_bg_molecule(species_list=species_list)

    def set_density_per_J(self, value: np.ndarray):
        assert isinstance(value, np.ndarray)
        assert value.ndim == 1
        assert value.size == self.grid_number
        self.density_per_J = value

    def set_parameters(self, *, E=None, Tgas=None, N=None):
        if E is not None:
            self.electric_field = E
        if Tgas is not None:
            self.gas_temperature = Tgas
        if N is not None:
            self.total_bg_molecule_density = N

    def set_flux(self, *, total_species_density: np.ndarray):
        self._set_flux_electric_field(
            total_species_density=total_species_density)
        self._set_flux_elastic_colli(
            total_species_density=total_species_density)
        self._set_flux_ee_colli()

    def get_deriv_total(self, total_species_density=None):
        dndt = self._get_deriv_ef() + self._get_deriv_el() + self._get_deriv_ee()
        if total_species_density is None:
            return dndt
        else:
            return dndt + self._get_electron_rate_e_inelas(
                density=total_species_density)

    # ------------------------------------------------------------------------ #
    #   private functions
    # ------------------------------------------------------------------------ #
    def _set_crostn_elastic(self, *, species_list: list):
        r"""
        Set elastic collision cross sections at energy_nodes.

        Parameters
        ----------
        species_list : list

        Notes
        -----
        SET :
            self.bg_molecule_elas
                .bg_molecule_mass_elas
                .crostn_elas

        """
        assert isinstance(species_list, list)
        # total_elas_molecules = {'CO2', 'CO', 'O2', 'H2O', 'H2', 'CH4', 'H', 'N2', 'He'}
        total_elas_molecules = {'CO2', 'CO', 'O2', 'H2O',
                                'H2', 'CH4', 'N2', 'Ar'}
        #   bg_molecules are the total species in total_elas_moleucles
        self.bg_molecule_elas = [_ for _ in species_list if
                                 _ in total_elas_molecules]
        self.bg_molecule_mass_elas = np.array(
            [species_thermal_data.loc[_, 'absolute_mass']
             for _ in self.bg_molecule_elas])
        crostn_path = os.path.dirname(__file__) + r"\elastic_cross_section\\"
        crostn_elas = []
        for molecule in self.bg_molecule_elas:
            _crostn = np.loadtxt(crostn_path + f"{molecule}.dat").transpose()
            energy = np.hstack((0.0, _crostn[0], np.inf))
            crostn = np.hstack((0.0, _crostn[1], 0.0))
            # Add the elastic cross section interped into the crostn matrix.
            crostn_elas.append(interp1d(energy, crostn, bounds_error=True)
                               (self.energy_nodes * const.J2eV))
        self.crostn_elas = np.array(crostn_elas)

    def _set_crostn_inelastic(self, *,
                              inelas_reaction_dataframe: DataFrame_type):
        r"""
        Set inelas_reaction_dataframe.
            columns : [set previously]
                        formula
                        type
                        threshold_eV
                        cross_section
                      [set here]
                        bg_molecule
                        low_threshold

        Returns
        -------
        Set :
            self.inelas_reaction_dataframe
                .bg_molecule_inelas
                ._set_rate_const_matrix_e_inelas_electron()
                ._set_rate_const_matrix_e_inelas_molecule()

        Notes
        -----
        type : excitation deexcitation ionization

        """
        assert isinstance(inelas_reaction_dataframe, DataFrame_type)
        _dataframe = inelas_reaction_dataframe.copy()
        _dataframe = _dataframe.reset_index(drop=True)
        _dataframe['bg_molecule'] = ''
        _dataframe['low_threshold'] = None
        _get_bg_molecule = re.compile(r"""
        [eE]
        \s+\+\s+
        (?P<bg_molecule>[a-zA-Z]\S*)
        \s+=>.*""", re.VERBOSE | re.MULTILINE)  # e + M => ...
        #   Set low_threshold, bg_molecule
        for i_rctn in _dataframe.index:
            # ---------------------------------------------------------------- #
            # set background molecule
            # ---------------------------------------------------------------- #
            _temp = _get_bg_molecule.fullmatch(_dataframe.at[i_rctn, 'formula'])
            assert _temp, f"The format of {_dataframe.at[i_rctn, 'formula']} is wrong."
            _dataframe.at[i_rctn, 'bg_molecule'] = _temp.groupdict()[
                'bg_molecule']
            # ---------------------------------------------------------------- #
            # set low_threshold
            # ---------------------------------------------------------------- #
            _threshold_eV = _dataframe.at[i_rctn, 'threshold_eV']
            _phi, _n = math.modf(
                math.fabs(_threshold_eV) / (self.energy_intvl * const.J2eV))
            _n = int(_n)
            _dataframe.at[i_rctn, "low_threshold"] = True if _n == 0 else False
        self.inelas_reaction_dataframe = _dataframe
        # self.bg_molecule_inelas = np.unique(_dataframe["bg_molecule"]).tolist()
        self.bg_molecule_inelas_in_order = _dataframe["bg_molecule"].values
        self.bg_molecule_inelas = np.unique(self.bg_molecule_inelas_in_order)
        self._set_rate_const_matrix_e_inelas_electron()
        self._set_rate_const_matrix_e_inelas_molecule()

    def _set_flux_electric_field(self, *, total_species_density: np.ndarray):
        r"""
        Calculate the flux driven by electric field.
        Boundary condition:
            0 = J_flux[0] = J_flux[-1]

        Parameters
        ----------
        _density

        Returns
        -------
        SET :
        self.D_k_ef
            .F_k_ef
            .J_flux_ef

        Notes
        -----
        Details in CalEEDF.pdf

        """
        _density_bg_molecule_elas = self._index_bg_molecule_elas.dot(
            total_species_density)
        _mole_frac = _density_bg_molecule_elas / _density_bg_molecule_elas.sum()
        _const = const.e ** 2 / 3 * np.sqrt(
            2 / const.m_e / self.energy_nodes[1:-1])
        _variable = self.electric_field ** 2 / self.total_bg_molecule_density / \
                    (_mole_frac.dot(self.crostn_elas))[1:-1]
        # D F
        self.D_k_ef = np.empty_like(self.energy_nodes)
        self.D_k_ef[0] = self.D_k_ef[-1] = 0.0
        self.D_k_ef[1:-1] = _const * _variable * (
                self.energy_nodes[1:-1] / self.energy_intvl)
        self.F_k_ef = np.empty_like(self.energy_nodes)
        self.F_k_ef[0] = self.F_k_ef[-1] = 0.0
        self.F_k_ef[1:-1] = _const * _variable * (1 / 2)
        # J_flux
        self.J_flux_ef = np.empty_like(self.energy_nodes)
        lam_F = self.get_lam_F(D_k=self.D_k_ef, F_k=self.F_k_ef)
        self.J_flux_ef[0] = self.J_flux_ef[-1] = 0.0
        self.J_flux_ef[1:-1] = -(self.D_k_ef - lam_F)[
                                1:-1] * self.density_per_J[1:] + \
                               (self.D_k_ef + self.F_k_ef - lam_F)[
                               1:-1] * self.density_per_J[:-1]

    def _set_flux_elastic_colli(self, *, total_species_density: np.ndarray):
        r"""
        Calculate J_flux of elastic collisions basing on D and F.
        Boundary condition:
            0 = J_flux[0] = J_flux[-1]

        Parameters
        ----------
        total_species_density

        Returns
        -------
        SET :
        self.D_k_el
            .F_k_el
            .J_flux_el

        """
        _density_bg_molecule_elas = self._index_bg_molecule_elas.dot(
            total_species_density)
        _mole_frac = _density_bg_molecule_elas / _density_bg_molecule_elas.sum()
        _const = 2 * const.m_e * np.sqrt(
            2 * self.energy_nodes[1:-1] / const.m_e)
        _variable = const.kB * self.gas_temperature * self.total_bg_molecule_density * \
                    ((_mole_frac / self.bg_molecule_mass_elas).dot(
                        self.crostn_elas))[1:-1]
        self.D_k_el = np.empty_like(self.energy_nodes)
        self.D_k_el[0] = self.D_k_el[-1] = 0.0
        self.D_k_el[1:-1] = _const * _variable * (
                self.energy_nodes[1:-1] / self.energy_intvl)
        self.F_k_el = np.empty_like(self.energy_nodes)
        self.F_k_el[0] = self.F_k_el[-1] = 0.0
        self.F_k_el[1:-1] = _const * _variable * \
                            (.5 - self.energy_nodes[
                                  1:-1] / const.kB / self.gas_temperature)
        self.J_flux_el = np.empty_like(self.energy_nodes)
        lam_F = self.get_lam_F(D_k=self.D_k_el, F_k=self.F_k_el)
        self.J_flux_el[0] = self.J_flux_el[-1] = 0.0
        self.J_flux_el[1:-1] = -(self.D_k_el - lam_F)[
                                1:-1] * self.density_per_J[1:] + \
                               (self.D_k_el + self.F_k_el - lam_F)[
                               1:-1] * self.density_per_J[:-1]

    def _pre_set_flux_ee_colli(self):
        r"""
        op_ means operator on n.
            a_p = alpha * op_a_p * n
            a_m = alpha * op_a_m * n
            P = op_P * n
            Q = op_Q * n
        Notes
        -----
            op_P1, op_P2, op_Q : (N+1, N)

        Returns
        -------
        SET :
        self.op_P1
            .op_P2
            .op_P
            .op_Q
            .ee_op_a
            .ee_op_b

        """
        op_P1 = np.tril(np.tile(self.energy_point, (self.energy_nodes.size, 1)),
                        -1)
        op_P1[:, 0] = op_P1[:, 0] * 4 * sqrt(2) / 5
        _factor = 2 / np.sqrt(
            self.energy_nodes[1:][np.newaxis].transpose()) * self.energy_intvl
        _factor = np.vstack((0.0, _factor))
        op_P1 = _factor * op_P1

        # -------------------------------------------------------------------- #
        op_P2 = np.triu(np.tile(1 / np.sqrt(self.energy_point),
                                (self.energy_nodes.size, 1)), 0)
        op_P2[:, 0] = op_P2[:, 0] * 1
        _factor = 2 * self.energy_nodes[
            np.newaxis].transpose() * self.energy_intvl
        op_P2 = _factor * op_P2

        # -------------------------------------------------------------------- #
        op_Q = np.tril(
            np.ones((self.energy_nodes.size, self.energy_point.size)), -1)
        op_Q[:, 0] = op_Q[:, 0] * 2 * sqrt(2) / 3
        _factor = 3 / np.sqrt(
            self.energy_nodes[1:][np.newaxis].transpose()) * self.energy_intvl
        _factor = np.vstack((0.0, _factor))
        op_Q = _factor * op_Q

        # -------------------------------------------------------------------- #
        op_P = op_P1 + op_P2
        op_a = 1 / self.energy_intvl * (
                (1 / self.energy_intvl + 0.25 / self.energy_nodes[1:][
                    np.newaxis].transpose()) *
                op_P[1:, :] - 1 / 2 * op_Q[1:, :])
        # -------------------------------------------------------------------- #
        #   Consider the detailed balance
        #       A[i,j] = A[i,j]*A[j-1,i+1]*sqrt(e[j-1]*e[i+1])/sqrt(e[j]*e[i])
        # -------------------------------------------------------------------- #
        _sqrt_point = np.sqrt(self.energy_point)[np.newaxis]
        temp1 = op_a * _sqrt_point * _sqrt_point.transpose()
        temp2 = temp1.copy()
        temp2[:-1, 1:] = temp1[:-1, 1:].transpose()

        # -------------------------------------------------------------------- #
        ee_op_a = np.sqrt(temp1 * temp2) / _sqrt_point / _sqrt_point.transpose()
        ee_op_a[-1] = 0.0
        ee_op_a[:, 0] = 0.0
        ee_op_b = ee_op_a.copy().transpose()

        # -------------------------------------------------------------------- #
        self._op_P1 = op_P1
        self._op_P2 = op_P2
        self._op_P = self._op_P1 + self._op_P2
        self._op_Q = op_Q
        self.ee_op_a = ee_op_a
        self.ee_op_b = ee_op_b

    def _set_flux_ee_colli(self):
        r"""

        Returns
        -------
        SET :
        self.ee_alpha
            .J_flux_ee

        """
        _mean_energy = self.electron_mean_energy_in_J
        _density = self.electron_density
        _temp = 2 / 3 * (const.epsilon_0 * _mean_energy) ** 3 / _density

        assert _temp > 0, _temp
        L = 8 * pi / const.e ** 3 * sqrt(_temp)
        _const = 2 * pi / 3 * sqrt(2 / const.m_e) * const.e ** 4 / (
                4 * pi * const.epsilon_0) ** 2
        self.ee_alpha = _const * log(L)
        self.J_flux_ee = np.zeros_like(self.energy_nodes)
        ee_a = self.ee_op_a.dot(self.density_per_J)
        ee_b = self.ee_op_b.dot(self.density_per_J)
        positive_term = ee_a * self.density_per_J
        negative_term = ee_b * self.density_per_J
        self.J_flux_ee[1:-1] = self.ee_alpha * (positive_term[:-1]
                                                - negative_term[
                                                  1:]) * self.energy_intvl

    def _set_index_bg_molecule(self, *, species_list: list):
        r"""
        Set _index_bg_molecule_elas.
            _index_bg_molecule_inelas.

        Parameters
        ----------
        total_species : list

        Notes
        -----
            _index[mxn] * density_species[n*1] = density_bg_molecule[m*1]
                      A * n                    = m

        """
        assert isinstance(species_list, list)

        def set_index(_bg_molecule, species_list):
            assert set(_bg_molecule) <= set(species_list)
            _series = pd.Series(data=range(len(species_list)),
                                index=species_list)
            _index = np.zeros((len(_bg_molecule), len(species_list)))
            for i_molecule in range(len(_bg_molecule)):
                _index[i_molecule, _series[_bg_molecule[i_molecule]]] = 1
            return sprs.csr_matrix(_index, shape=_index.shape)

        self._index_bg_molecule_elas = set_index(self.bg_molecule_elas,
                                                 species_list)
        self._index_bg_molecule_inelas = set_index(self.bg_molecule_inelas,
                                                   species_list)
        self._index_bg_molecule_inelas_in_order = set_index(
            self.bg_molecule_inelas_in_order,
            species_list)

    def _set_rate_const_matrix_e_inelas_electron(self):
        r"""
        E(i) + N => E(j) + N'

        dn1 / dt
        dn2 / dt
            ...
        dni / dt = N * Kii * fi
            ...

        N:   1 x m
        Kii: (m x n) x n
        fi:  n x 1

        dnidt: 1 x n

        Returns
        -------
        Kii : ndarray
            m x n x n shape

        Notes
        -----

        """
        _shape = self.grid_number
        _rate_const_matrix = np.empty((self.inelas_reaction_dataframe.shape[0],
                                       _shape, _shape))
        for i_rctn in self.inelas_reaction_dataframe.index:
            _reaction_type = self.inelas_reaction_dataframe.at[i_rctn, 'type']
            _threshold_eV = self.inelas_reaction_dataframe.at[
                i_rctn, 'threshold_eV']
            _crostn = self.inelas_reaction_dataframe.at[i_rctn, 'cross_section']
            _op = self.get_rate_const_matrix_electron(
                reaction_type=_reaction_type,
                energy_grid_J=self.energy_point,
                threshold_eV=_threshold_eV,
                crostn_eV_m2=_crostn)
            _rate_const_matrix[i_rctn] = _op
        # -------------------------------------------------------------------- #
        #   merge
        # -------------------------------------------------------------------- #
        bg_molecule = self.inelas_reaction_dataframe['bg_molecule'].values
        _matrix_new = np.empty((len(self.bg_molecule_inelas), _shape, _shape))
        for i_index, i_molecule in enumerate(self.bg_molecule_inelas):
            _matrix_new[i_index] = _rate_const_matrix[
                bg_molecule == i_molecule].sum(axis=0)

        # -------------------------------------------------------------------- #
        #   to sparse
        # -------------------------------------------------------------------- #
        _matrix_new = sprs.csr_matrix(_matrix_new.reshape(-1, self.grid_number))
        self.rate_const_matrix_e_inelas_electron = _matrix_new

    def _set_rate_const_matrix_e_inelas_molecule(self):
        r"""
        E(i) + N => E(j) + N'

        dN / dt = N * Ki * fi

        Returns
        -------
        Ki : ndarray
            m x n shape

        """
        _shape = self.grid_number
        _rate_const_matrix = np.empty(
            (self.inelas_reaction_dataframe.shape[0], _shape))

        for i_rctn in self.inelas_reaction_dataframe.index:
            _reaction_type = self.inelas_reaction_dataframe.at[i_rctn, 'type']
            _threshold_eV = self.inelas_reaction_dataframe.at[
                i_rctn, 'threshold_eV']
            _crostn = self.inelas_reaction_dataframe.at[i_rctn, 'cross_section']
            _op = self.get_rate_const_matrix_molecule(
                reaction_type=_reaction_type,
                energy_grid_J=self.energy_point,
                threshold_eV=_threshold_eV,
                crostn_eV_m2=_crostn)
            _rate_const_matrix[i_rctn] = _op
        # -------------------------------------------------------------------- #
        # merge
        #   it should not be merged.
        # -------------------------------------------------------------------- #
        # bg_molecule = self.inelas_reaction_dataframe['bg_molecule'].values
        # _matrix_new = np.empty((len(self.bg_molecule_inelas), _shape))
        # for i_index, i_molecule in enumerate(self.bg_molecule_inelas):
        #     _matrix_new[i_index] = _rate_const_matrix[bg_molecule == i_molecule].sum(axis=0)

        # self.rate_const_matrix_e_inelas_molecule = _matrix_new
        self.rate_const_matrix_e_inelas_molecule = _rate_const_matrix

    def _get_electron_rate_e_inelas(self, *, density):
        r"""
        dn1 / dt
        dn2 / dt
            ...
        dni / dt = N * Kii * fi
            ...

        Returns
        -------

        """
        _density_bg_molecule = self._index_bg_molecule_inelas.dot(density)
        _temp = self.rate_const_matrix_e_inelas_electron.dot(self.density_per_J)
        dnidt = _density_bg_molecule.dot(_temp.reshape(-1, self.grid_number))
        return dnidt

    def _get_molecule_rate_const_e_inelas(self):
        _temp = self.rate_const_matrix_e_inelas_molecule.dot(self.density_per_J)
        return _temp / self.electron_density

    def _get_molecule_rate_e_inelas(self, *, density):
        r"""

        Parameters
        ----------
        density

        Returns
        -------

        """
        # _density_bg_molecule = self._index_bg_molecule_inelas.dot(density)
        _density_bg_molecule_in_order = self._index_bg_molecule_inelas_in_order.dot(
            density)
        _rate_const = self._get_molecule_rate_const_e_inelas()
        dNdt = _density_bg_molecule_in_order * _rate_const * self.electron_density
        return dNdt

    def _get_deriv_ef(self):
        return -(self.J_flux_ef[1:] - self.J_flux_ef[:-1]) / self.energy_intvl

    def _get_deriv_el(self):
        return -(self.J_flux_el[1:] - self.J_flux_el[:-1]) / self.energy_intvl

    def _get_deriv_ee(self):
        return -(self.J_flux_ee[1:] - self.J_flux_ee[:-1]) / self.energy_intvl

    @staticmethod
    def get_lam_F(*, D_k, F_k):
        r"""
        power-law method

        Parameters
        ----------
        D_k :
        F_k

        Returns
        -------
        lambda * F

        """
        with np.errstate(divide='ignore', invalid='ignore'):
            Pe_k = np.divide(F_k, D_k)
            Pe_k[np.isinf(Pe_k) | np.isnan(Pe_k)] = 0.0
        return D_k + D_k * np.minimum(0.0, (
                0.1 * np.abs(Pe_k) - 1) ** 5) + np.minimum(0.0, F_k)

    @staticmethod
    def get_rate_const_matrix_electron(*, reaction_type, energy_grid_J,
                                       threshold_eV, crostn_eV_m2):
        r"""
        Get the rate constant matrix.
            d(ni)/dt = N * Kii * fi
            where,
                Kii = gamma * crostn * sqrt(energy) * de
                gamma = sqrt(2 / m_e)

        Parameters
        ----------
        reaction_type : str
            'excitation', 'deexcitation', 'ionization', 'attachment'
        energy_grid_J : ndarray
        threshold_eV : float
        crostn_eV_m2 : ndarray

        Returns
        -------
        Kii : ndarray
            n x n shape

        """
        #   check reaction_type
        assert reaction_type.lower() in ('excitation', 'deexcitation',
                                         'ionization',
                                         'attachment'), reaction_type
        #   check energy_grid_J
        assert isinstance(energy_grid_J, np.ndarray)
        assert energy_grid_J.ndim == 1
        #   check crostn_eV_m2
        assert isinstance(crostn_eV_m2, np.ndarray)
        assert crostn_eV_m2.ndim == 2
        assert crostn_eV_m2.shape[0] == 2
        #   check threshold_eV
        assert isinstance(threshold_eV, (float, int, np.number))
        assert 0 <= math.fabs(threshold_eV) * const.eV2J < energy_grid_J[-1] + \
               energy_grid_J[0], \
            '{}'.format(threshold_eV)
        if reaction_type.lower() in ('excitation', 'ionization'):
            assert threshold_eV > 0.0
        elif reaction_type.lower() in ('deexcitation',):
            assert threshold_eV < 0.0
        elif reaction_type.lower() in ('attachment',):
            assert threshold_eV >= 0.0
        else:
            raise EEDFerror('The threshold_eV is error.')
        # -------------------------------------------------------------------- #
        #   set n, phi, low_threshold
        #       threshold = (_n + _phi) * de
        #       0 < _phi < 1
        #       n is an integer
        # -------------------------------------------------------------------- #
        _gamma = math.sqrt(2 / const.m_e)
        grid_number = energy_grid_J.size
        de = energy_grid_J[1] - energy_grid_J[0]
        _phi, _n = math.modf(math.fabs(threshold_eV) / (de * const.J2eV))
        _n = int(_n)
        assert 0 <= _phi < 1
        assert 0 <= _n < grid_number
        low_threshold = True if _n == 0 else False
        # -------------------------------------------------------------------- #
        #   set discretized energy and cross section.
        # -------------------------------------------------------------------- #
        _energy = np.hstack((0.0, crostn_eV_m2[0], np.inf))
        _crostn = np.hstack((0.0, crostn_eV_m2[1], 0.0))
        # -------------------------------------------------------------------- #
        _energy_discretized = energy_grid_J * const.J2eV
        # -------------------------------------------------------------------- #
        #   set discretized cross section
        # -------------------------------------------------------------------- #
        _crostn_func = interp1d(_energy, _crostn)
        _crostn_discretized = np.zeros_like(_energy_discretized)
        _crostn_discretized[_n] = _crostn_func(
            _energy_discretized[_n] + 0.5 * _phi * de)
        _crostn_discretized[_n + 1:] = _crostn_func(
            _energy_discretized[_n + 1:])
        # print(_crostn_discretized)
        # -------------------------------------------------------------------- #
        _shape = grid_number
        _op = None
        if low_threshold:
            # print("low threshold")
            # print(reaction_type)
            # ---------------------------------------------------------------- #
            #   low threshold case
            # ---------------------------------------------------------------- #
            if reaction_type.lower() == 'excitation':
                _diags = np.vstack((-1 * np.ones(_shape), +1 * np.ones(_shape)))
                _op = sprs.spdiags(_diags, [0, 1], _shape, _shape).toarray()
                _op[0, 0] = 1 * (1 - _phi)
                _op[1, 0] = -1 * (1 - _phi)
                _op = _op * _phi * _gamma * _crostn_discretized * np.sqrt(
                    energy_grid_J)
            elif reaction_type.lower() == 'deexcitation':
                _diags = np.vstack((-1 * np.ones(_shape), +1 * np.ones(_shape)))
                _op = sprs.spdiags(_diags, [0, -1], _shape, _shape).toarray()
                _op[-1, -1] = 0
                _op = _op * _phi * _gamma * _crostn_discretized * np.sqrt(
                    energy_grid_J)
            elif reaction_type.lower() == 'ionization':
                raise EEDFerror(
                    'The ionization in low_threshold mode should be avoid.')
            elif reaction_type.lower() == 'attachment':
                _op = sprs.spdiags(-1 * np.ones(_shape), [0], _shape,
                                   _shape).toarray()
                _op = _op * _gamma * _crostn_discretized * np.sqrt(
                    energy_grid_J)  # TODO check _phi
            else:
                raise EEDFerror(
                    'The reaction_type {} is error.'.format(reaction_type))
            # ---------------------------------------------------------------- #
            #   high threshold case
            # ---------------------------------------------------------------- #
        else:
            # print("high threshold")
            # print(reaction_type)
            if reaction_type.lower() == 'excitation':
                _diag = np.zeros(_shape)
                _diag[:_n] = 0.0
                _diag[_n] = -(1 - _phi)
                _diag[_n + 1:] = -1
                _diags = np.vstack((_diag,
                                    (1 - _phi) * np.ones(_shape),
                                    _phi * np.ones(_shape)))
                _op = sprs.spdiags(_diags, [0, _n, _n + 1], _shape,
                                   _shape).toarray()
                # print(_op)
                # print(_gamma)
                # print(_crostn_discretized)
                _op = _op * _gamma * _crostn_discretized * np.sqrt(
                    energy_grid_J)
            elif reaction_type.lower() == 'ionization':
                _diag = np.zeros(_shape)
                _diag[:_n] = 0.0
                _diag[_n] = -(1 - _phi)
                _diag[_n + 1:] = -1
                _diags = np.vstack((_diag,
                                    (1 - _phi) * np.ones(_shape),
                                    _phi * np.ones(_shape)))
                _op = sprs.spdiags(_diags, [0, _n, _n + 1], _shape,
                                   _shape).toarray()
                _op_extra_e = EEDF.get_rate_const_matrix_molecule(
                    reaction_type='ionization',
                    energy_grid_J=energy_grid_J,
                    threshold_eV=threshold_eV,
                    crostn_eV_m2=crostn_eV_m2)
                _op = _op * _gamma * _crostn_discretized * np.sqrt(
                    energy_grid_J)
                _op[0] = _op[0] + 3 / 2 / de * _op_extra_e
                _op[1] = _op[1] - 1 / 2 / de * _op_extra_e
            elif reaction_type.lower() == 'deexcitation':
                _diags = np.vstack(((-1) * np.ones(_shape - _n - 1),
                                    (1 - _phi) * np.ones(_shape - _n - 1),
                                    (_phi) * np.ones(_shape - _n - 1)))
                _op = sprs.spdiags(_diags, [0, -_n, -_n - 1], _shape,
                                   _shape).toarray()
                _op = _op * _gamma * _crostn_discretized * np.sqrt(
                    energy_grid_J)
            elif reaction_type.lower() == 'attachment':
                _op = sprs.spdiags(-1 * np.ones(_shape), [0], _shape,
                                   _shape).toarray()
                _op = _op * _gamma * _crostn_discretized * np.sqrt(
                    energy_grid_J)
            else:
                raise EEDFerror(f"The reaction_type {reaction_type} is error.")
        return _op

    @staticmethod
    def get_rate_const_matrix_molecule(*, reaction_type, energy_grid_J,
                                       threshold_eV, crostn_eV_m2):
        r"""
        Set rate constant matrix of molecule.
            dNdt = N * Ki * fi

        Parameters
        ----------
        reaction_type : str
        energy_grid_J
        threshold_eV
        crostn_eV_m2

        Returns
        -------
        Ki : ndarray

        """
        #   check reaction_type
        assert reaction_type.lower() in ('excitation', 'deexcitation',
                                         'ionization',
                                         'attachment'), reaction_type
        #   check energy_grid_J
        assert isinstance(energy_grid_J, np.ndarray)
        assert energy_grid_J.ndim == 1
        #   check crostn_eV_m2
        assert isinstance(crostn_eV_m2, np.ndarray)
        assert crostn_eV_m2.ndim == 2
        assert crostn_eV_m2.shape[0] == 2
        #   check threshold_eV
        assert isinstance(threshold_eV, (int, float, np.number))
        assert 0 <= math.fabs(threshold_eV) * const.eV2J < energy_grid_J[-1] + \
               energy_grid_J[0], \
            '{}'.format(threshold_eV)
        if reaction_type.lower() in ('excitation', 'ionization'):
            assert threshold_eV > 0.0, threshold_eV
        elif reaction_type.lower() in ('deexcitation',):
            assert threshold_eV < 0.0
        elif reaction_type.lower() in ('attachment',):
            assert threshold_eV >= 0.0
        else:
            raise EEDFerror(f"The threshold_eV {threshold_eV} is error.")

        # ------------------------------------------------- #
        _gamma = math.sqrt(2 / const.m_e)
        grid_number = energy_grid_J.size

        de = energy_grid_J[1] - energy_grid_J[0]
        _phi, _n = math.modf(math.fabs(threshold_eV) / (de * const.J2eV))
        _n = int(_n)
        assert 0 <= _phi < 1, 'phi = {i}'.format(i=_phi)
        assert 0 <= _n < grid_number
        low_threshold = True if _n == 0 else False

        _energy = np.hstack((0.0, crostn_eV_m2[0], np.inf))
        _crostn = np.hstack((0.0, crostn_eV_m2[1], 0.0))
        _energy_discretized = energy_grid_J.copy()
        _energy_discretized[_n] = _energy_discretized[_n] + 0.5 * _phi * de
        _crostn_discretized = interp1d(_energy, _crostn)(
            _energy_discretized * const.J2eV)

        _shape = grid_number

        _op = np.zeros(_shape)
        if low_threshold:
            if reaction_type.lower() == 'excitation':
                _op[0] = 1.0 * (1 - _phi)
                _op[1:] = 1.0
            elif reaction_type.lower() == 'deexcitation':
                _op[:-1] = 1.0
            elif reaction_type.lower() == 'attachment':
                _op[:] = 1.0
            elif reaction_type.lower() == 'ionization':
                raise EEDFerror(
                    'The ionization in low_threshold mode should be avoid.')
            else:
                raise EEDFerror(
                    'The reaction_type {} is error.'.format(reaction_type))
        else:
            if reaction_type.lower() in ('excitation', 'ionization'):
                _op[:_n] = 0.0
                _op[_n] = (1 - _phi) * _n / (_n + _phi)
                _op[_n + 1:] = 1.0
            elif reaction_type.lower() == 'attachment':
                _op[:] = 1.0
            elif reaction_type.lower() == 'deexcitation':
                _op[:(grid_number - _n - 1)] = 1.0
            else:
                raise EEDFerror(
                    'The reaction_type {} is error.'.format(reaction_type))
        _op = _op * _gamma * _crostn_discretized * np.sqrt(energy_grid_J) * de
        return _op

    def plot_normalized_eedf(self):
        plt.figure()
        plt.plot(self.energy_point_eV, self.normalized_eedf_eV, marker=".")
        plt.xlabel("energy (eV)")
        plt.ylabel("eV^-1")
        plt.grid()

    def plot_normalized_eepf(self, xlim=None, ylim=(1e-12, 1e1)):
        plt.figure()
        plt.semilogy(self.energy_point_eV, self.normalized_eepf_eV, marker=".")
        plt.xlabel("energy (eV)")
        plt.ylabel("eV^-3/2")
        plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        plt.grid()

    # ------------------------------------------------------------------------ #
    def __str__(self):
        # _temp = self.inelas_reaction_dataframe[['reaction',
        #                                         'bg_molecule',
        #                                         'type',
        #                                         'threshold_eV',
        #                                         ]].copy()
        # _text = '''
        # \n INELASTIC REACTIONS :
        # \n {inelas_reaction_info}
        # '''.format(temperature=self.electron_temperature * const.K2eV,
        #            density=self.electron_density * 1e-6,
        #            bg_molecule_density=self.total_bg_molecule_density * 1e-6,
        #            electric_field=self.electric_field,
        #            gas_temperature=self.gas_temperature,
        #            reduced_electric_field=self.reduced_electric_field,
        #            cell_number=self.grid_number,
        #            energy_space=self.energy_max_bound * const.J2eV,
        #            interval=self.energy_intvl * const.J2eV,
        #            elastic_molecules=self.bg_molecule_elas,
        #            inelas_reaction_info=_temp)
        # \n BACKGROUND MOLECULE DENSITY (N) : {self.total_bg_molecule_density:.2e} m^-3"""
        _text = f"""
        \n ===============
        \n               ENERGY SPACE (eV) : (0.0, {self.energy_max_bound * const.J2eV:.1f})
        \n     NUMBER OF DISCRETIZED CELLS : {self.grid_number} cells
        \n            ENERGY INTERVAL (eV) : {self.energy_intvl * const.J2eV:.3f} 
        \n ===============
        \n            ELECTRON TEMPERATURE : {self.electron_temperature_in_eV:.4f} eV
        \n            ELECTRON MEAN ENERGY : {self.electron_mean_energy_in_eV:.4f} eV
        \n                ELECTRON DENSITY : {self.electron_density:.2e} m^-3
        \n ===============
        \n              ELECTRIC FIELD (E) : {self.electric_field:.0f} V/m ({self.electric_field / 1e5:.2f} kV/cm) 
        \n          GAS TEMPERATURE (Tgas) : {self.gas_temperature:.0f} K
        \n BACKGROUND MOLECULE DENSITY (N) : {self.total_bg_molecule_density:.1e} m^-3
        \n    REDUCED ELECTRIC FIELD (E/N) : {self.reduced_electric_field_in_Td:.1f} Td
        \n ===============
        \n     ELASTIC COLLISION MOLECULES : {self.bg_molecule_elas}
        \n   INELASTIC COLLISION MOLECULES : {self.bg_molecule_inelas}
        """
        return _text


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    pass
