#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23:55 2017/7/14

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from pandas.core.series import Series as Series_type
import pandas as pd
import re
from .. import constants as const


# %%--------------------------------------------------------------------------------------------- #
def get_reverse_crostn(cs_series, stcs_wght_ratio_dict):
    r"""
    Calculate reverse cross section of reaction basing on detailed balance law.

    Parameters
    ----------
    cs_series : series
        Cross section series
        Index : type,
                thres_info,
                info_dict,
                energy,
                energy_range,
                crostn

    stcs_wght_ratio_dict : dict
        { molecule : statics weight }

    Returns
    -------
    output : series
        Same with cs_series

    """
    assert isinstance(cs_series, Series_type)
    assert isinstance(stcs_wght_ratio_dict, dict)
    assert '->' in cs_series.name
    assert cs_series['type'] != 'ATTACHMENT'

    output = pd.Series(index=['type',
                              'thres_info',
                              'info_dict',
                              'energy',
                              'energy_range',
                              'crostn'])
    _rcnt, _prdt = re.split(r"\s*[-][>]\s*", cs_series.name.strip())
    output.name = "{rcnt}->{prdt}".format(rcnt=_prdt, prdt=_rcnt)
    # %%----------------------------------------------------------------------------------------- #
    #   reverse's type is same?
    # %%----------------------------------------------------------------------------------------- #
    output['type'] = cs_series['type']
    output['thres_info'] = '-{}'.format(cs_series['thres_info'].strip())
    thres_energy = float(cs_series['thres_info'].strip().split(maxsplit=1)[0])
    stcs_wght_ratio = stcs_wght_ratio_dict.get(_rcnt, 1.0) / stcs_wght_ratio_dict.get(_prdt, 1.0)
    energy_0 = cs_series['energy'] - thres_energy
    crostn_0 = cs_series['crostn'] * stcs_wght_ratio * cs_series['energy']
    energy = energy_0[energy_0 > 0.0]
    crostn = crostn_0[energy_0 > 0.0] / energy
    output['energy'] = energy
    output['energy_range'] = (energy.min(), energy.max())
    output['crostn'] = crostn
    output['info_dict'] = cs_series['info_dict']
    output['info_dict']['COMMENT'] = "Reversed cross section from '{}'.".format(cs_series.name)

    return output


def arrhenius_rate_const(*, T, A, n, Ea_J_mol):
    return A * T ** n * np.exp(-Ea_J_mol / const.R / T)


def pressure_reaction_const(*, T, third_body_density, low_coefs, high_coefs):
    k_0 = arrhenius_rate_const(T=T, A=low_coefs[0], n=low_coefs[1], Ea_J_mol=low_coefs[2])
    k_oo = arrhenius_rate_const(T=T, A=high_coefs[0], n=high_coefs[1], Ea_J_mol=high_coefs[2])
    Pr = k_0 * third_body_density / k_oo


def get_reversed_arrhenius_const(*, reaction_str, A, n, Ea_J_mol):
    tmpr_seq = np.linspace(200, 6000, num=50)
    k_f = arrhenius_rate_const(T=tmpr_seq, A=A, n=n, Ea_J_mol=Ea_J_mol)
    K_p_seq = get_equilibrium_const(reaction_str=reaction_str, temperature_seq=tmpr_seq)
    k_r = k_f / K_p_seq
    return k_r
# %%--------------------------------------------------------------------------------------------- #
