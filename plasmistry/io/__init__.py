#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2:11 2017/7/4

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import
import copy
from pandas.core.frame import DataFrame as DataFrame_type
from .io_cross_section import *
from .io_reactions import *
from .pre_treat import treat_lines


# %%--------------------------------------------------------------------------------------------- #
def combine_crostn_reaction_dataframe(*, crostn_dataframe, reactn_dataframe):
    r"""
    Combine crostn_dataframe and reactn_dataframe.

    Parameters
    ----------
    crostn_dataframe : dataframe
        columns : cs_key    type    thres_info  cross_section

    reactn_dataframe
        columns : reactant  product dH  k_str

    Returns
    -------
        reaction    reactant    product type    threshold_eV    cs_key  dH_e    cross_section

    Notes
    -----
    k_str : cs_sign + ' ' + cs_key

    Examples
    --------
    crostn_dataframe
        cs_key          type      thres_info  cross_section
    1   a->b    dissocaition        13.0        [[1, 2, 3], [2, 0, 1]]
    2   b->c      attachment        12.0        [[2, 3, 4], [1, 0, 3]]
    3   d->a      ionization        14.1        [[3, 4, 5], [2, 1, 3]]
    reactn_dataframe
        reactant    product      dH      k_str
    1   e + a         e + d      -1     BOLSIG a->b
    2   e + d         e + c       4     BOLSIG b->c
    3   e + c         e + a      -4     BOLSIG d->a

    """
    assert isinstance(crostn_dataframe, DataFrame_type)
    assert isinstance(reactn_dataframe, DataFrame_type)
    assert set(crostn_dataframe.columns) >= {'cs_key', 'type', 'thres_info', 'cross_section'}
    assert set(reactn_dataframe.columns) >= {'reactant', 'product', 'dH', 'k_str'}

    _dataframe = pd.DataFrame(columns=['reaction', 'reactant', 'product', 'type',
                                       'threshold_eV', 'cs_key', 'dH_e', 'cross_section'])
    for i_rctn in reactn_dataframe.index:
        _temp = dict()
        _temp['cs_key'] = reactn_dataframe.at[i_rctn,
                                              'k_str'].split(maxsplit=1)[1].replace(' ', '')
        assert _temp['cs_key'] in crostn_dataframe['cs_key'].values, \
            "The '{}' is not in cross section dataframe.".format(_temp['cs_key'])
        crostn_series = crostn_dataframe[crostn_dataframe['cs_key'] == _temp['cs_key']]
        crostn_series = crostn_series.reset_index(drop=True).loc[0]
        _temp['reactant'] = reactn_dataframe.at[i_rctn, 'reactant']
        _temp['product'] = reactn_dataframe.at[i_rctn, 'product']
        _temp['reaction'] = '{rc} => {pr}'.format(rc=_temp['reactant'], pr=_temp['product'])
        _temp['dH_e'] = reactn_dataframe.at[i_rctn, 'dH']
        _temp['type'] = crostn_series['type']
        if crostn_series['thres_info'] == '':
            _temp['threshold_eV'] = 0.0
        else:
            _temp['threshold_eV'] = float(crostn_series['thres_info'].split(maxsplit=1)[0])
        _temp['cross_section'] = crostn_series['cross_section']
        _dataframe = _dataframe.append(pd.Series(_temp), ignore_index=True)
    return _dataframe


def split_reaction_dataframe(reaction_dataframe, *, cs_sign):
    r"""
    Split reaction dataframe by the cs_sign.

    Parameters
    ----------
    reaction_dataframe : dataframe
        columns : reactant, product, dH, k_str

    cs_sign : str

    Returns
    -------

    """
    assert isinstance(reaction_dataframe, DataFrame_type)
    assert 'k_str' in reaction_dataframe.columns

    _boolean = reaction_dataframe['k_str'].str.startswith(cs_sign)
    cros_dataframe = reaction_dataframe.loc[_boolean].reset_index(drop=True)
    coef_dataframe = reaction_dataframe.loc[~_boolean].reset_index(drop=True)
    assert cros_dataframe.size > 0
    # assert coef_dataframe.size > 0
    return copy.deepcopy(cros_dataframe), copy.deepcopy(coef_dataframe)


# %%--------------------------------------------------------------------------------------------- #
__all__ = [s for s in dir() if not s.startswith('_')]

from numpy.testing import Tester

test = Tester().test
