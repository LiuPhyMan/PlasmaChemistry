#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9:13 2018/3/21

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import

from .reactions import CrosReactions
from .electron import EEDF
from . import constants as const


class tracer_func:
    def __init__(self, func):
        self.calls = 0
        self.func = func

    def __call__(self, *args, **kwargs):
        self.calls += 1
        # print(self.calls)
        return self.func(*args, **kwargs)


def create_eedf_and_cros_reactions(*, rctn_with_crostn_df, max_energy_eV, grid_number):
    _cros_reactions = CrosReactions(reactant=rctn_with_crostn_df['reactant'],
                                    product=rctn_with_crostn_df['product'],
                                    k_str=rctn_with_crostn_df['cs_key'],
                                    dH_e=rctn_with_crostn_df['threshold_eV'])

    #   create eedf
    _eedf = EEDF(max_energy_J=max_energy_eV * const.eV2J, grid_number=grid_number)
    _eedf.initialize(rctn_with_crostn_df=rctn_with_crostn_df,
                     total_species=_cros_reactions.species.tolist())

    #   associate eedf and cros reactions
    _cros_reactions.set_rate_const_matrix(crostn_dataframe=rctn_with_crostn_df,
                                          electron_energy_grid=_eedf.energy_point)

    return _eedf, _cros_reactions


from numpy.testing import Tester

test = Tester().test
