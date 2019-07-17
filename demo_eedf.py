#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  9:49 2019/7/17

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

import pandas as pd
from plasmistry.electron import EEDF
from plasmistry import constants as const
from plasmistry.electron import get_maxwell_eedf

eedf = EEDF(max_energy_J=10*const.eV2J, grid_number=100)
eedf.set_density_in_J(get_maxwell_eedf(eedf.energy_point, Te_eV=1.0))
inelas_df = pd.read_pickle("plasmistry/electron/tests/e.g._inelas_colli_dataframe.pkl")
species = ['CO2', 'CO2(v0)', 'CO2(v1)', 'CO2(vc)', 'CO', 'O2']
eedf.initialize(rctn_with_crostn_df=inelas_df,
                total_species=species)
eedf.set_parameters(E=1, Tgas=1000, N=1e20)
# eedf.set_flux(total_species_density=)