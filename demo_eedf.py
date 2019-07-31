#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  9:49 2019/7/17

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

import numpy as np
import pandas as pd
from plasmistry.electron import EEDF
from plasmistry import constants as const
from plasmistry.electron import get_maxwell_eedf
# from scipy.integrate import ode as ode_ivp
from plasmistry.solvers import ode_ivp
from plasmistry.molecule import get_ideal_gas_density

inelas_df = pd.read_pickle("plasmistry/electron/tests/e.g._inelas_colli_dataframe.pkl")
species = ['CO2', 'CO2(v0)', 'CO2(v1)', 'CO2(vc)', 'CO', 'O2']
# ----------------------------------------------------------------------------------------------- #
eedf = EEDF(max_energy_J=10 * const.eV2J, grid_number=100)
eedf.set_density_in_J(get_maxwell_eedf(eedf.energy_point, Te_eV=1.0))
eedf.initialize(rctn_with_crostn_df=inelas_df,
                total_species=species)
eedf.set_parameters(E=1, Tgas=1000, N=1e20)
total_species_density = get_ideal_gas_density(p_Pa=1e5, Tgas_K=2000)
eedf.set_flux(total_species_density=total_species_density * np.array([1, 0, 0, 0, 0, 0]))


# eedf.set_flux(total_species_density=)

def dndt(t, y, _eedf):
    _eedf.density_in_J = y
    return _eedf.get_deriv_total()


# y_0 = np.zeros_like(eedf.energy_point)
# y_0[(eedf.energy_point < 5.5 * const.eV2J) & (eedf.energy_point > 0.5 * const.eV2J)] = \
#     1e16 / 5.0 / const.eV2J
# eedf.density_in_J = y_0
# density_0 = eedf.electron_density
# Te_0 = eedf.electron_temperature
y_0 = get_maxwell_eedf(eedf.energy_point, Te_eV=0.026)
sol = ode_ivp(deriv_func=dndt,
              func_args=(eedf,),
              time_span=(0, 1e5),
              y_0=y_0,
              rtol=1e-6,
              atol=1e-12,
              show_time=False)
# plt.plot(eedf.energy_point, a.transpose())

# eedf.density_in_J = a[-1]
# eedf.set_density_in_J(sol.y[-1])
