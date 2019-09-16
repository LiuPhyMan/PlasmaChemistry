#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  15:17 2019/7/10

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

import re
import yaml
import pandas as pd
from math import exp
import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
from plasmistry.reactions import CrosReactions
from plasmistry.molecule import get_ideal_gas_density
from plasmistry.solvers import ode_ivp
from plasmistry.molecule import get_vib_energy
from plasmistry.electron import get_maxwell_eedf
from plasmistry.electron import EEDF
from plasmistry.io import Reaction_block, Cros_Reaction_block, Coef_Reaction_block
from plasmistry import constants as const


def eval_constructor(loader, node):
    _str = loader.construct_scalar(node)
    return eval(_str)


def LT_constructor(loader, node):
    _list = loader.construct_sequence(node)
    A, B, C = _list
    return f"({A})*exp(({B})*Tgas**(-1/3)+({C})*Tgas**(-2/3))"


def Arr_constructor(loader, node):
    _list = loader.construct_sequence(node)
    A, b, E = _list
    return f"({A})*Tgas**({b})*exp(-({E})/Tgas)"


if __name__ == "__main__":
    # yaml.add_constructor(u"!CO2", CO2_energy_constructor)
    yaml.add_constructor(u"!eval", eval_constructor)
    yaml.add_constructor(u"!LT", LT_constructor)
    yaml.add_constructor(u"!Arr", Arr_constructor)
    with open("test_0.yaml") as f:
        temp = yaml.load(f)

    ele_rctn_block_list = temp[-1]["The reactions considered"]["electron reactions"]
    rctn_block_list = ele_rctn_block_list['CO2_VT_with_CO2']
    rctn_block = Reaction_block(rctn_dict=rctn_block_list)
    # rctn_block_list = ele_rctn_block_list["CO2_ele_vib_rctn_forward"]
    # rctn_block = Cros_Reaction_block(rctn_dict=rctn_block_list)
    # 0.5 * (3 - 2 / 3 * exp(1)) * exp(-2 / 3 * 1)
    r"""
    rctn_block_list = ele_rctn_block_list["H2_ele_dis_rctn_via_b"]
    rctn_block_0 = Cros_Reaction_block(rctn_dict=ele_rctn_block_list["H2_ele_vib_rctn_forward"])
    rctn_block_1 = Cros_Reaction_block(rctn_dict=ele_rctn_block_list["H2_ele_vib_rctn_backward"])
    rctn_block = rctn_block_0 + rctn_block_1
    crostn_dataframe = rctn_block.generate_crostn_dataframe(factor=1e-20)
    crostn_dataframe["reaction"] = crostn_dataframe["cs_key"]
    # ------------------------------------------------------------------------------------------- #
    eedf = EEDF(max_energy_J=10 * const.eV2J,
                grid_number=2000)
    electron_energy_grid = eedf.energy_point
    # electron_energy_grid = electron_energy_grid[1:] * const.eV2J
    rctn = CrosReactions(reactant=rctn_block._reactant_str_list,
                         product=rctn_block._product_str_list,
                         k_str=pd.Series(rctn_block._kstr_list),
                         dH_e=pd.Series(rctn_block._threshold_list))
    rctn.set_rate_const_matrix(crostn_dataframe=crostn_dataframe,
                               electron_energy_grid=electron_energy_grid)
    rctn.set_rate_const(eedf_normalized=get_maxwell_eedf(electron_energy_grid, Te_eV=1.0))
    # ------------------------------------------------------------------------------------------- #
    eedf.set_density_in_J(1e14 * get_maxwell_eedf(eedf.energy_point, Te_eV=1.0))
    eedf.initialize(rctn_with_crostn_df=crostn_dataframe,
                    total_species=rctn.species.to_list())
    Tgas = 2000
    Electric_field = 3500 / 0.01
    eedf.set_parameters(E=Electric_field,
                        Tgas=Tgas,
                        N=get_ideal_gas_density(p_Pa=1e5, Tgas_K=Tgas))
    # total_species_density = np.zeros(16)
    # total_species_density[1] = get_ideal_gas_density(p_Pa=1e5, Tgas_K=2000)
    # total_species_density[0] = 1e20
    total_species_density = np.ones(rctn.species.size) / rctn.species.size \
                            * get_ideal_gas_density(p_Pa=1e5, Tgas_K=2000)
    total_species_density[0] = 1e20
    eedf.set_flux(total_species_density=total_species_density)


    # eedf.set_flux()

    def dndt(t, y, _eedf):
        _eedf.density_in_J = y
        return _eedf.get_deriv_total(total_species_density=total_species_density)


    y_0 = 1e14 * get_maxwell_eedf(eedf.energy_point, Te_eV=1.3)
    solver = ode(dndt)
    solver.set_integrator(name='vode', method='bdf', with_jacobian=True)
    solver.set_f_params(eedf)
    solver.set_initial_value(y_0, t=0)
    # ------------------------------------------------------------------------------------------- #
    time_seq = []
    time_end = 1e-3
    y_seq = y_0
    while solver.successful() and solver.t < time_end:
        time_step = time_end
        solver.integrate(time_step, step=True)
        print(f"TIME : {solver.t:.2e}s\t")
        time_seq.append(solver.t)
        y_seq = np.vstack((y_seq, solver.y))

    # plt.plot(eedf.energy_point, (sol.y / np.sqrt(eedf.energy_point)).transpose())
    """
