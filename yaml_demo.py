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
from plasmistry import constants as const


class Reaction_block(object):

    def __init__(self, *, rctn_dict=None):
        super().__init__()
        self._formula = None
        self._kstr = None
        self._formula_list = None
        self._kstr_list = None
        self._type_list = None
        if rctn_dict is not None:
            self.rctn_dict = rctn_dict
            self._treat_rctn_dict()

    @property
    def _reactant_str_list(self):
        return [re.split(r"\s*<?=>\s*", _)[0] for _ in self._formula_list]

    @property
    def _product_str_list(self):
        return [re.split(r"\s*<?=>\s*", _)[1] for _ in self._formula_list]

    @property
    def size(self):
        assert len(self._formula_list) == len(self._kstr_list)
        return len(self._formula_list)

    def __add__(self, other):
        result = Reaction_block()
        result._formula_list = self._formula_list + other._formula_list
        result._kstr_list = self._kstr_list + other._kstr_list
        result._type_list = self._type_list + other._type_list
        return result

    def _treat_rctn_dict(self):
        self._formula = self.rctn_dict['formula']
        self._kstr = self.rctn_dict['kstr']

        assert ("zip" not in self.rctn_dict) or ("lambda" not in self.rctn_dict)
        # --------------------------------------------------------------------------------------- #
        if "where" in self.rctn_dict:
            assert isinstance(self.rctn_dict['where'], dict)
            if "abbr" in self.rctn_dict["where"]:
                reversed_abbr_dict = self.rctn_dict["where"]["abbr"][::-1]
                for _key_value in reversed_abbr_dict:
                    _key = list(_key_value.items())[0][0]
                    _value = str(list(_key_value.items())[0][1])
                    self._formula = self._formula.replace(_key, _value)
                    self._kstr = self._kstr.replace(_key, _value)
            if "vari" in self.rctn_dict["where"]:
                reversed_vari_dict = self.rctn_dict["where"]["vari"][::-1]
                for _key_value in reversed_vari_dict:
                    _key = list(_key_value.items())[0][0]
                    _value = f"({str(list(_key_value.items())[0][1])})"
                    self._formula = self._formula.replace(_key, _value)
                    self._kstr = self._kstr.replace(_key, _value)

        # --------------------------------------------------------------------------------------- #
        if "zip" in self.rctn_dict:
            _formula_list = []
            _kstr_list = []
            _zip_dict = self.rctn_dict['zip']
            _zip_number = len(_zip_dict[list(_zip_dict.keys())[0]])
            for _i in range(_zip_number):
                _formula = self._formula
                _kstr = self._kstr
                for _key in _zip_dict:
                    _formula = _formula.replace(_key, str(_zip_dict[_key][_i]))
                    _kstr = _kstr.replace(_key, str(_zip_dict[_key][_i]))
                _formula_list.append(_formula)
                _kstr_list.append(_kstr)
            self._formula_list = _formula_list
            self._kstr_list = _kstr_list
        if "lambda" in self.rctn_dict:
            lambda_func = self.rctn_dict['lambda']
            self._formula_list = lambda_func(self._formula)
            self._kstr_list = lambda_func(self._kstr)

        # --------------------------------------------------------------------------------------- #
        self._type_list = [self.rctn_dict["type"] for _ in range(self.size)]


class Cros_Reaction_block(Reaction_block):

    def __init__(self, *, rctn_dict=None):
        super().__init__(rctn_dict=rctn_dict)
        if rctn_dict is not None:
            self._threshold_list = self.rctn_dict["threshold"]

    def __add__(self, other):
        result = Cros_Reaction_block()
        result._formula_list = self._formula_list + other._formula_list
        result._kstr_list = self._kstr_list + other._kstr_list
        result._type_list = self._type_list + other._type_list
        result._threshold_list = self._threshold_list + other._threshold_list
        return result

    def generate_crostn_dataframe(self, *, factor=1):
        # _df = pd.DataFrame(index=range(self.size),
        #                    columns=["cs_key", "type", "threshold_eV", "cross_section"])
        _df = dict()
        _df["formula"] = self._formula_list
        _df["type"] = self._type_list
        _df["threshold_eV"] = self._threshold_list
        _df["cross_section"] = [np.loadtxt(_path).transpose() * factor for _path in
                                self._kstr_list]
        # # _df["cross_section"] = self._kstr_list
        # for _index, _path in zip(range(self.size), self._kstr_list):
        #     print(_index)
        #     print(_path)
        #     _df.loc[_index, "cross_section"] = np.loadtxt(_path)
        _df = pd.DataFrame(data=_df, index=range(self.size))
        return _df


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


def H2_vib_energy_in_eV(*, v):
    return get_vib_energy('H2', quantum_number=v, minimum_is_zero=True)


def CO2_vib_energy_in_eV(*, v):
    return get_vib_energy('CO2', quantum_number=v, minimum_is_zero=True)


if __name__ == "__main__":
    # yaml.add_constructor(u"!CO2", CO2_energy_constructor)
    yaml.add_constructor(u"!eval", eval_constructor)
    yaml.add_constructor(u"!LT", LT_constructor)
    yaml.add_constructor(u"!Arr", Arr_constructor)
    with open("test_0.yaml") as f:
        temp = yaml.load(f)

    ele_rctn_block_list = temp[-1]["The reaction considered"]["electron reaction"]
    # rctn_block_list = ele_rctn_block_list["H2_ele_dis_rctn_via_b"]
    rctn_block_0 = Cros_Reaction_block(rctn_dict=ele_rctn_block_list["H2_ele_vib_rctn_forward"])
    rctn_block_1 = Cros_Reaction_block(rctn_dict=ele_rctn_block_list["H2_ele_vib_rctn_backward"])
    rctn_block = rctn_block_0 + rctn_block_1
    crostn_dataframe = rctn_block.generate_crostn_dataframe(factor=1e-20)
    crostn_dataframe["reaction"] = crostn_dataframe["cs_key"]
    # ------------------------------------------------------------------------------------------- #
    r"""
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
