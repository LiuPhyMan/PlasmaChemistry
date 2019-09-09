#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20:09 2018/10/25

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plasmistry.io import read_reactionFile
from plasmistry import constants as const
from yaml_demo import H2_vib_energy_in_eV


class H2_vib_group(object):
    def __init__(self, *, total_density, Tvib_K):
        self.total_density = total_density
        self.formula = ['H2'] + [f'H2(v{v})' for v in range(1, 15)]
        self.vib_energy = np.array([H2_vib_energy_in_eV(v=v) for v in range(15)])
        self.set_Tvib_K(Tvib_K)

    def set_Tvib_K(self, Tvib_K):
        _distri_factor = np.exp(-self.vib_energy / (Tvib_K * const.K2eV))
        self.densities = self.total_density / _distri_factor.sum() * _distri_factor

    def view(self):
        _df = pd.DataFrame(self.vib_energy, index=self.formula, columns=['vib_energy'])
        _df['density'] = self.densities
        _df['ratio'] = [f'{_ * 100:.1f}%' for _ in self.densities / self.total_density]
        return _df

    def plot_vdf(self):
        _ratio = self.densities / self.total_density
        plt.semilogy(self.vib_energy, _ratio, marker='.')
        plt.xticks(self.vib_energy, range(len(self.vib_energy)))


# ----------------------------------------------------------------------------------------------- #
def treat_where_or_lambda_cmds(origin_line, treat_cmd):
    r"""
    Examples
    --------
    origin_line
        'H2(v{v}) + H2(v{w1}) => H2(v{v1}) + H2(v{w})    !     ({v}+1)*({w}+1)*kVV0110_H2*(1.5-0.5*exp(-delta*dv))*exp(Delta_1*dv-Delta_2*dv**2)',

    treat_cmd
        '@WHERE: dv = {v} - {w}',
        'delta = 0.21*sqrt(Tgas/300)',
        'Delta_1 = 0.236*(Tgas/300)**0.25',
        'Delta_2 = 0.0572*(300/Tgas)**(1/3)']

    """
    assert isinstance(origin_line, str)

    _line = origin_line
    if isinstance(treat_cmd, list) and treat_cmd[0].startswith("@WHERE"):
        _cmds = [re.sub(r'@WHERE\s*:\s*', '', _) for _ in treat_cmd]
        _cmds = _cmds[::-1]
        for _ in _cmds:
            # get the variable and its expressive string.
            #   var = expressive string
            match_str = re.fullmatch("\s*(?P<var>[^=\s]+)\s*=\s*(?P<expr>[^=\s]+)\s*", _)
            assert match_str
            _var = match_str["var"]
            assert _var in _line, _var
            _expr = match_str["expr"]
            # remove the while space
            _expr = _expr.replace(' ', '')
            # add a bracket
            _expr = f"({_expr})"
            _line = _line.replace(_var, _expr)
        return _line
    elif isinstance(treat_cmd, str) and treat_cmd.startswith("@LAMBDA"):
        match_str = re.fullmatch(r"@LAMBDA\s*:\s*(?P<func>.+)\s*", treat_cmd)
        assert match_str
        lambda_cmd = match_str.groupdict()['func']
        # eval the lambda function.
        lambda_func = eval(lambda_cmd)
        return lambda_func(origin_line)
    else:
        raise Exception(f"The '{origin_line}' is error.")


# ----------------------------------------------------------------------------------------------- #
def treat_multi_cmds(origin_line, _cmds):
    treat_cmds = [_.strip() for _ in _cmds]
    treat_cmds = re.findall(r"(?P<cmds>@(?:WHERE|LAMBDA)\s*:\s*[^@]+)", '\n'.join(treat_cmds))
    _line = origin_line
    for _cmd in treat_cmds:
        _line = treat_where_or_lambda_cmds(_line, _cmd)
    return _line


if __name__ == "__main__":
    pass

    # temp = read_reactionFile('_rctn_list\H2.inp', start_line=123, end_line=131)

    # a = temp['reaction_info']
    # b = temp['pre_exec_list']
    # with open("_rctn_list/CO2_chemistry.gum") as f:
    #     lines = ''.join(f.readlines())
    #
    # a = re.findall(r"Reaction\s*{[^{}]+{[^{}]+}[^{}]+}", lines)
    # for _ in a:
    #     print(re.findall(r"Format[^\n]+\n", _))
