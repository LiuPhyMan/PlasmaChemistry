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
from plasmistry.electron import get_rate_const_from_crostn
from plasmistry.electron import get_maxwell_eedf
from scipy.integrate import simps
import math
import numba as nb

# def get_latex_format(_str):
#     _temp = re.search("\(\(([^()]*)\)\*Tgas\*\*(.*)\*exp\(-(.*)/Tgas\)\)",
#                       _str)
#     assert _temp
#     A_str = f"{eval(_temp.groups()[0]):.2e}"
#     n_str = f"{eval(_temp.groups()[1]):.4f}"
#     E_str = f"{eval(_temp.groups()[2]):.0f}"
#     return r"\[{a} \times 10^#{b}$ #Tgas$ ^ #{c}$ \exp \left( -{d}/Tgas " \
#            "\\right)\]".format(a=A_str.split('e')[0],
#                                b=A_str.split('e')[1],
#                                c=n_str,
#                                d=E_str).replace('#', '{').replace('$',
#                                                                   '}').replace(
#         'Tgas', 'T_{\\rm gas}')
# @nb.njit()
# def test(Tgas):
#     a = 0.44 * math.exp(-407.0 * Tgas ** (-1 / 3) + 824.0 * Tgas ** (-2 / 3))
#     return a
#

# this is a test line.
# index = 164
# a = pd.read_csv("test_decom_recom.dat")
# print(get_latex_format(a.loc[index, 'kstr']))
# with open("_cs_list\koelman2016\models\CO2_chemistry_v2.gum") as f:
#     line = f.readlines()
# line = [_.strip() for _ in line if _.strip().startswith(('Name',))]
# with open('output.dat', 'w') as f:
#     f.write('\n'.join(line))
# path = f'_cs_list/H2/H2(X)_to_H2(_b)_to_2H/H2_to_H2(b)_to_2H.csv'
# data = np.loadtxt(path)
# data[:, 1] = data[:, 1] * 1e-20
# np.savetxt(f"H2_to_H2(b)_to_2H.csv", data, fmt='%.2e')
# for i in (100, 200, 300, 400, 500, 600, 800, 1200):
#     _energy = np.linspace(0.001, 40, num=i)
#     _f = get_maxwell_eedf(_energy * const.eV2J, Te_eV=1.0)
#     _crostn = 1 / np.sqrt(_energy * const.eV2J)
#     k = get_rate_const_from_crostn(Te_eV=1.0, crostn=np.vstack((_energy,
#                                                                 _crostn)))
#     print(i, end=' ')
#     print(f"{(1 - k ** 2 * const.m_e / 2) * 100:.1f}%")
# ------------------------------------------------------------------------ #
# import numpy as np
# from matplotlib import pyplot as plt
#
# _crostn = np.loadtxt(
#     r"_cs_list/koelman2016/cs_set/scaling/CO2/cs_CO2v0_vibexc_CO2v2.lut")
# dE = 0.5794
# _energy = _crostn[:, 0]
# _crostn = _crostn[:, 1]
# _crostn_new = _crostn * _energy / (_energy - dE)
# _energy_new = _energy - dE
#
#
# _crostn = np.loadtxt(r"_cs_list/koelman2016/cs_set/reverse/CO2"
#                      r"/cs_CO2v0_vibexc_CO2v2_reverse.lut")
# plt.plot(_energy_new, _crostn_new, marker='.')
# plt.plot(_crostn[:,0], _crostn[:,1], marker='.')

_number = int(1e2)

with open("_cache/cache.txt") as f:
    # multi_expr = [_.strip() for i, _ in enumerate(f.readlines()) if i < _number]
    multi_expr = [ "(((LT(1.05e-3, -324, 1053, Tgas)) * (F((Lij(0.208487, "
                   "(5.172159), (1.9277), Tgas)))) / (F((Lij(0.2085, "
                   "(5.172159), (1.9277), Tgas))))) + ((LT(2.60e-4, -279, 892, "
                   "Tgas)) * (F((Lij(0.125663, (5.172159), (1.9277), Tgas)))) "
                   "/ (F((Lij(0.1257, (5.172159), (1.9277), Tgas))))) + ((LT("
                   "2.61e-5, -293, 914, Tgas)) * (F((Lij(0.042784, "
                   "(5.172159), (1.9277),Tgas)))) / (F((Lij(0.0428, "
                   "(5.172159), (1.9277), Tgas)))))) * 1 * 1e-6"] * _number
    # multi_expr = ["(3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp("
    #               "-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)"] *\
    #              _number
    # multi_expr = ["exp(1)"] * _number

multi_expr_compiled = compile(f"[{','.join(multi_expr)}]", '<string>', 'eval')

with open(r"_cache/cache_template.py") as f:
    _str = f.readlines()
_str = "".join(_str)
_list = [f"value.append({_})" for i, _ in enumerate(multi_expr)]
_multi_expr_str = "\n    ".join(_list)

_str_to_write = _str.replace('"""__REPLACE__"""', f"{_multi_expr_str}")

with open(r"_cache/cache.py", "r+") as f:
    f.seek(0)
    f.truncate()
    f.write(_str_to_write)
