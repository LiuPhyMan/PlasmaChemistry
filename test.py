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


if __name__ == "__main__":
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
    import numpy as np
    from matplotlib import pyplot as plt

    _crostn = np.loadtxt(
        r"_cs_list/koelman2016/cs_set/scaling/CO2/cs_CO2v0_vibexc_CO2v2.lut")
    dE = 0.5794
    _energy = _crostn[:, 0]
    _crostn = _crostn[:, 1]
    _crostn_new = _crostn * _energy / (_energy - dE)
    _energy_new = _energy - dE


    _crostn = np.loadtxt(r"_cs_list/koelman2016/cs_set/reverse/CO2"
                         r"/cs_CO2v0_vibexc_CO2v2_reverse.lut")
    plt.plot(_energy_new, _crostn_new, marker='.')
    plt.plot(_crostn[:,0], _crostn[:,1], marker='.')