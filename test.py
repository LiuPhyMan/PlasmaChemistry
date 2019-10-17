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


def get_latex_format(_str):
    _temp = re.search("\(\(([^()]*)\)\*Tgas\*\*(.*)\*exp\(-(.*)/Tgas\)\)",
                      _str)
    assert _temp
    A_str = f"{eval(_temp.groups()[0]):.2e}"
    n_str = f"{eval(_temp.groups()[1]):.4f}"
    E_str = f"{eval(_temp.groups()[2]):.0f}"
    return r"\[{a} \times 10^#{b}$ #Tgas$ ^ #{c}$ \exp \left( -{d}/Tgas " \
           "\\right)\]".format(a=A_str.split('e')[0],
                               b=A_str.split('e')[1],
                               c=n_str,
                               d=E_str).replace('#', '{').replace('$',
                                                                  '}').replace(
        'Tgas', 'T_{\\rm gas}')


if __name__ == "__main__":
    # this is a test line.
    # index = 164
    # a = pd.read_csv("test_decom_recom.dat")
    # print(get_latex_format(a.loc[index, 'kstr']))
    with open("_cs_list/koelman2016/models/CO2_chemistry.gum") as f:
        line = f.readlines()
    line = [_ for _ in line if _.strip().startswith(('Format', 'Func'))]
    with open('output.dat', 'w') as f:
        f.write(''.join(line))