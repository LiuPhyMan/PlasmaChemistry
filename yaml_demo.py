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
from plasmistry.reactions import CrosReactions


class TEST(yaml.YAMLObject):
    yaml_tag = u'!CO2'

    energy_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    def __init__(self, index):
        self.index = int(index)

    def __repr__(self):
        return f"the energy is {self.energy_list[self.index]}"


class Reaction_block(object):

    def __init__(self, *, rctn_dict):
        super().__init__()
        self.rctn_dict = rctn_dict
        self._treat_rctn_dict()

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


# ----------------------------------------------------------------------------------------------- #
# add a constructor
# def CO2_energy_constructor(loader, node):
#     CO2_energy = [0.1, 0.2, 0.3, 0.4]
#     value = loader.construct_scalar(node)
#     return CO2_energy[int(value[-2])]


def eval_constructor(loader, node):
    _str = loader.construct_scalar(node)
    return eval(_str)


CO2_energy = np.arange(20) * 0.2
if __name__ == "__main__":
    # yaml.add_constructor(u"!CO2", CO2_energy_constructor)
    yaml.add_constructor(u"!eval", eval_constructor)
    with open("test_0.yaml") as f:
        temp = yaml.load(f)

    rctn_block_list = temp[-1]['The reaction considered']
    rctn_block = Reaction_block(rctn_dict=rctn_block_list[0])
    reactant = [re.split(r"\s*<?=>\s*", _)[0] for _ in rctn_block._formula_list]
    product = [re.split(r"\s*<?=>\s*", _)[1] for _ in rctn_block._formula_list]

    rctn = CrosReactions(reactant=reactant,
                         product=product,
                         k_str=pd.Series(rctn_block._kstr_list))


# with open("test_0.yaml") as f:
#     a = yaml.load_all(f)
#     a = [_ for _ in a]
# a = [_ for _ in yaml.load_all(lines)]
