#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  15:17 2019/7/10

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import yaml
from plasmistry.reactions import CoefReactions
from plasmistry.io import (standard_Arr_constructor,
                           chemkin_Arr_2_rcnts_constructor,
                           chemkin_Arr_3_rcnts_constructor,
                           reversed_reaction_constructor,
                           alpha_constructor,
                           LT_constructor,
                           Coef_Reaction_block)
from plasmistry.molecule import (CO2_vib_energy_in_K, CO2_vib_energy_in_eV)

yaml.add_constructor("!StandardArr", standard_Arr_constructor)
yaml.add_constructor("!ChemKinArr_2_rcnt", chemkin_Arr_2_rcnts_constructor)
yaml.add_constructor("!ChemKinArr_3_rcnt", chemkin_Arr_3_rcnts_constructor)
yaml.add_constructor("!rev", reversed_reaction_constructor)
yaml.add_constructor("!LT", LT_constructor)
yaml.add_constructor("!alpha", alpha_constructor)
# ---------------------------------------------------------------------------- #
vari_dict = {'CO2_vib_energy_in_K': CO2_vib_energy_in_K}
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    with open("./_yaml/test_0.yaml") as f:
        rctn = yaml.load(f)
    rctn_rel = rctn[-1]['The reactions considered']['relaxation reactions']
    rctn_block = Coef_Reaction_block(rctn_dict=rctn_rel['CO2_O_to_CO_O2'],
                                     vari_dict=vari_dict)
