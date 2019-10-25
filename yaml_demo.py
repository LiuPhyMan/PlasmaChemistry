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
                           F_gamma_constructor,
                           alpha_constructor,
                           LT_constructor,
                           Coef_Reaction_block,
                           Cros_Reaction_block)
from plasmistry.molecule import (CO2_vib_energy_in_K, CO2_vib_energy_in_eV,
                                 H2_vib_energy_in_K, H2_vib_energy_in_eV,
                                 CO_vib_energy_in_K, CO_vib_energy_in_eV)

yaml.add_constructor("!StandardArr", standard_Arr_constructor)
yaml.add_constructor("!ChemKinArr_2_rcnt", chemkin_Arr_2_rcnts_constructor)
yaml.add_constructor("!ChemKinArr_3_rcnt", chemkin_Arr_3_rcnts_constructor)
yaml.add_constructor("!rev", reversed_reaction_constructor)
yaml.add_constructor("!LT", LT_constructor)
yaml.add_constructor("!alpha", alpha_constructor)
yaml.add_constructor("!F_gamma", F_gamma_constructor)
# ---------------------------------------------------------------------------- #
vari_dict = {'CO2_vib_energy_in_K': CO2_vib_energy_in_K,
             'H2_vib_energy_in_K': H2_vib_energy_in_K,
             'H2_vib_energy_in_eV': H2_vib_energy_in_eV,
             'CO2_vib_energy_in_eV': CO2_vib_energy_in_eV,
             'CO_vib_energy_in_eV': CO_vib_energy_in_eV,
             'CO_vib_energy_in_K': CO_vib_energy_in_K}
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    with open("./_yaml/test_0.yaml") as f:
        rctn = yaml.load(f)
    rctn_test = rctn[-1]['Test']['test']
    # rctn_ele = rctn[-1]['The reactions considered']['electron reactions']
    global_abbr = rctn[-1]['The reactions considered']['global_abbr']
    rctn_block = Cros_Reaction_block(rctn_dict=rctn_test,
                                     vari_dict=vari_dict,
                                     global_abbr=global_abbr)
    a = rctn_block.generate_crostn_dataframe()
    ###################################
    # rctn_block_ele = Cros_Reaction_block(rctn_dict=rctn_ele[
    #     'H2_ele_vib_forward'],
    #                                      vari_dict=vari_dict,
    #                                      global_abbr=global_abbr)


    # for i in range(22):
    #     for j in range(12):
    #         energy = CO2_vib_energy_in_eV(v=(0,0,i)) + \
    #             CO_vib_energy_in_eV(v=j) - \
    #             CO2_vib_energy_in_eV(v=(0,0,i-1)) - \
    #             CO_vib_energy_in_eV(v=j+1)
    #         print(f"CO2 {i}->{i-1}   CO {j}->{j+1} energy {energy}")



































