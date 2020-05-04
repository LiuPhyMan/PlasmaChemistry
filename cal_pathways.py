# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd

from plasmistry.reactions import CoefReactions
from plasmistry.pathways import Pathways

# rctn_str = ["O3=O+O2",
#             "O2=O+O",
#             "O+O2=O3",
#             "O+O3=O2+O2"]
# rctn_str = ["CO2 + O => CO + O2",
#             "CO + O2 => CO2 + O",
#             "CO2 + H => CO + OH",
#             "CO + OH => CO2 + H",
#             "CO2 + C => CO + CO",
#             "O2 + C => CO + O"
#             "H2 + O => H + OH",
#             "H + OH => H2 + O",
#             "H2 + OH => H + H2O",
#             "H + H2O => H2 + OH",
#             "O2 + H => O + OH",
#             "O + OH => O2 + H",
#             "O + H2O => OH + OH",
#             "OH + OH => O + H2O",
#             "CO2 + M => CO + O + M",
#             "CO + O + M => CO2 + M",
#             "O2 + M => O + O + M",
#             "O + O + M => O2 + M",
#             "H2O + M => OH + H + M",
#             "OH + H + M => H2O + M",
#             "H2 + M => H + H + M",
#             "H + H + M => H2 + M",
#             "OH + M => O + H + M",
#             "O + H + M => OH + M"]
_data = pd.read_csv("test0.dat", header=None, index_col=0)
# _data = pd.read_csv("reaction_O3_O2_O.dat", header=None, index_col=0)
rctn_str = _data.index.to_list()
rate_seq = _data[1].values
rcnt_list = [_.split("=>")[0].strip() for _ in rctn_str]
prdt_list = [_.split("=>")[1].strip() for _ in rctn_str]

rcnt_species_list = [_.strip() for _ in " + ".join(rcnt_list).split("+")]
prdt_species_list = [_.strip() for _ in " + ".join(prdt_list).split("+")]
species_list = pd.Series(list(set(rcnt_species_list + prdt_species_list)))
# species_list = pd.Series(["O", "O2", "O3"])
# rate = [80, 20, 99, 1]
rctn = CoefReactions(species=species_list,
                     reactant=rcnt_list,
                     product=prdt_list)
rctn.rate = rate_seq + 1

p = Pathways(reactions=rctn)
# for _spc in ("C", "O", "OH", "H", "O2"):
for _spc in ("C", "O", "H" ):
    print(f"CURRENT SPECIE: {_spc}")
    p.set_crrnt_brspc(_spc)
    p.set_spcs_rate()
    p.sort_by_f1k()
    p.delete_insignificant_pthwys(100)
    p.set_spcs_rate()
    p.update()
