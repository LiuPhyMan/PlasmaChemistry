#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  11:16 2019/7/2

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

import re
from matplotlib import pyplot as plt
from scipy.integrate import ode as sp_ode
import pandas as pd
from plasmistry.reactions import (Reactions, CoefReactions)

reactions = ['2A + 3B =>C+D',
             '2B =>3C+A',
             '=>2D+3A',
             'A =>2C+D',
             ' C =>']
reactant = [re.split(r"=>", _)[0] for _ in reactions]
product = [re.split(r"=>", _)[1] for _ in reactions]
# ----------------------------------------------------------------------------------------------- #
# rctn = Reactions(reactant=pd.Series(reactant),
#                  product=pd.Series(product),
#                  k_str=pd.Series(['Tgas*2', 'Te', 'EN', '2.0d2', '2.0']))
# ----------------------------------------------------------------------------------------------- #
rctn = CoefReactions(reactant=pd.Series(reactant),
                     product=pd.Series(product),
                     k_str=pd.Series(['Tgas*2', 'Te', 'EN', '2.0d2', '2.0']))

rctn.compile_k_str()


# rctn.set_rate(density=rctn.get_initial_density(density_dict=dict(A=1, B=2,C=3)))
# dn = rctn.get_dn()

def deriv_func(t, y, rctn_instance):
    rctn_instance.set_rate_const(Tgas_K=1000, Te_eV=1, EN_Td=1)
    rctn_instance.set_rate(density=y)
    return rctn_instance.get_dn()


solver = sp_ode(deriv_func)
solver.set_integrator(name="vode", method="bdf", with_jacobian=True)
solver.set_f_params(rctn)
init_y0 = rctn.get_initial_density(density_dict=dict(A=200, B=10, C=1))
solver.set_initial_value(init_y0)

# ----------------------------------------------------------------------------------------------- #
time_sep = []
output = pd.DataFrame(init_y0, index=rctn.species, columns=[0.0])
time_end = 100
# evolve
while solver.successful() and solver.t < time_end:
    time_step = time_end
    solver.integrate(time_step, step=True)
    time_sep.append(solver.t)
    output[solver.t] = solver.y

# ----------------------------------------------------------------------------------------------- #
data_to_plot = output.transpose().values
plt.figure()
plt.semilogx(data_to_plot, marker='.')
