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
from yaml_demo import (H2_vib_energy_in_eV,
                       CO_vib_energy_in_eV,
                       CO2_vib_energy_in_eV)
from plasmistry.molecule import H2_vib_group, CO2



if __name__ == "__main__":
    H2_group = H2_vib_group(total_density=1e25, Tvib_K=2000)
    CO2_group = CO2_vib_group(total_density=1e25, Tvib_K=2000)
    CO_group = CO_vib_group(total_density=1e25, Tvib_K=2000)
