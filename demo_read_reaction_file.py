#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  15:22 2019/7/6

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

import pandas as pd
from plasmistry.reactions import Reactions


rctn = Reactions(species=pd.Series(['O(1)', 'O(2)', 'O(3)','O2', 'O(all)',
                                    'O3']),
                 reactant=pd.Series(['O(1)',
                                     'O(2)',
                                     'O(1) + O(2)',
                                     '3O2 + O(all)']),
                 product=pd.Series(["O(2)",
                                    "O(3)",
                                    "O2",
                                    "2O3 + O(all)"]),
                 k_str=None)

