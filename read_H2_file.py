#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10:36 2018/10/13

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

from plasmistry.io import read_reactionFile

output = read_reactionFile('_rctn_list/H2.inp')
a = output['reaction_info']
