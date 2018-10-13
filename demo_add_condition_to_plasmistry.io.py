#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9:54 2018/10/13

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

from plasmistry.io import test_func

line = 'E + @A@ => E + @B@'
replc_list = ['@A@ = H2(1) H2(2) H2(3)',
              '@B@ = H2(1) H2(2) H2(3)',
              '@CONDITION : @A@[3]>=@B@[3]']
output = test_func(line, replc_list)
