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

with open("test.yaml") as f:
    lines = ''.join(f.readlines())

a = [_ for _ in yaml.load_all(lines)]
