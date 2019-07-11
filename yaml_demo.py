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


# stream =
# with open("test.yaml") as f:
#     lines = ''.join(f.readlines())

# a = [_ for _ in yaml.load_all(lines)]

class Hero:
    def __init__(self, name, hp, sp):
        self.name = name
        self.hp = hp
        self.sp = sp

    def __repr__(self):
        return f"name={self.name}, hp={self.hp}, sp={self.sp}"

a = yaml.load(r"""
!!python/object:__main__.Hero
name: WS
hp: 1200
sp: 0
""")