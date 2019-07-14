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
from math import exp
import numpy as np

a = 2


class Entity:
    def __init__(self, idNum, components):
        self.id = idNum
        self.components = components

    def __repr__(self):
        return "%s(id=%r, components=%r)" % (
            self.__class__.__name__, self.id, self.components)


class Component:
    def __init__(self, name):
        self.name = name

    # def __repr__(self):
    #     return "%s(name=%r)" % (
    #         self.__class__.__name__, self.name)


class TEST(yaml.YAMLObject):
    yaml_tag = u'!CO2'

    energy_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    def __init__(self, index):
        self.index = int(index)

    def __repr__(self):
        return f"the energy is {self.energy_list[self.index]}"

    # def __repr__(self):
    #     return f"TEST.name: {self.name}"


class Dice(tuple):
    def __new__(cls, a, b):
        return tuple.__new__(cls, [a, b])

    def __repr__(self):
        return "Dice(%s,%s)" % self


# ----------------------------------------------------------------------------------------------- #
import re


# yaml.add_implicit_resolver(u"!ljb", re.compile(r"CO2_energy"))
# ----------------------------------------------------------------------------------------------- #
# add a constructor
def CO2_energy_constructor(loader, node):
    CO2_energy = [0.1, 0.2, 0.3, 0.4]
    value = loader.construct_scalar(node)
    return CO2_energy[int(value[-2])]


def eval_constructor(loader, node):
    _str = loader.construct_scalar(node)
    return eval(_str)


if __name__ == "__main__":
    # yaml.add_constructor(u"!CO2", CO2_energy_constructor)
    yaml.add_constructor(u"!eval", eval_constructor)
# import re
# pattern = re.compile(r"^CO2\[\d+\]")
# yaml.add_implicit_resolver(u"!CO2", pattern)
CO2 = [i for i in range(100)]
with open("test_0.yaml") as f:
    a = yaml.load_all(f)
    a = [_ for _ in a]
# a = [_ for _ in yaml.load_all(lines)]
r"""
class Hero:
    def __init__(self, name, hp, sp):
        self.name = name
        self.hp = hp
        self.sp = sp

    def __repr__(self):
        return f"name={self.name}, hp={self.hp}, sp={self.sp}"


class Monster(yaml.YAMLObject):
    yaml_tag = u'!Monster'

    def __init__(self, name, hp, ac, attacks):
        self.name = name

    def __repr__(self):
        return f"This is a monster {self.name}"


"""
