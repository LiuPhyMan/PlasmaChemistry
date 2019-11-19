#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19:32 2019/11/19

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

from math import exp
import numpy as np
import numba
from numba import float32, void
from numba.pycc import CC


cc = CC("my_module_1")

@cc.export("test", "f8[:](f8,)")
def test(Tgas):
    """__REPLACE__"""

if __name__ == "__main__":
    cc.compile()
