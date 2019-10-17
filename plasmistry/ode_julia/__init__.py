#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16:22 2018/8/17

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import
from .diffeq import sode

__all__ = [s for s in dir() if not s.startswith('_')]

from numpy.testing import Tester

test = Tester().test
