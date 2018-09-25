#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15:18 2017/08/24

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import numpy as np
from .ode_solvers import ode_ivp


class ODE_EVENT(object):

    def __init__(self, y0, *, t_min, rtol, atol):
        self.terminal = True
        self.direction = 0  # both goes from positive to negative.
        self._y = y0
        self.calls = 0
        self.residual = []
        self.rtol = rtol
        self.atol = atol
        self.t_min = t_min

    def __call__(self, t, y):
        self.calls += 1
        print("event_time : {t:.2e}".format(t=t))
        if t <= self.t_min:
            is_diff = 1
        else:
            if np.allclose(y, self._y, rtol=self.rtol, atol=self.atol):
                print('close')
                is_diff = 0
            else:
                print('diff')
                is_diff = 1
        self.residual.append(np.abs(self._y - y).max())
        self._y = y
        return is_diff


__all__ = [s for s in dir() if not s.startswith('_')]

from numpy.testing import Tester

test = Tester().test
