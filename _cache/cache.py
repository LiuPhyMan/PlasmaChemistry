#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19:32 2019/11/19

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

from math import exp, sqrt
import numpy as np
import numba
from numba.pycc import CC
from numba import float64, void

cc = CC("the_module")
cc.verbose = True

@numba.jit("float64(float64, float64, float64, float64)", nopython=True, nogil=False, parallel=True, cache=True, fastmath=True)
def LT(A, B, C, Tgas):
    return A * exp(B*Tgas**(-1/3)+C*Tgas**(-2/3))

@numba.jit("float64(float64, float64, float64, float64)", nopython=True, nogil=False, parallel=True, cache=True, fastmath=True)
def Lij(dE, a, mu, Tgas):
    return 0.32*dE*11604.5/a*sqrt(mu/Tgas)

@numba.jit("float64(float64)", nopython=True, nogil=False, parallel=True, cache=True, fastmath=True)
def F(_Lij):
    return 0.5*(3-exp(-2/3*_Lij))*exp(-2/3*_Lij)

@cc.export("test", "void(float64[:], float64)")
@numba.jit("void(float64[:], float64)", nopython=True, nogil=False, parallel=True, cache=True, fastmath=True)
def test(value, Tgas):
    value[0] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[1] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[2] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[3] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[4] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[5] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[6] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[7] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[8] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[9] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[10] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[11] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[12] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[13] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[14] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[15] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[16] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[17] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[18] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[19] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[20] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[21] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[22] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[23] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[24] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[25] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[26] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[27] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[28] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[29] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[30] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[31] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[32] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[33] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[34] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[35] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[36] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[37] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[38] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[39] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[40] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[41] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[42] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[43] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[44] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[45] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[46] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[47] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[48] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[49] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[50] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[51] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[52] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[53] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[54] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[55] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[56] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[57] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[58] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[59] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[60] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[61] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[62] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[63] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[64] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[65] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[66] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[67] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[68] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[69] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[70] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[71] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[72] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[73] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[74] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[75] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[76] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[77] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[78] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[79] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[80] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[81] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[82] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[83] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[84] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[85] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[86] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[87] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[88] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[89] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[90] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[91] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[92] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[93] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[94] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[95] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[96] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[97] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[98] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)
    value[99] = (3.8)*((4.58e+19*1.66e-30)*Tgas**(-1.4)*exp(-52536/Tgas))*exp(min(51994*(1),(104400*0.5032))/Tgas)


if __name__ == "__main__":
    cc.compile()

