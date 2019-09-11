import math
import numpy as np

Cd = 1.3
tho = 1.13
D = 1.5e-3

r = np.linspace(5e-3, 12e-3, num=100)
r1 = 12e-3
dr = r[1] - r[0]
S = math.pi * r1 ** 2
F = 2e-3 / 60
vz = F / S


def Fd(f):
    _Fd = math.pi * Cd * tho * D * r * f * np.sqrt((2 * math.pi * r * f) ** 2 + vz ** 2)
    return _Fd


def Md(f):
    _Fd = Fd(f)
    return (_Fd * r).sum() * dr


Md_array = r"""
1.26217E-8
9.22213E-8
1.51772E-7
1.95792E-7
2.17903E-7
1.13681E-7
1.1468E-7
1.13114E-7
1.08358E-7
9.86612E-8
8.1284E-8"""
Md_array = [float(_) for _ in Md_array.split()]


def find(_Md):
    _f_array = np.linspace(1, 40, num=300)
    _Md_array = np.array([Md(f) for f in _f_array])
    return _f_array[np.abs(_Md_array - _Md).argmin()]


for _ in Md_array:
    print(find(_))
