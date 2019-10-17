#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16:25 2018/8/17

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import os
import julia
j = julia.Julia()
jl_script = os.path.dirname(__file__) + r"\_diffeq.jl"
with open(jl_script) as f:
    j.eval(''.join(f.readlines()))

sode = j.eval("sode")
