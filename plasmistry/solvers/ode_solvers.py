#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15:18 2017/10/24

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import numpy as np
import pandas as pd
from scipy.integrate import ode
from .. import constants as const


def ode_ivp(*, deriv_func, func_args, jac_func, jac_args,
            time_span, y_0, rtol=1e-3, atol=1e-6, show_time=True):
    r"""

    Parameters
    ----------
    deriv_func
    func_args
    time_span
    y_0
    rtol
    atol
    show_time

    Returns
    -------
    output : DataFrame
        Index | time
        Columns | species

    """

    class sol(object):
        pass

    assert isinstance(func_args, tuple)
    assert isinstance(time_span, tuple) and len(time_span) == 2
    time_start, time_end = time_span
    solver = ode(deriv_func, jac_func)
    solver.set_integrator(name='vode', method='bdf', with_jacobian=True, atol=atol, rtol=rtol)
    # solver.set_integrator(name='vode', method='Radau', with_jacobian=True, atol=atol, rtol=rtol)

    # _eedf = func_args[0]
    # _eedf.density_in_J = y_0
    # ne_0 = _eedf.electron_density
    # Te_0 = _eedf.electron_temperature * const.K2eV

    solver.set_f_params(*func_args)
    solver.set_jac_params(*jac_args)
    solver.set_initial_value(y=y_0, t=time_start)

    time_seq = [time_span[0]]
    y_seq = y_0
    # f_seq = y_0 / ne_0 / np.sqrt(func_args[0].energy_point) * const.J2eV ** (-3 / 2)
    # ne_seq = [ne_0]
    # Te_seq = [Te_0]
    while solver.successful() and solver.t < time_end:
        # time_step = time_end
        time_step = 1e-15 if solver.t < 1e-15 else time_end
        # time_step = solver.t * 10 if time_step > solver.t * 10 else time_step
        # print(time_step)
        solver.integrate(time_step, step=True)
        # ne = func_args[0].electron_density
        # Te = func_args[0].electron_temperature * const.K2eV
        if show_time:
            _str = "TIME : {t:.2e}s\tTe : {Te:8.4f}_eV#{Te_K:8.1f}_K]\tne : {ne:8.2e}_m^3"
            _str = "TIME : {t:.2e}s\t"
            print(_str.format(t=solver.t))
        solver.y[solver.y < 1e-30] = 1e-30
        time_seq.append(solver.t)
        y_seq = np.vstack((y_seq, solver.y))
    sol.t = np.array(time_seq)
    sol.y = y_seq
    return sol
