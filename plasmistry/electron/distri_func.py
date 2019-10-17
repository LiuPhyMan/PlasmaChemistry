#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21:01 2017/7/13

@author:    Liu Jinbao    
@mail:      liu.jinbao@outlook.com  
@project:   PlasmaChemistry
@IDE:       PyCharm
"""

from math import pi
import numpy as np
from .. import constants as const


# %%--------------------------------------------------------------------------------------------- #
def __electron_energy_grid_check(electron_energy_grid):
    """
    Check electron energy grid.
    
    Parameters
    ----------
    electron_energy_grid : ndarray of float.
        Electron energy space grid.

    """
    assert np.issubsctype(electron_energy_grid, np.float)
    assert np.all(electron_energy_grid > 0)
    assert np.all(np.diff(electron_energy_grid) > 0)
    return True


# %%--------------------------------------------------------------------------------------------- #
def get_maxwell_eedf(electron_energy_grid, *, Te_eV):
    r"""
    Calculate maxwell energy distribution function of electron.
    
    Parameters
    ----------
    electron_energy_grid : ndarray
        Electron energy space grid in unit of J.
    Te_eV : float
        Temperature of electron in unit of eV.

    Returns
    -------
    eedf : ndarray of float
        electron energy distribution function in unit of J^-1
        
    """
    assert __electron_energy_grid_check(electron_energy_grid)
    assert isinstance(Te_eV, float)

    Te_J = Te_eV * const.eV2J
    return 2 * np.sqrt(electron_energy_grid / pi) * (Te_J) ** (-1.5) * \
           np.exp(-electron_energy_grid / Te_J)


# %%--------------------------------------------------------------------------------------------- #
def get_maxwell_eepf(electron_energy_grid, *, Te_eV):
    r"""
    Calculate maxwell energy probability function of electron.
    
    Parameters
    ----------
    electron_energy_grid : ndarray of float
        Electron energy grid in unit of J.
    Te_eV : float
        Temperature of electron in unit of eV.

    Returns
    -------
    eepf : ndarray of float
        electron energy probability function in unit of J^(-3/2)
        
    """
    assert __electron_energy_grid_check(electron_energy_grid)
    assert isinstance(Te_eV, float)

    Te_J = Te_eV * const.eV2J
    return 2 * np.sqrt(1 / pi) * (Te_J) ** (-1.5) * np.exp(-electron_energy_grid / Te_J)

# %%--------------------------------------------------------------------------------------------- #
