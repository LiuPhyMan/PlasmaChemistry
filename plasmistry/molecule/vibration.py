#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  16:34 2019/9/16

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from . import get_vib_energy
from .. import constants as const


def H2_vib_energy_in_eV(*, v):
    return get_vib_energy('H2', quantum_number=v, minimum_is_zero=True)


def H2_vib_energy_in_K(*, v):
    return H2_vib_energy_in_eV(v=v) * const.eV2K


def CO2_vib_energy_in_eV(*, v):
    return get_vib_energy('CO2', quantum_number=v, minimum_is_zero=True)


def CO2_vib_energy_in_K(*, v):
    return CO2_vib_energy_in_eV(v=v) * const.eV2K


def CO_vib_energy_in_eV(*, v):
    return get_vib_energy("CO", quantum_number=v, minimum_is_zero=True)


def CO_vib_energy_in_K(*, v):
    return CO_vib_energy_in_eV(v=v) * const.eV2K


class Molecule_vib_group(object):
    def __init__(self, *, total_density):
        self.total_density = total_density
        self.vib_energy = None
        self.formula = None

    def set_Tvib_K(self, Tvib_K):
        _distri_factor = np.exp(-self.vib_energy / (Tvib_K * const.K2eV))
        self.densities = self.total_density / _distri_factor.sum() * _distri_factor

    def view(self):
        _df = pd.DataFrame(self.vib_energy, index=self.formula,
                           columns=['vib_energy_eV'])
        _df['density'] = self.densities
        _df['ratio'] = [f'{_ * 100:.1f}%' for _ in
                        self.densities / self.total_density]
        return _df

    def plot_vdf(self):
        _ratio = self.densities / self.total_density
        plt.semilogy(self.vib_energy, _ratio, marker='.')
        plt.xticks(self.vib_energy, range(len(self.vib_energy)))

    def __repr__(self):
        _str = self.view().__str__() + f"\nTotal density: {self.total_density}"
        return _str


class H2_vib_group(Molecule_vib_group):
    def __init__(self, *, total_density, Tvib_K, max_v=14):
        super().__init__(total_density=total_density)
        self.formula = ['H2'] + [f'H2(v{v})' for v in range(1, max_v + 1)]
        self.vib_energy = np.array(
            [H2_vib_energy_in_eV(v=v) for v in range(max_v + 1)])
        self.set_Tvib_K(Tvib_K)


class CO_vib_group(Molecule_vib_group):
    def __init__(self, *, total_density, Tvib_K, max_v=10):
        super().__init__(total_density=total_density)
        self.formula = ['CO'] + [f'CO(v{v})' for v in range(1, max_v + 1)]
        self.vib_energy = np.array(
            [CO_vib_energy_in_eV(v=v) for v in range(max_v + 1)])
        self.set_Tvib_K(Tvib_K)


class CO2_vib_group(Molecule_vib_group):
    def __init__(self, *, total_density, Tvib_K, max_v=21):
        super().__init__(total_density=total_density)
        self.formula = ['CO2'] + [f'CO2(v{v})' for v in range(1, max_v + 1)]
        self.vib_energy = np.array(
            [CO2_vib_energy_in_eV(v=(0, 0, v)) for v in range(max_v + 1)])
        self.set_Tvib_K(Tvib_K)


# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    pass
