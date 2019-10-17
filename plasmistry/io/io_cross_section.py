#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16:32 2017/7/11

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import
import re
import numpy as np
import pandas as pd


# %%--------------------------------------------------------------------------------------------- #
def read_cross_section_to_frame(file_path, regexp_key='all'):
    """
    Read cross section file downloaded from http://fr.lxcat.net/home/

    Parameters
    ----------
    file_path : str

    regexp_key : str
        default : all

    Notes
    -----
    lxcat form:
    1st line : possible collision types(in capitals)
        ELASTIC     elastic momentum transfer cross section
        EFFECTIVE   total momentum transfer cross section (sum of elastic momentum
                    transfer and total inelastic cross section)
        EXCITATION
        IONIZATION
        ATTACHMENT
    2nd line : Name of the target particle species.
        E.G.    Ar -> Ar*
                Ar <-> Ar*
    3rd line :
        elastic or effective        the ratio of the electron mass to the target particle
                                    mass.
        excitation or ionization    the threshold energy in eV
        attachment                  missing
    from 4th line (optionally)
        Comments or reference information
    Table of the cross section as a function of energy. The table starts and ends by a
    line of dashes '-----' (at least 5) and has otherwise two numbers per line: the
    energy in eV and the cross section in m2.

    Returns
    -------
    cs_frame : DataFrame_type
        Index   |   cs_key  type  thres_info  energy_range  cross_section  info_list
          ...   |   ...

        Index
            0, 1, 2, 3, ...
        Columns
            cs_key : str
                A->B
            type : str
                elastic effective excitation ionization attachment
            thres_info : str
                Threshold info. '' if type is attachment.
            energy_range : tuple
                (energy_min, energy_max)
            cross_section : np.array
                Cross sections.
                array([[ energy ],
                       [ crostn ]])
            info_list : list

    """
    assert isinstance(file_path, str)
    assert isinstance(regexp_key, str)
    # %%----------------------------------------------------------------------------------------- #
    csM_table = pd.DataFrame(columns=['cs_key',
                                      'type',
                                      'thres_info',
                                      'energy_range',
                                      'cross_section',
                                      'info_list'])
    with open(file_path) as f:
        for line in f:
            if line.strip() in ('ELASTIC', 'EFFECTIVE', 'EXCITATION', 'IONIZATION', 'ATTACHMENT'):
                # %%----------------------------------------------------------------------------- #
                #   read key, type, thres_info
                # %%----------------------------------------------------------------------------- #
                cs_type = line.strip().lower()
                cs_key = f.readline().strip().replace(' ', '')
                thres_info = f.readline().strip() if cs_type != 'attachment' else ''
                # %%----------------------------------------------------------------------------- #
                #   read info_list
                # %%----------------------------------------------------------------------------- #
                info_list = []
                while True:
                    temp = f.readline().strip()
                    if temp.startswith('-----'):
                        break
                    else:
                        info_list.append(temp.strip())
                # %%----------------------------------------------------------------------------- #
                #   read energy and cross section
                # %%----------------------------------------------------------------------------- #
                energy, crostn = [], []
                while True:
                    temp = f.readline().strip()
                    if not temp.startswith('----'):
                        energy_str, crostn_str = temp.split()
                        energy.append(float(energy_str.strip()))
                        crostn.append(float(crostn_str.strip()))
                    else:
                        break
                energy = np.array(energy, dtype=np.float64)
                energy_range = (energy.min(), energy.max())
                crostn = np.array(crostn, dtype=np.float64)
                # %%----------------------------------------------------------------------------- #
                #   check whether the cross section is repeated and if no put it into csM_table
                # %%----------------------------------------------------------------------------- #
                assert cs_key not in csM_table.columns, "'{key}' is repeated.".format(key=cs_key)
                csM_table = csM_table.append(pd.Series(dict(cs_key=cs_key,
                                                            type=cs_type,
                                                            thres_info=thres_info,
                                                            energy_range=energy_range,
                                                            cross_section=np.vstack(
                                                                    (energy, crostn)),
                                                            info_list=info_list)),
                                             ignore_index=True)

    if regexp_key == 'all':
        return csM_table
    else:
        criterion = csM_table['cs_key'].map(
                lambda x: True if re.fullmatch(regexp_key, x) else False)
        assert criterion.any(), "The regexp_key '{}' doesn't work.".format(regexp_key)
        return csM_table[criterion].reset_index(drop=True)
# %%--------------------------------------------------------------------------------------------- #
