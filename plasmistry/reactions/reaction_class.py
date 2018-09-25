#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14:03 2017/7/10

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import

import copy
import math
import re
import numpy as np
import pandas as pd

from numpy import ndarray as ndarray_type
from scipy.integrate import simps, trapz
from scipy import sparse as spr
from scipy.interpolate import interp1d
from scipy.integrate import ode as sp_ode
from pandas.core.series import Series as Series_type
from pandas.core.frame import DataFrame as DataFrame_type

from .. import constants as const
from ..electron import EEDF


# ----------------------------------------------------------------------------------------------- #
class ReactionClassError(Exception):
    pass


# ----------------------------------------------------------------------------------------------- #
class Reactions(object):
    r"""
    Reactions class.

    Class Attributes
    ----------------
    specie_regexp : str
        Regular expression of specie.
    cmpnds_regexp
        Regular expression of reactant or product.

    Attributes
    ----------
    reaction_type : str
        {'k_coefficients related', 'cross_sections related'}
    species : Series of str
        All species in products and reactants.
    n_species : int
        Number of species.
    n_reactions : int
        Number of reactions.
    reactant : Series of str
        Reactants of all reactions.
    product : Series of str
        Products of all reactions.
    k_str : Series of str
        Strings of rate constants formulas.
    dH_e : Series of float
        Reaction enthalpies in electron.
    dH_g : Series of float
        Reaction enthalpies in molecule and atom.
    rate_const : Series of float
        Rate constants of reactions.
    rate : Series of float
        Reaction rates of reactions.
    __rcntsij : csc_matrix
        Compressed sparse columns matrix.
    __rcnt_index : numpy.array of int
        Index of reactants.
    __rcnt_expnt : numpy.array of int
        Exponent of reactants.
    __prdtsij : csc_matrix
        Compressed sparse columns matrix.
    __sij : csr_matrix
        Compressed sparse rows matrix.

    """
    specie_regexp = r"{head}(?:{middle})*(?:{tail})?".format(head=r"[a-zA-Z]",
                                                             middle=r"(?<=\^)[+]|[^\+\s@]",
                                                             tail=r"\^[\+\-]")
    cmpnds_regexp = r"{molecule}(?:{sep}{molecule})*|".format(molecule=r"\d*" + specie_regexp,
                                                              sep=r"[ ][+][ ]")
    __slots__ = ['reaction_type',
                 'species',
                 'reactant',
                 'product',
                 'n_species',
                 'n_reactions',
                 'k_str',
                 'dH_e',
                 'dH_g',
                 'rate_const',
                 'rate',
                 '__rcntsij',
                 '__prdtsij',
                 '__sij',
                 '__rcnt_index',
                 '__rcnt_expnt']
    
    # ------------------------------------------------------------------------------------------- #
    #   __init__
    # ------------------------------------------------------------------------------------------- #
    def __init__(self, *, reactant, product, k_str, dH_g=None, dH_e=None):
        r"""
        Initiation of instance.

        Parameters
        ----------
        reactant : Series of str.
            Reactants of all reactions.
        product : Series of str.
            Products of all reactions.
        k_str : Series of str.
            Rate constants formulas.
        dH_g : Series of float, optional

        dH_e : Series of float, optional

        """
        assert isinstance(k_str, Series_type) or (k_str is None)
        assert isinstance(dH_g, Series_type) or (dH_g is None)
        assert isinstance(dH_e, Series_type) or (dH_e is None)
        
        # --------------------------------------------------------------------------------------- #
        #   reactant product species
        # --------------------------------------------------------------------------------------- #
        self.reactant = self.format_cmpnds(pd.Series(reactant))
        self.product = self.format_cmpnds(pd.Series(product))
        _temp = re.findall(self.specie_regexp, " ".join(self.reactant + " " + self.product))
        self.species = pd.Series(np.unique([_ for _ in _temp if _ != ""]))
        del _temp
        assert self._regexp_check(self.species, self.specie_regexp)
        assert self._regexp_check(self.reactant, self.cmpnds_regexp)
        assert self._regexp_check(self.product, self.cmpnds_regexp)
        
        # --------------------------------------------------------------------------------------- #
        #   number of species and that of reactions.
        # --------------------------------------------------------------------------------------- #
        self.n_species = self.species.size
        self.n_reactions = self.reactant.size
        assert self.reactant.size == self.product.size
        
        # --------------------------------------------------------------------------------------- #
        #   k_str
        # --------------------------------------------------------------------------------------- #
        self.k_str = k_str.str.strip().str.replace(r"\s+", '') if k_str is not None else None
        
        # --------------------------------------------------------------------------------------- #
        #   rate constant, rate
        # --------------------------------------------------------------------------------------- #
        self.rate_const = None
        self.rate = None
        
        # --------------------------------------------------------------------------------------- #
        #   __rcntsij, __prdtsij, __sij, __rcnt_index and __rcnt_expnt
        # --------------------------------------------------------------------------------------- #
        self.__rcntsij = spr.csc_matrix(self._get_sparse_paras(self.species, self.reactant),
                                        shape=(self.n_species, self.n_reactions))
        self.__prdtsij = spr.csc_matrix(self._get_sparse_paras(self.species, self.product),
                                        shape=(self.n_species, self.n_reactions))
        self.__sij = (-self.__rcntsij + self.__prdtsij).tocsr(copy=True)
        self.__rcnt_index, self.__rcnt_expnt = self._get_rcnt_index_expnt(self.reactant,
                                                                          self.__rcntsij)
        
        # --------------------------------------------------------------------------------------- #
        #   dH_g, dH_e
        # --------------------------------------------------------------------------------------- #
        init_dH = pd.Series(np.zeros(self.n_reactions), dtype=np.float64)
        self.dH_g = init_dH if dH_g is None else dH_g
        self.dH_e = init_dH if dH_e is None else dH_e
        del init_dH
    
    # ------------------------------------------------------------------------------------------- #
    #   __setattr__
    # ------------------------------------------------------------------------------------------- #
    def __setattr__(self, key, value):
        if key == 'reaction_type':
            assert value in ('k_coefficients related', 'cross_sections related', 'mixed reactions')
        if key in ('rate_const', 'rate'):
            assert value is None or \
                   (isinstance(value, ndarray_type) and value.size == self.n_reactions)
        if key in ('dH_e', 'dH_g'):
            assert value is None or \
                   (isinstance(value, Series_type) and value.size == self.n_reactions)
        object.__setattr__(self, key, value)
    
    # ------------------------------------------------------------------------------------------- #
    #   Static_methods
    #       - fortran2python
    #       _ format_cmpnds
    #       - _regexp_check
    #       - _get_sparse_paras
    #       - _get_rcnt_index_expnt
    # ------------------------------------------------------------------------------------------- #
    @staticmethod
    def fortran2python(_expr):
        r"""

        Parameters
        ----------
        _expr : str
            Fortran number expression.

        Returns
        -------
        _expr_python : str
            Python number expression.

        Examples
        --------
        '1d2'      -> '1e2'
        '2.0d+21'  -> '2.0e+21'

        """
        assert isinstance(_expr, str)
        return re.sub(r"(?<=[\d\.])d(?=[\+\-\d])", "e", _expr)
    
    @staticmethod
    def format_cmpnds(_series_str):
        r"""

        Parameters
        ----------
        _series_str : Series of str

        Returns
        -------
        formatted_cmpnds : Series of str

        Examples
        --------
        ' 2A+3B ' -> '2A + 3B'
        'A^++C'   -> 'A^+ + C'

        """
        assert isinstance(_series_str, Series_type)
        return _series_str.str.replace(r"(?<![\^\s])[+](?!\s)",
                                       " + ").str.replace(r"\s+[+]\s+", " + ").str.strip()
    
    @staticmethod
    def _regexp_check(_series, _regexp):
        r"""
        Check species, reactants or products formats.

        Parameters
        ----------
        _series : Series of str
            Specie, reactant or product.
        _regexp : str
            Regular expression.

        Returns
        -------
        TrueOrFalse : bool

        """
        assert isinstance(_series, Series_type)
        regexp_check = re.compile(_regexp)
        lamb_regexp_check = lambda _str: True if regexp_check.fullmatch(_str) else False
        return _series.apply(lamb_regexp_check).all()
    
    @staticmethod
    def _get_sparse_paras(species, cmpnds):
        r"""
        Returns
        -------
        data : ndarray

        indices : ndarray

        indptr : ndarray

        Notes
        -----
        >>> cmpnds = pd.Series(['2A + B','2C','B + B',''])
        0    2A + B
        1        2C
        2     B + B
        3
        dtype: object
        >>> cmpnds_str_total
        ' 2A + B + 2C + B + B + '
        >>> data_str
        ['2', '', '2', '', '']
        >>> spcs_all
        ['A', 'B', 'C', 'B', 'B']
        >>> data
        [2 1 2 1 1]
        >>> indices
        [0 1 2 1 1]
        >>> indptr
        [0 2 3 5 5]

        """
        assert isinstance(cmpnds, Series_type)
        assert isinstance(species, Series_type)
        cmpnds_str_total = " " + " + ".join(cmpnds) + " "
        data_str = re.findall(r"(?<=\s)(\d*){}(?=\s)".format(Reactions.specie_regexp),
                              cmpnds_str_total)
        spcs_all = re.findall(r"(?<=\s)\d*({})(?=\s)".format(Reactions.specie_regexp),
                              cmpnds_str_total)
        assert set(species) >= set(spcs_all)
        lamb_n_cmpnds = lambda x: x.count(" + ") + 1 if x.strip() else 0
        # --------------------------------------------------------------------------------------- #
        #   data indices indptr
        # --------------------------------------------------------------------------------------- #
        data = np.array([int(_) if _ else 1 for _ in data_str], dtype=np.int64)
        indices = pd.Series(range(len(species)), index=species, dtype=np.int64)[spcs_all].values
        indptr = np.insert(cmpnds.apply(lamb_n_cmpnds).cumsum().values, 0, [0])
        return data, indices, indptr
    
    @staticmethod
    def _get_rcnt_index_expnt(rcnt, rcntsij):
        """

        Returns
        -------
        rcnt_index : ndarray of int
        rcnt_expnt : ndarray of int

        Notes
        -----
        >>> rcntsij
        array([[2, 0, 0, 1, 0],
               [3, 2, 0, 0, 0],
               [0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0]], dtype=int64)
        >>> sji_lil
        array([[2, 3, 0, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [1, 0, 0, 0],
               [0, 0, 1, 0]], dtype=int64)
        >>> sji_lil.rows
        array([[0, 1],
               [1],
               [],
               [0],
               [2]], dtype=object)
        >>> sji_lil.data
        array([[2, 3],
               [2],
               [],
               [1],
               [1]], dtype=object)

        >>> rcnt_index
        array([[ 0,  1],
               [ 1, -1],
               [-1, -1],
               [ 0, -1],
               [ 2, -1]], dtype=int64)
        >>> rcnt_expnt
        array([[2, 3],
               [2, 0],
               [0, 0],
               [1, 0],
               [1, 0]], dtype=int64)

        """
        n_reactions = rcnt.size
        sji_lil = rcntsij.transpose(copy=True).tolil()
        n_rcnt = pd.Series(sji_lil.rows).apply(len)
        max_n_rcnt = n_rcnt.max()
        # assert max_n_rcnt >= 2  # if necessary
        index_adjsmnt = pd.Series([[-1]] * n_reactions) * (max_n_rcnt - n_rcnt)
        expnt_adjsmnt = pd.Series([[+0]] * n_reactions) * (max_n_rcnt - n_rcnt)
        rcnt_index = np.array((sji_lil.rows + index_adjsmnt).tolist(), dtype=np.int64)
        rcnt_expnt = np.array((sji_lil.data + expnt_adjsmnt).tolist(), dtype=np.int64)
        return rcnt_index, rcnt_expnt
    
    # ------------------------------------------------------------------------------------------- #
    #   rate, dn, dH_e, dH_g
    # ------------------------------------------------------------------------------------------- #
    def set_rate(self, *, density):
        r"""
        Calculate the reaction rate.
        rate = k * [A]**a * [B]**b ...
            set self.rate

        Parameters
        ----------
        density : ndarray of float.
            Densities of species.

        """
        assert isinstance(density, ndarray_type)
        # assert np.all(self.rate_const >= 0.0)
        self.rate = self.rate_const * np.prod(density[self.__rcnt_index] ** self.__rcnt_expnt,
                                              axis=1)
    
    def get_dn(self):
        r"""
        Calculate the derivations of densities.

        Returns
        -------
        dn : pandas.Series
            Index   | self.species
            Values  | derivations of densities

        """
        # assert np.all(self.rate>=0.0)
        return self.__sij.dot(self.rate)
    
    def get_dH_e(self):
        r"""
        Calculate the total enthalpy in electron.

        Returns
        -------
        dH_e : float

        """
        return self.dH_e.dot(self.rate)
    
    def get_dH_g(self):
        r"""
        Calculate the total enthalpy in molecule and atom.

        Returns
        -------
        dH_g : float

        """
        return self.dH_g.dot(self.rate)
    
    # ------------------------------------------------------------------------------------------- #
    def get_initial_density(self, *, density_dict, min_density=0.0):
        r"""
        Set the initial densities of species basing on the density_dict.

        Parameters
        ----------
        density_dict : dict
            {species: density}
        min_density : float, optional

        Returns
        -------
        initial_density : ndarray of float

        """
        assert isinstance(density_dict, dict)
        assert isinstance(min_density, float)
        
        density_0 = pd.Series(min_density, index=self.species, dtype=np.float64)
        for key in density_dict:
            assert key in self.species.values, "{} not in species.".format(key)
            assert isinstance(density_dict[key], float)
            density_0[key] = density_dict[key]
        return density_0.values
    
    def __str__(self):
        r"""
        Print the info of reactions.

        """
        table = pd.DataFrame({
            'reactions': self.reactant + ' => ' + self.product,
            'dH_g[eV]': self.dH_g,
            'dH_e[eV]': self.dH_e,
            'rate_const_str': self.k_str,
            'rate_const': self.rate_const,
            'rate': self.rate
            }, columns=['reactions',
                        'dH_g[eV]',
                        'dH_e[eV]',
                        'rate_const_str',
                        'rate_const',
                        'rate'])
        output = """
        \n====SPECIES====
        \n{species}
        \n====REACTIONS====
        \n{table}
        \n====
        \nCLASS : {class_name}.
        \nType : {type}.
        \n__{n_spcs}__ species. __{n_rctns}__ reactions.
        """.format(species=self.species,
                   table=table,
                   class_name=self.__class__,
                   type=self.reaction_type,
                   n_spcs=self.n_species,
                   n_rctns=self.n_reactions)
        return output
    
    @property
    def view(self):
        output = pd.DataFrame({
            'formula': self.reactant + ' => ' + self.product,
            'k_str': self.k_str,
            'dH_e': self.dH_e,
            'dH_g': self.dH_g,
            'rate_const': self.rate_const,
            'rate': self.rate
            },
                columns=['formula', 'dH_e', 'dH_g', 'k_str', 'rate_const', 'rate'])
        return output


# ----------------------------------------------------------------------------------------------- #
class CoefReactions(Reactions):
    r"""

    Parameters
    ----------
    pre_exec_list : list of str
        Previous execute statements.
    pre_exec_list_compiled : list of code object
        Compiled previous execute statements.
    mid_variables : dict
        Intermediate variables that is used to calculate the rate constants.
    k_str_compiled : func
        Function that calculate the rate constants
        Calling format : self.k_str_compiled(Tgas_K=..., Te_eV=..., EN_Td=...)

    """
    __slots__ = Reactions.__slots__ + ['rate_const_instance',
                                       'pre_exec_list',
                                       'pre_exec_list_compiled',
                                       'k_str_compiled',
                                       'mid_variables']
    
    # ------------------------------------------------------------------------------------------- #
    def __init__(self, *, reactant, product, k_str=None, dH_g=None, dH_e=None):
        Reactions.__init__(self, reactant=reactant, product=product,
                           k_str=k_str, dH_g=dH_g, dH_e=dH_e)
        self.reaction_type = 'k_coefficients related'
        self.rate_const_instance = None
        self.pre_exec_list = None
        self.pre_exec_list_compiled = None
        self.mid_variables = None
        self.k_str_compiled = None
    
    # ------------------------------------------------------------------------------------------- #
    def __str__(self):
        output = """
        \n====PRE_EXEC_LIST====
        \n{pre_exec_list}
        \n====K_STR_COMPILED====
        \n{k_str_compiled}
        \n====MID_VARIABLES====
        \n{mid_variables}""".format(
                pre_exec_list=r'\n'.join(self.pre_exec_list) if self.pre_exec_list else [],
                mid_variables=pd.Series(self.mid_variables),
                k_str_compiled=self.k_str_compiled)
        return Reactions.__str__(self) + output
    
    # ------------------------------------------------------------------------------------------- #
    def set_pre_exec_list(self, exec_list):
        r"""
        Set list of pre-execute statements.
            set self.pre_exec_list

        Parameters
        ----------
        exec_list : list

        """
        assert isinstance(exec_list, list)
        self.pre_exec_list = [self.fortran2python(_.strip()) for _ in exec_list]
        self.pre_exec_list_compiled = [compile(_, '<string>', 'exec') for _ in self.pre_exec_list]
    
    # ------------------------------------------------------------------------------------------- #
    def compile_k_str(self):
        r"""
        Compile the k_str. Raise error if '' in k_str.
            set self.k_str_compiled

        """
        assert "" not in self.k_str.values
        k_str_formated = self.k_str.apply(self.fortran2python)
        self.k_str_compiled = compile('({})'.format(', '.join(k_str_formated)), '<string>', 'eval')
    
    # ------------------------------------------------------------------------------------------- #
    def set_rate_const_instance(self, rate_const_instance):
        self.rate_const_instance = rate_const_instance
    
    # ------------------------------------------------------------------------------------------- #
    def set_rate_const(self, *, density, Tgas_K, Te_eV, EN_Td):
        r"""
        Calculate rate constants basing on k_str expressions.
            set self.mid_variables
                self.rate_const

        Notes
        -----
        1. update mid_variables.
            run pre_exec_list
        2. calculate rate constants.
            evaluate k_str_compiled in mid_variables

        Parameters
        ----------
        Tgas_K : float
            Temperature of gas in Kelvin.
        Te_eV : float
            Temperature of electron in eV.
        EN_Td : float
            Electric field.

        """
        # assert isinstance(Tgas_K, float)
        # assert isinstance(Te_eV, float)
        # assert isinstance(EN_Td, float)
        #
        # self.mid_variables = {'EN': EN_Td, 'Tgas': Tgas_K, 'Te': Te_eV}
        # if self.pre_exec_list is not None:
        #     for line_compiled in self.pre_exec_list_compiled:
        #         exec(line_compiled, {'exp': math.exp}, self.mid_variables)
        # self.rate_const = np.array(eval(self.k_str_compiled,
        #                                 {'exp': math.exp, 'log': math.log},
        #                                 self.mid_variables), dtype=np.float64)
        # --------------------------------------------------------------------------------------- #
        self.rate_const_instance.set_density(density)
        # print(self.rate_const_instance.rate_const(Tgas_K))
        self.rate_const = self.rate_const_instance.rate_const(Tgas_K)


# ----------------------------------------------------------------------------------------------- #
class CrosReactions(Reactions):
    r"""

    Parameters
    ----------
    cs_type : None or Series of str.
        None for 'k_coefficients related' type.
    cs_thres : None or Series of str.
        None for 'k_coefficients related' type.
    cs_crostn : None or ndarray of float.
        None for 'k_coefficients related' type.
    cs_electron_energy_grid : None or ndarray of float.
        None for 'k_coefficients related' type.

    """
    __slots__ = Reactions.__slots__ + ['rate_const_matrix',
                                       'crostn_dataframe']
    
    # ------------------------------------------------------------------------------------------- #
    def __init__(self, *, reactant, product, k_str, dH_g=None, dH_e=None):
        Reactions.__init__(self, reactant=reactant, product=product,
                           k_str=k_str, dH_g=dH_g, dH_e=dH_e)
        self.reaction_type = 'cross_sections related'
    
    # def __init__(self, *, reaction_dataframe):
    
    # ------------------------------------------------------------------------------------------- #
    def set_rate_const_matrix(self, *, crostn_dataframe, electron_energy_grid):
        r"""
        Import cross sections from cs_DataFrame basing on electron_energy_grid.

        Parameters
        ----------
        cs_frame : DataFrame
            Index   |   type    thres_info  energy  energy_range    crostn  info_dict
              ...   |   ...
            Columns
                type : str
                    ELASTIC EFFECTIVE EXCITATION IONIZATION ATTACHMENT
                thres_info : str
                    Threshold info. '' if type is ATTACHMENT
                energy : ndarray
                    Energy space.
                energy_range : tuple
                    (energy_min, energy_max)
                crostn : np.array
                    Cross sections.
                info_dict : dict


        """
        assert isinstance(crostn_dataframe, DataFrame_type)
        assert isinstance(electron_energy_grid, ndarray_type)
        _rate_const_matrix = np.empty((crostn_dataframe.shape[0], electron_energy_grid.size))
        for i_rctn, cs_key in enumerate(self.k_str):
            if cs_key not in crostn_dataframe['cs_key'].tolist():
                raise ReactionClassError('"{}" is not in the cs_frame.'.format(cs_key))
            else:
                _cs_series = crostn_dataframe[crostn_dataframe['cs_key'] == cs_key]
                _cs_series = _cs_series.reset_index(drop=True).loc[0]
                _args = dict(reaction_type=_cs_series['type'],
                             energy_grid_J=electron_energy_grid,
                             threshold_eV=_cs_series['threshold_eV'],
                             crostn_eV_m2=_cs_series['cross_section'])
                _rate_const_matrix[i_rctn] = EEDF.get_rate_const_matrix_molecule(**_args)
        self.rate_const_matrix = _rate_const_matrix
    
    # ------------------------------------------------------------------------------------------- #
    def set_rate_const(self, *, eedf_normalized):
        r"""
        Calculate rate constants basing on eedf and cross sections.
            set self.rate_const

        Parameters
        ----------
        eedf_normalized : ndarray of float

        eedf : ndarray of float

        """
        self.rate_const = self.rate_const_matrix.dot(eedf_normalized)


# ----------------------------------------------------------------------------------------------- #
class MixReactions(Reactions):
    __slots__ = Reactions.__slots__ + ['cros_reactions', 'coef_reactions']
    
    # ------------------------------------------------------------------------------------------- #
    def __init__(self, *, cros_instance, coef_instance):
        assert isinstance(cros_instance, CrosReactions)
        assert isinstance(coef_instance, CoefReactions)
        
        def mix_series(a, b):
            assert isinstance(a, Series_type)
            assert isinstance(b, Series_type)
            return a.append(b, ignore_index=True)
        
        mix_reactant = mix_series(cros_instance.reactant, coef_instance.reactant)
        mix_product = mix_series(cros_instance.product, coef_instance.product)
        mix_dH_g = mix_series(cros_instance.dH_g, coef_instance.dH_g)
        mix_dH_e = mix_series(cros_instance.dH_e, coef_instance.dH_e)
        mix_k_str = mix_series(cros_instance.k_str, coef_instance.k_str)
        
        Reactions.__init__(self, reactant=mix_reactant,
                           product=mix_product,
                           k_str=mix_k_str,
                           dH_g=mix_dH_g,
                           dH_e=mix_dH_e)
        
        self.cros_reactions = copy.deepcopy(cros_instance)
        self.coef_reactions = copy.deepcopy(coef_instance)
        self.reaction_type = 'mixed reactions'
    
    def set_rate_const(self, *, eedf_normalized, Tgas_K, Te_eV, EN_Td):
        self.cros_reactions.set_rate_const(eedf_normalized=eedf_normalized)
        self.coef_reactions.set_rate_const(Tgas_K=Tgas_K, Te_eV=Te_eV, EN_Td=EN_Td)
        self.rate_const = np.hstack((self.cros_reactions.rate_const,
                                     self.coef_reactions.rate_const))


# ----------------------------------------------------------------------------------------------- #
