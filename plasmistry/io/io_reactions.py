# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:29:46 2016

@author: ljb
"""
import math
import re

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------------------- #
class IoReactionsError(Exception):
    pass


def fortran2python(_expr):
    return re.sub(r"(?<=[\d\.])d(?=[\+\-\d])", "e", _expr)


def __get_delete_dH(_str):
    dH_str = re.findall(r"(?P<dH>\S*)_eV", _str)[0]
    dH = float(fortran2python(dH_str))
    _str_deleted = re.sub(r"(?:\s+\+\s+)?\S+_eV", " ", _str).strip()
    return dH, _str_deleted


# ----------------------------------------------------------------------------------------------- #
def __read_rcnt_prdt_dH_kStr(reaction_str):
    r"""
    Read reaction string.

    Parameters
    ----------
    reaction_str : str
        Reaction string like 'CO + O => CO2 + 2_eV ! 1.0E3'.

    Returns
    -------
    Result : (str, str, float, str)
        (reactant, product, reaction_enthalpy, rate_constant_string)

    Examples
    --------
    A + B => C + D + 1e2_eV ! 1.0d2
        reactant : A + B
        product : C + D
        reaction_enthalpy : -1e2
        k_str : 1.0d2

    """
    assert isinstance(reaction_str, str)
    assert reaction_str.count('=>') == 1
    assert reaction_str.count('!') <= 1
    assert reaction_str.count('_eV') <= 1
    # ------------------------------------------------------------------------------------------- #
    rctn_regexp = re.compile(
            r"\s*{rcnt}{sep}{prdt}\s*(?:[!]\s*{k_str})?\s*".format(rcnt=r"(?P<reactant>.*?)",
                                                                   sep=r"\s*\<?\=\>\s*",
                                                                   prdt=r"(?P<product>.*?)",
                                                                   k_str=r"(?P<k_str>.*?)"))
    rcnt_str, prdt_str, k_str = [rctn_regexp.fullmatch(reaction_str).groupdict()[_]
                                 for _ in ('reactant', 'product', 'k_str')]

    # ------------------------------------------------------------------------------------------- #

    # ------------------------------------------------------------------------------------------- #
    if '_eV' in rcnt_str:
        dH, rcnt_str = __get_delete_dH(rcnt_str)
        dH = dH * (+1)
    elif '_eV' in prdt_str:
        dH, prdt_str = __get_delete_dH(prdt_str)
        dH = dH * (-1)
    else:
        dH = 0.0
    k_str = "" if k_str is None else fortran2python(k_str)
    return rcnt_str, prdt_str, dH, k_str


# ----------------------------------------------------------------------------------------------- #
def __read_reactionlist_block(line, replc_input):
    r"""
    Read reactions block.

    Parameters
    ----------
    line : str
        Reaction string.
    replc_input : list of str
        Substitutions of @[A-Z]
        format :
            ['@A = a b c'
             '@B = 1 2 3']

    Returns
    -------
    Result : tuple of list
        (rcntM, prdtM, dHM, k_strM)
            rcntM : list of str
                reactants
            prdtM : list of str
                product
            dHM : list of float
                reaction enthalpy
            k_strM : list of str
                string of equation for rate constant calculation

    Notes
    -----
    [0] A + @B => C + @A
            @B = H O N
            @A = X Y Z
    equal to
        A + H => C + X ;    A + O => C + Y ;    A + N => C + Z
    [1] A + B(@M@) => C + D(V@N@)
            @M@ = 1 2 3
            @N@ = a b c
    equal to
        Cartesian Product: set{@M@} X set{@N@}
        A + B(1) => C + D(Va) ; A + B(1) => C + D(Vb);  A + B(1) => C + D(Vc)
        A + B(2) => C + D(Va) ; A + B(2) => C + D(Vb);  A + B(2) => C + D(Vc)
        A + B(3) => C + D(Va) ; A + B(3) => C + D(Vb);  A + B(3) => C + D(Vc)

    """
    assert isinstance(line, str)
    assert isinstance(replc_input, list)
    assert '@' in line

    # ------------------------------------------------------------------------------------------- #
    if re.findall(r"@[A-Z]@", line):
        # --------------------------------------------------------------------------------------- #
        #   Replace @[A-Z]@ to generate reaction lines
        #   Examples
        #   --------
        #       replc_input = ['@M@ = 1 2 3',
        #                      '@N@ = a b c']
        # --------------------------------------------------------------------------------------- #
        assert len(set(re.findall(r"@[A-Z]@", line))) == 2
        assert len(replc_input) >= 2
        substi_list_0 = replc_input[0].split()[2:]
        substi_sign_0 = replc_input[0].split()[0]
        substi_list_1 = replc_input[1].split()[2:]
        substi_sign_1 = replc_input[1].split()[0]
        has_condition = False
        if replc_input[-1].startswith("@CONDITION"):
            has_condition = True
            _temp = re.split(r":", replc_input[-1])[1].strip()
            _condition = _temp.replace(substi_sign_0, '_sign_0').replace(substi_sign_1, '_sign_1')
        reaction_lines = []
        for _sign_0 in substi_list_0:
            for _sign_1 in substi_list_1:
                if has_condition:
                    if not eval(_condition):
                        continue
                reaction_lines.append(line.replace(substi_sign_0,
                                                   _sign_0).replace(substi_sign_1, _sign_1))

    elif re.findall('@[A-Z]', line):
        # --------------------------------------------------------------------------------------- #
        #   Replace @[A-Z] to generate reaction lines
        #   Examples
        #   --------
        #       replc_input = ['@A = a b c',
        #                      '@B = 1 2 3']
        #       replc_list = [['A', '=', 'a', 'b', 'c'],
        #                     ['B', '=', '1', '2', '3']]
        # --------------------------------------------------------------------------------------- #
        assert len(set(re.findall('@[A-Z]', line))) == len(replc_input)
        replc_arr = np.array([_.strip()[1:].split() for _ in replc_input])
        assert replc_arr.ndim == 2
        reaction_lines = []
        for i_lines in range(2, len(replc_arr[0])):
            replc_dict = {_key: _value for _key, _value in
                          zip(replc_arr[:, 0], replc_arr[:, i_lines])}
            new_line = re.sub(r"@(?P<label>[A-Z])", '{\g<label>}', line)
            reaction_lines.append(new_line.format(**replc_dict))
    else:
        raise IoReactionsError("The {} block is error".format(line))
    # Read rcnt prdt dH k_str of lines
    result = [__read_rcnt_prdt_dH_kStr(_) for _ in reaction_lines]
    rcntM, prdtM, dHM, k_strM = [[s[i] for s in result] for i in range(4)]
    return rcntM, prdtM, dHM, k_strM


# ----------------------------------------------------------------------------------------------- #
def read_reactionFile(file_path, start_line=-math.inf, end_line=math.inf):
    r"""
    Read a file of reactions from start_line position to end_line position

    Parameters
    ----------
    file_path : str
        Reaction file path.
    start_line : int or -math.inf
        Position where it starts reading.
    end_line : int or math.inf
        Position where it ends reading.

    Returns
    -------
    Results : tuple
        (rcntM, prdtM, dHM, k_strM, pre_exec_list)
            rcntM : pandas.Series
                reactants
            prdtM : pandas.Series
                product
            dHM : pandas.Series
                enthalpy of reaction
            k_strM : pandas.Series
                string of rate constant calculation
            pre_exec_list : list
                pre-execution statements list

    """
    assert isinstance(file_path, str)
    assert start_line == -math.inf or (isinstance(start_line, int) and start_line > 0)
    assert end_line == math.inf or (isinstance(end_line, int) and end_line > 0)
    assert end_line > start_line

    # ------------------------------------------------------------------------------------------- #
    rcntM, prdtM, dHM, k_strM = pd.Series(), pd.Series(), pd.Series(), pd.Series()
    pre_exec_list = []
    envir_vars = dict()

    def replace_envir_vars(_str):
        init_str = _str
        for _key in envir_vars:
            init_str = init_str.replace(_key, envir_vars[_key])
        return init_str

    # read rctn_list in the range.
    rctn_list = []
    with open(file_path) as f:
        for _i, line in enumerate(f):
            if _i < start_line - 1:
                continue
            if _i > end_line - 1:
                break
            rctn_list.append(line.strip())
    # remove all comment.
    rctn_list = [_.strip() for _ in rctn_list if not _.startswith('#')]
    ##
    i_line = 0
    while i_line < len(rctn_list):
        line = rctn_list[i_line]
        # --------------------------------------------------------------------------------------- #
        #   Abbreviation
        # --------------------------------------------------------------------------------------- #
        if re.match("%\S+%\s+=\s+", line):
            abbr_str = line
            if ('{' in abbr_str) and ('}' not in abbr_str):
                while '}' not in rctn_list[i_line]:
                    abbr_str = abbr_str + ' ' + rctn_list[i_line + 1]
                    i_line += 1
                i_line -= 1
            temp = re.match(r"%(?P<key>\S+)%\s+=\s+{(?P<abbr>[^}]+)}\s*", abbr_str)
            key = temp.groupdict()['key']
            abbr = temp.groupdict()['abbr']
            assert key not in envir_vars
            envir_vars[key] = abbr.strip()

        # --------------------------------------------------------------------------------------- #
        #   Pre-execution statement
        # --------------------------------------------------------------------------------------- #
        if line.startswith('$'):
            subline = line[1:].strip()
            pre_exec_list.append(subline)

        # --------------------------------------------------------------------------------------- #
        #   Read reactions
        # --------------------------------------------------------------------------------------- #
        if '=>' in line:
            lamb_series_append = lambda x, _sers: _sers.append(pd.Series(x), ignore_index=True)
            if '@' not in line:
                # ------------------------------------------------------------------------------- #
                #   read line
                # ------------------------------------------------------------------------------- #
                rcnt, prdt, dH, k_str = __read_rcnt_prdt_dH_kStr(replace_envir_vars(line))
                rcntM, prdtM, dHM, k_strM = (lamb_series_append(x, xM) for x, xM in
                                             zip([rcnt, prdt, dH, k_str],
                                                 [rcntM, prdtM, dHM, k_strM]))
            else:
                # --------------------------------------------------------------------------- #
                #   read block
                # --------------------------------------------------------------------------- #
                replc_strM = set(re.findall(r'@[A-Z]@?', line))
                replc_input = []

                while rctn_list[i_line+1].startswith('@'):
                    replc_input.append(rctn_list[i_line+1])
                    i_line += 1
                #TODO
                for _ in replc_strM:
                    _line = f.readline().strip()
                    assert _line.startswith('@')
                    replc_input.append(_line)
                _line = f.readline().strip()
                if _line.startswith('@CONDITION'):
                    replc_input.append(_line)
                sub_rcntM, sub_prdtM, sub_dHM, sub_k_strM = \
                    __read_reactionlist_block(replace_envir_vars(line), replc_input)
                rcntM, prdtM, dHM, k_strM = (lamb_series_append(x, xM)
                                             for x, xM in
                                             zip([sub_rcntM, sub_prdtM, sub_dHM, sub_k_strM],
                                                 [rcntM, prdtM, dHM, k_strM]))
                i_line += len(replc_input)
        i_line += 1

    if len(rcntM) == 0:
        raise IoReactionsError('None reactions found in ' + file_path)
    else:
        reaction_info = pd.DataFrame(dict(reactant=rcntM,
                                          product=prdtM,
                                          dH=dHM,
                                          k_str=k_strM))
        reaction_info = reaction_info[['reactant', 'product', 'dH', 'k_str']]
        return dict(reaction_info=reaction_info,
                    pre_exec_list=pre_exec_list)


# ----------------------------------------------------------------------------------------------- #
def read_reactionList(reaction_list):
    r"""
    Read list of reaction string.

    Parameters
    ----------
    reaction_list : list of str

    Returns
    -------
    Results : tuple of pandas.Series
        (rcntM, prdtM, dHM, k_strM)
            rcntM : pandas.Series
                reactants
            prdtM : pandas.Series
                product
            dHM : pandas.Series
                enthalpy of reaction
            k_strM : pandas.Series
                string of rate constant calculation

    """
    assert isinstance(reaction_list, list)
    rcntM, prdtM, dHM, k_strM = pd.Series(), pd.Series(), pd.Series(), pd.Series()
    lamb_series_append = lambda x, _series: _series.append(pd.Series(x), ignore_index=True)
    for line in reaction_list:
        assert isinstance(line, str) and '=>' in line
        rcnt, prdt, dH, k_str = __read_rcnt_prdt_dH_kStr(line)
        rcntM, prdtM, dHM, k_strM = (lamb_series_append(x, xM) for x, xM in
                                     zip([rcnt, prdt, dH, k_str], [rcntM, prdtM, dHM, k_strM]))

    reaction_info = pd.DataFrame(dict(reactant=rcntM,
                                      product=prdtM,
                                      dH=dHM,
                                      k_str=k_strM))
    reaction_info = reaction_info[['reactant', 'product', 'dH', 'k_str']]
    return dict(reaction_info=reaction_info)

# ----------------------------------------------------------------------------------------------- #
# def __instance_from_rcnt_prdt_dH_k_str(rcnt, prdt, dH, k_str, *, pre_exec_list, class_name):
#     r"""
#     Instance
#     Parameters
#     ----------
#     rcnt
#
#     prdt
#
#     dH
#
#     k_str
#
#     pre_exec_list
#
#     class_name
#
#     Returns
#     -------
#
#     """
#     if issubclass(class_name, CrosReactions):
#         return class_name(reactant=rcnt, product=prdt, k_str=k_str, dH_e=dH)
#     elif issubclass(class_name, CoefReactions):
#         return class_name(reactant=rcnt, product=prdt, k_str=k_str, dH_g=dH)
#     elif issubclass(class_name, MixReactions):
#         cros_bool = k_str.str.startswith('BOLSIG')
#         coef_bool = ~cros_bool
#         lamb_bool_index = lambda _bool, _series: _series[_bool].reset_index(drop=True)
#         assert (not cros_bool.all()) and (not coef_bool.all()), "MixReactions is not suited."
#         cros_instance = CrosReactions(reactant=lamb_bool_index(cros_bool, rcnt),
#                                       product=lamb_bool_index(cros_bool, prdt),
#                                       k_str=lamb_bool_index(cros_bool, k_str),
#                                       dH_e=lamb_bool_index(cros_bool, dH))
#         coef_instance = CoefReactions(reactant=lamb_bool_index(coef_bool, rcnt),
#                                       product=lamb_bool_index(coef_bool, prdt),
#                                       k_str=lamb_bool_index(coef_bool, k_str),
#                                       dH_g=lamb_bool_index(coef_bool, dH))
#         coef_instance.set_pre_exec_list(pre_exec_list)
#         coef_instance.compile_k_str()
#         return dict(coef_reactions=coef_instance,
#                     cros_reactions=cros_instance)
#     else:
#         raise IoReactionsError('Class name is error.')
#
#
# def instance_reactionFile(*, file_path, class_name, start_line=-math.inf, end_line=math.inf):
# r"""
# Read reactions from file_path and instance it in a specific class.
#
# Parameters
# ----------
# file_path : str
#     Reaction file path
# class_name : Reactions class
#
# start_line
#
# end_line
#
# Returns
# -------
#
# """
# assert isinstance(file_path, str)
# assert issubclass(class_name, Reactions)
#
# rcnt, prdt, dH, k_str, pre_exec_list = read_reactionFile(file_path, start_line, end_line)
# return __instance_from_rcnt_prdt_dH_k_str(rcnt, prdt, dH, k_str,
#                                           pre_exec_list=pre_exec_list,
#                                           class_name=class_name)


# ----------------------------------------------------------------------------------------------- #
# def instance_reactionList(*, reaction_list, class_name):
# r"""
# Read reactions from file_path and instance it in a specific class.
#
# Parameters
# ----------
# reaction_list : list
#     Reactions list.
# class_name
#
# Returns
# -------
#
# """
# assert isinstance(reaction_list, list)
# assert issubclass(class_name, Reactions)
#
# rcnt, prdt, dH, k_str = read_reactionList(reaction_list)
# return __instance_from_rcnt_prdt_dH_k_str(rcnt, prdt, dH, k_str,
#                                           pre_exec_list=[],
#                                           class_name=class_name)

# ----------------------------------------------------------------------------------------------- #
