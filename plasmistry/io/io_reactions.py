# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:29:46 2016

@author: ljb
"""
import platform
import math
import re

import numpy as np
import pandas as pd

number_regexp = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"


# --------------------------------------------------------------------------- #
class IoReactionsError(Exception):
    pass


def fortran2python(_expr):
    r"""
    Convert 2.d-2 to 2.e-2
    """
    return re.sub(r"(?<=[\d.])[dD](?=[\d+-])", "e", _expr)


def __treat_where_or_lambda_cmd(origin_line, treat_cmd):
    r"""

    Parameters
    ----------
    origin_line
    treat_cmd

    Returns
    -------

    """
    _line = origin_line
    if treat_cmd.startswith("@WHERE"):
        _args = re.fullmatch(r"@WHERE\s*:\s*(?P<cmds>(?:[^\n]+\n)+)",
                             treat_cmd)
        _cmds = _args.groupdict()['cmds'].strip().split(sep='\n')
        _cmds.reverse()
        for _ in _cmds:
            _var = _.split(sep='=')[0].strip()
            _str = _.split(sep='=')[1].strip()
            assert _var in _line, _var
            _line = _line.replace(_var, '( ' + _str.replace(' ', '') + ' )')
        return _line
    elif treat_cmd.startswith("@LAMBDA"):
        _args = re.fullmatch(r"@LAMBDA\s*:\s*(?P<cmds>[\s\S]*)", treat_cmd)
        assert _args, _args
        lambda_cmd = _args.groupdict()['cmds'].strip()
        treat_func = eval(lambda_cmd)
        return treat_func(origin_line)


def __treat_multi_cmds(origin_line, _cmds):
    treat_cmds = [_.strip() for _ in _cmds]
    treat_cmds = re.findall(r"(?P<cmds>@(?:WHERE|LAMBDA)\s*:\s*[^@]+)",
                            '\n'.join(treat_cmds))
    _line = origin_line
    for _cmd in treat_cmds:
        _line = __treat_where_or_lambda_cmd(_line, _cmd)
    return _line


def __get_delete_dH(_str):
    dH_str = re.findall(r"(?P<dH>\S*)_eV", _str)[0]
    dH = float(fortran2python(dH_str))
    _str_deleted = re.sub(r"(?:\s+\+\s+)?\S+_eV", " ", _str).strip()
    return dH, _str_deleted


# --------------------------------------------------------------------------- #
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
    # ------------------------------------------------------------------------- #
    rctn_regexp = re.compile(
        r"\s*{rcnt}{sep}{prdt}\s*(?:[!]\s*{k_str})?\s*".format(
            rcnt=r"(?P<reactant>.*?)",
            sep=r"\s*\<?\=\>\s*",
            prdt=r"(?P<product>.*?)",
            k_str=r"(?P<k_str>.*?)"))
    rcnt_str, prdt_str, k_str = [
        rctn_regexp.fullmatch(reaction_str).groupdict()[_]
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


# --------------------------------------------------------------------------- #
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
    # assert '@' in line

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
            _temp = re.split(r":", replc_input[-1], maxsplit=1)[1].strip()
            _condition = _temp.replace(substi_sign_0, '_sign_0').replace(
                substi_sign_1, '_sign_1')
        reaction_lines = []
        for _sign_0 in substi_list_0:
            for _sign_1 in substi_list_1:
                if has_condition:
                    print(_sign_0, end=' ')
                    print(_sign_1)
                    if not eval(_condition):
                        continue
                reaction_lines.append(line.replace(substi_sign_0,
                                                   _sign_0).replace(
                    substi_sign_1, _sign_1))

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
    elif re.match(r"@(WHERE|LAMBDA)", replc_input[0]):
        reaction_lines = __treat_multi_cmds(line, replc_input)
    else:
        raise IoReactionsError("The {} block is error".format(line))
    # Read rcnt prdt dH k_str of lines
    result = [__read_rcnt_prdt_dH_kStr(_) for _ in reaction_lines]
    rcntM, prdtM, dHM, k_strM = [[s[i] for s in result] for i in range(4)]
    return rcntM, prdtM, dHM, k_strM


# --------------------------------------------------------------------------- #
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
    assert start_line == -math.inf or (
            isinstance(start_line, int) and start_line > 0)
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

    # ------------------------------------------------------------------------------------------- #
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
    # ------------------------------------------------------------------------------------------- #
    ##
    i_line = 0
    while i_line < len(rctn_list):
        line = rctn_list[i_line]
        # --------------------------------------------------------------------------------------- #
        #   Abbreviation
        #   e.g.
        #       %CS_PATH% = \
        #           { D:/Coding/Python_code/cs_data
        #             /home/liujinbao/Documents/CODE/PlasmaChemistry/CrossSectionFile/H2
        #           }
        #       %H2_vib% = \
        #           {   H2      H2(v1)  H2(v2)  H2(v3)  H2(v4)  H2(v5) H2(v6) H2(v7) H2(v8) H2(v9)
        #               H2(v10) H2(v11) H2(v12) H2(v13) H2(v14)
        #           }
        # --------------------------------------------------------------------------------------- #
        if re.match(r"(?P<abbr>%\S+%)\s+=\s+\\?", line):
            abbr_str = line
            if '}' not in abbr_str:
                while '}' not in rctn_list[i_line]:
                    abbr_str = abbr_str + ' ' + rctn_list[i_line + 1]
                    i_line += 1
                i_line -= 1
            temp = re.match(
                r"(?P<key>%\S+%)\s+=\s+(\\\s+)?{(?P<abbr>[^}]+)}\s*", abbr_str)
            assert temp, abbr_str
            key = temp.groupdict()['key']
            abbr = temp.groupdict()['abbr']
            # ----------------------------------------------------------------------------------- #
            #   cs_path is different for windows and linux
            # ----------------------------------------------------------------------------------- #
            if key == '%CS_PATH%':
                if platform.platform().startswith('Windows'):
                    abbr = abbr.strip().split()[0]
                elif platform.platform().startswith("Linux"):
                    abbr = abbr.strip().split()[1]
                else:
                    raise Exception("The {s} is not supported.".format(
                        s=platform.platform()))

            assert key not in envir_vars
            envir_vars[key] = abbr.strip()

        # --------------------------------------------------------------------------------------- #
        #   Pre-execution statement
        # --------------------------------------------------------------------------------------- #
        if line.startswith('$'):
            pre_exec_list.append(line[1:].strip())

        # --------------------------------------------------------------------------------------- #
        #   Read reactions
        # --------------------------------------------------------------------------------------- #
        if '=>' in line:  # start reading reactions
            lamb_series_append = lambda x, _sers: _sers.append(pd.Series(x),
                                                               ignore_index=True)
            if not rctn_list[i_line + 1].startswith("@"):  # read a simple one.
                # ------------------------------------------------------------------------------- #
                #   read line
                # ------------------------------------------------------------------------------- #
                rcnt, prdt, dH, k_str = __read_rcnt_prdt_dH_kStr(
                    replace_envir_vars(line))
                rcntM, prdtM, dHM, k_strM = (lamb_series_append(x, xM) for
                                             x, xM in
                                             zip([rcnt, prdt, dH, k_str],
                                                 [rcntM, prdtM, dHM, k_strM]))
            else:  # startswith("@")
                # ------------------------------------------------------------------------------- #
                #   read block
                # ------------------------------------------------------------------------------- #
                replc_strM = set(re.findall(r'@[A-Z]@?', line))

                replc_input = []
                if re.match(r"@(WHERE|LAMBDA)", rctn_list[i_line + 1]):
                    while not rctn_list[i_line + 1].startswith("@END"):
                        replc_input.append(
                            replace_envir_vars(rctn_list[i_line + 1]))
                        i_line += 1
                    i_line -= 1
                elif re.match(r"@([A-Z]@?|CONDITION)", rctn_list[i_line + 1]):
                    while re.match(r"@([A-Z]@?|CONDITION).*",
                                   rctn_list[i_line + 1]):
                        replc_input.append(
                            replace_envir_vars(rctn_list[i_line + 1]))
                        i_line += 1
                    i_line -= 1
                assert len(replc_input) >= len(replc_strM), replc_input
                sub_rcntM, sub_prdtM, sub_dHM, sub_k_strM = \
                    __read_reactionlist_block(replace_envir_vars(line),
                                              replc_input)
                rcntM, prdtM, dHM, k_strM = (lamb_series_append(x, xM)
                                             for x, xM in
                                             zip([sub_rcntM, sub_prdtM,
                                                  sub_dHM, sub_k_strM],
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


# --------------------------------------------------------------------------- #
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
    lamb_series_append = lambda x, _series: _series.append(pd.Series(x),
                                                           ignore_index=True)
    for line in reaction_list:
        assert isinstance(line, str) and '=>' in line
        rcnt, prdt, dH, k_str = __read_rcnt_prdt_dH_kStr(line)
        rcntM, prdtM, dHM, k_strM = (lamb_series_append(x, xM) for x, xM in
                                     zip([rcnt, prdt, dH, k_str],
                                         [rcntM, prdtM, dHM, k_strM]))

    reaction_info = pd.DataFrame(dict(reactant=rcntM,
                                      product=prdtM,
                                      dH=dHM,
                                      k_str=k_strM))
    reaction_info = reaction_info[['reactant', 'product', 'dH', 'k_str']]
    return dict(reaction_info=reaction_info)


# ============================================================================ #
#   Reaction block
# ============================================================================ #
class Reaction_block(object):

    def __init__(self, *, rctn_dict=None, vari_dict=None):
        r"""
        
        """
        super().__init__()
        self._type = None
        self._formula = None
        self._kstr = None

        self._type_list = None
        self._formula_list = None
        self._kstr_list = None

        self._vari_dict = vari_dict
        if rctn_dict is not None:
            self.rctn_dict = rctn_dict
            self._formula = rctn_dict['formula']
            self._kstr = rctn_dict['kstr']
            self._treat_where_vari()
            self._treat_iterator()
            self._treat_where_abbr()

    @property
    def _reactant_str_list(self):
        return [re.split(r"\s*<?=>\s*", _)[0] for _ in self._formula_list]

    @property
    def _product_str_list(self):
        return [re.split(r"\s*<?=>\s*", _)[1] for _ in self._formula_list]

    @property
    def size(self):
        assert len(self._formula_list) == len(self._kstr_list)
        return len(self._formula_list)

    def __add__(self, other):
        result = Reaction_block()
        result._formula_list = self._formula_list + other._formula_list
        result._kstr_list = self._kstr_list + other._kstr_list
        result._type_list = self._type_list + other._type_list
        return result

    def add_variable_dict(self, *, _dict):
        self._variable_dict = _dict

    def _treat_iterator(self):
        if 'iterator' not in self.rctn_dict:
            self._formula_list = [self._formula, ]
            self._kstr_list = [self._kstr, ]
            self._type_list = [self.rctn_dict['type'], ]
            return None
        else:
            _iter = self.rctn_dict['iterator']
            if 'formula' in _iter['repl']:
                _formula_list = eval(self.repl_func(self._formula,
                                                    _iter['repl']['formula'],
                                                    _iter),
                                     self._vari_dict)
            if 'kstr' in _iter['repl']:
                _kstr_list = eval(self.repl_func(self._kstr,
                                                 _iter['repl']['kstr'],
                                                 _iter),
                                  self._vari_dict)
        self._formula_list = _formula_list
        self._kstr_list = _kstr_list
        self._type_list = [self.rctn_dict['type'] for _ in
                           range(len(self._kstr_list))]

    def _treat_where_abbr(self):
        r"""
        Notes
        -----
        _formula_list to _formula_list
        _kstr_list to _kstr_list

        """
        if 'where' in self.rctn_dict:
            if 'abbr' in self.rctn_dict['where']:
                for _key in self.rctn_dict['where']['abbr']:
                    _value = self.rctn_dict['where']['abbr'][_key]
                    self._formula_list = [_.replace(_key, _value) for _ in
                                          self._formula_list]
                    self._kstr_list = [_.replace(_key, _value) for _ in
                                       self._kstr_list]

    def _treat_where_vari(self):
        r"""
        Notes
        -----
        _kstr to _kstr

        """
        if 'where' in self.rctn_dict:
            if 'vari' in self.rctn_dict['where']:
                reversed_vari_list = self.rctn_dict['where']['vari'][::-1]
                for _key_value in reversed_vari_list:
                    _key = list(_key_value.items())[0][0]
                    _value = f"({str(list(_key_value.items())[0][1])})"
                    self._kstr = self._kstr.replace(_key, _value)

    @staticmethod
    def repl_func(x, _repl, _iter):
        _str_expr = f"'{x}'." + '.'.join(
            [f"replace('{k}', str({v}))" for k, v in _repl.items()])
        # product loop
        _iter_loop = _iter['loop']
        if 'product' in _iter_loop:
            _loop_dict = _iter_loop['product']
            _loop_expr = ' '.join([f'for {key} in {value}'
                                   for key, value in _loop_dict.items()])
        # zip loop
        elif 'zip' in _iter_loop:
            _loop_dict = _iter_loop['zip']
            _loop_expr = 'for {key} in zip({value})'.format(
                key=', '.join(_loop_dict.keys()),
                value=', '.join(_loop_dict.values()))
        # else
        else:
            raise Exception(f"product or zip is not in loop. {_iter}")
        # condition
        if 'condition' in _iter:
            _expr = f"[{_str_expr} {_loop_expr} if {_iter['condition']}]"
        else:
            _expr = f"[{_str_expr} {_loop_expr}]"
        return _expr


# --------------------------------------------------------------------------- #
class Cros_Reaction_block(Reaction_block):

    def __init__(self, *, rctn_dict=None, vari_dict=None):
        super().__init__(rctn_dict=rctn_dict, vari_dict=vari_dict)
        if rctn_dict is not None:
            self._threshold = self.rctn_dict["threshold"]
        self._set_threshold_list()

    def __add__(self, other):
        result = Cros_Reaction_block()
        result._formula_list = self._formula_list + other._formula_list
        result._kstr_list = self._kstr_list + other._kstr_list
        result._type_list = self._type_list + other._type_list
        result._threshold_list = self._threshold_list + other._threshold_list
        return result

    def _set_threshold_list(self):
        self._threshold = self.rctn_dict['threshold']
        _iter = self.rctn_dict['iterator'] 
        if 'threshold' in _iter['repl']:
            _eval_str = self.repl_func(self._threshold,
                                       _iter['repl']['threshold'],
                                       _iter)
            self._threshold_list = eval(_eval_str, self._vari_dict)
        else:
            self._threshold_list = self._threshold

    def generate_crostn_dataframe(self, *, factor=1):
        _df = dict()
        _df["formula"] = self._formula_list
        _df["type"] = self._type_list
        _df["threshold_eV"] = self._threshold_list
        _df["cross_section"] = [
            np.vstack((np.loadtxt(_path, comments="#")[:, 0],
                       np.loadtxt(_path, comments="#")[:, 1] * factor))
            for _path in self._kstr_list]
        _df = pd.DataFrame(data=_df, index=range(self.size))
        _df = _df.astype({'threshold_eV': np.float})
        return _df


# ---------------------------------------------------------------------------- #
class Coef_Reaction_block(Reaction_block):

    def __init__(self, *, rctn_dict=None, vari_dict=None):
        super().__init__(rctn_dict=rctn_dict, vari_dict=vari_dict)

    def generate_crostn_dataframe(self):
        _df = dict()
        _df["formula"] = self._formula_list

        _df['reactant'] = [re.split(r"\s*=>\s*", _)[0] for _ in
                           self._formula_list]
        _df['product'] = [re.split(r"\s*=>\s*", _)[1] for _ in
                          self._formula_list]

        _df["type"] = self._type_list
        _df["kstr"] = self._kstr_list
        return pd.DataFrame(data=_df, index=range(self.size))

# --------------------------------------------------------------------------- #
