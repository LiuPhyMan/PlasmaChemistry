#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  10:43 2019/7/8

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import re
from plasmistry.io import treat_lines


# str_to_match = r"""
# %CO2_vib% = \
#     {   CO2      CO2(v1)  CO2(v2)  CO2(v3)  CO2(v4)  CO2(v5)  CO2(v6)  CO2(v7)  CO2(v8)  CO2(v9)
#         CO2(v10) CO2(v11) CO2(v12) CO2(v13) CO2(v14) CO2(v15) CO2(v16) CO2(v17) CO2(v18) CO2(v19)
#         CO2(v20) CO2(v21)
#     }
# %H2_EleState% = \
# {   H2(B)   H2(C)   H2(B')  H2(D)   H2(B'') H2(D')
# }
# E + H(2) => E + @A          !   cs_path     %CS_PATH%/H(n)_to_H(m)/H(2)_to_@A.csv
#     @A = H(3) H(4)
# @END
#
#     E + H(3) => E + @A          !   cs_path     %CS_PATH%/H(n)_to_H(m)/H(3)_to_@A.csv
#     @A = H(4)
#
# @END # the end line
#
#
# # ----------------------------------------------------------------------------------------------- #
# #   H2 VV
# #       H2(v) + H2(w+1) => H2(v+1) + H2(w)
#     #       Reference
# #       Matveyev, A.A. and V.P. Silakov, Kinetic Processes in Highly-Ionized Nonequilibrium
# #       Hydrogen Plasma. Plasma Sources Science & Technhology, 1995. 4(4): p. 606-617.
#
# # ----------------------------------------------------------------------------------------------- #
# H2(v{v}) + H2(v{w1}) => H2(v{v1}) + H2(v{w})    !    \
#                 ({v}+1)*({w}+1)*kVV0110_H2*(1.5-0.5*exp(-delta*dv))*exp(Delta_1*dv-Delta_2*dv**2)
#     @WHERE: dv = {v} - {w}
#             delta = 0.21*sqrt(Tgas/300)
#             Delta_1 = 0.236*(Tgas/300)**0.25
#             Delta_2 = 0.0572*(300/Tgas)**(1/3)
#     @LAMBDA: lambda x: [x.format(v=i, w=j, v1=i+1, w1=j+1).replace('H2(v0)', 'H2') \
#                         for i in range(10) for j in range(14)]
#
# @END
# """

# ----------------------------------------------------------------------------------------------- #
def _find_all_rctn_block(lines):
    r"""
    """
    assert isinstance(lines, str)
    rctn_block_regexp = re.compile(r"""
    (?P<rctn>  ^.+!.*$) \n      # the reaction line
    (?P<block>              
        (?: .+(?<!@END)$ \n)*   # block_body
    )    
    (?P<end>^@END$)             # end of the block
    """, re.VERBOSE | re.MULTILINE)
    return rctn_block_regexp.findall(lines)


# ----------------------------------------------------------------------------------------------- #
def get_rctn_kstr(line):
    r"""
    #TODO
    """
    assert isinstance(line, str)
    rctn_line_regexp = re.compile(r"""
    ^  (?P<rcnt>.*?) 
    \s*=>\s*
     (?P<prdt>.*?) 
    \s* ! \s*
    (?P<kstr> .*?) $
    """, re.VERBOSE | re.MULTILINE)
    return rctn_line_regexp.match(line)


def get_abbr(lines):
    r"""

    Parameters
    ----------
    lines

    Returns
    -------

    """
    abbr_regexp = re.compile(r"""
    ^(?P<abbr>%\S+%) \s*=\s*
    {\s* 
    (?P<abbr_expr>[^{}]+?)
    \s*}
    """, re.VERBOSE | re.MULTILINE)
    abbr_list = abbr_regexp.findall(lines)
    abbr_list = [(_[0], re.sub(r"\s+", " ", _[1])) for _ in abbr_list]
    return abbr_list


def replace_abbr(abbr_list, lines):
    _lines = lines
    for abbr, expr in abbr_list:
        _lines = _lines.replace(abbr, expr)
    return _lines


class Reaction_block(object):
    cross_func_regexp = re.compile(r"""
    ^ @CROSS:\s*    # start
    (?P<cross_expr> 
        (?: @[A-Z]@ # abbr
        \s* =  \s*  # equals
        \S .*  \n   # expr_list
        ){2}        # repeat twice
    )
    """, re.VERBOSE | re.MULTILINE)
    # ------------------------------------------------------------------------------------------- #
    zip_func_regexp = re.compile(r"""
    ^ @ZIP:\s*      # start
    (?P<zip_expr>
        (?: @[A-Z]  # abbr
        \s* =  \s*  # equals
        \S .*  \n   # expr_list
        ){1,}       # at least one line
    )                  
    """, re.VERBOSE | re.MULTILINE)
    # ------------------------------------------------------------------------------------------- #
    where_func_regexp = re.compile(r"""
    ^ @WHERE:\s*    # start
    (?P<eqtns> 
        (?: \S+
        \s* =  \s*  # equals
        \S .*  \n   # expr_list
        ){1,}       # at least one line
    )
    """, re.VERBOSE | re.MULTILINE)
    # ------------------------------------------------------------------------------------------- #
    lambda_func_regexp = re.compile(r"""
    ^ @LAMBDA:\s*
    (?: lambda.*
    )
    """, re.VERBOSE | re.MULTILINE)

    def __init__(self, *, rctn_list, func_str):
        super().__init__()
        assert isinstance(rctn_list, list)
        assert isinstance(func_str, str)
        self.rctn_list = rctn_list
        self.func_str = func_str
        self.type = None

    def _apply_cross_func(self):
        temp = re.findall(r"""
        (?P<key> @[A-Z]@)
            \s* = \s*
        (?P<expr> \S.*) $
        """, self.func_str, flags=re.VERBOSE | re.MULTILINE)
        assert temp
        key_0 = temp[0][0]
        key_1 = temp[1][0]
        expr_list_0 = re.split(r"\s+", temp[0][1])
        expr_list_1 = re.split(r"\s+", temp[1][1])
        _cross_func = lambda x: [x.replace(key_0, _expr_0).replace(key_1, _expr_1)
                                 for _expr_0 in expr_list_0
                                 for _expr_1 in expr_list_1]
        output_list = []
        for _ in self.rctn_list:
            output_list = output_list + _cross_func(_)
        return output_list

    def _apply_zip_func(self):
        temp = re.findall(r"""
        (?P<key> @[A-Z])
            \s* = \s*
        (?P<expr> \S.*) $
        """, self.func_str, flags=re.VERBOSE | re.MULTILINE)
        assert temp
        abbr_list = [_[0] for _ in temp]
        expr_list = [re.split(r"\s+", _[1]) for _ in temp]

        # expr_list_0 = re.split(r"\s+", temp[0][1])
        # expr_list_1 = re.split(r"\s+", temp[1][1])

        def replace_ith_expr(i_index, line):
            _line = line
            ith_expr = [_[i_index] for _ in expr_list]
            for _abbr, _expr in zip(abbr_list, ith_expr):
                assert _abbr in _line
                _line = _line.replace(_abbr, _expr)
            return _line

        output_list = []
        for _ in self.rctn_list:
            for i_index in range(len(expr_list[0])):
                output_list.append(replace_ith_expr(i_index, _))
        return output_list

    def _apply_where_func(self):
        temp = re.findall(r"""
        (?P<var> \S+ )
            \s* = \s*
        (?P<expr> \S.*) $
        """, self.func_str, flags=re.VERBOSE | re.MULTILINE)
        assert temp
        vari_list = [_[0] for _ in temp]
        expr_list = [_[1] for _ in temp]
        vari_list.reverse()
        expr_list.reverse()
        output_list = []
        for _ in self.rctn_list:
            _line = _
            for _var, _expr in zip(vari_list, expr_list):
                _line = _line.replace(_var, f"({_expr})")
            output_list.append(_line)
        return output_list

    def _apply_lambda_func(self):
        pass

    def apply_func(self):
        if self.cross_func_regexp.fullmatch(self.func_str):
            return self._apply_cross_func()
        elif self.zip_func_regexp.fullmatch(self.func_str):
            return self._apply_zip_func()
        elif self.where_func_regexp.fullmatch(func_str):
            pass
        elif self.lambda_func_regexp.fullmatch(func_str):
            pass
        else:
            raise Exception("The func_str is error.")

    # ------------------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    # replc_regexp = re.compile(r"""
    # ^ @[A-Z]@?
    # \s*=\s*
    # .* $
    # """, re.VERBOSE | re.MULTILINE)
    # ------------------------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    with open("_rctn_list/H2.inp") as f:
        str_to_match = "".join(f.readlines())
    lines = treat_lines(str_to_match)
    # temp = get_rctn_kstr(line_list[1][0])
    abbr_list = get_abbr(lines)
    lines = replace_abbr(abbr_list, lines)
    rctn_block_list = _find_all_rctn_block(lines)

    rctn_list = ["E + @A@ => E + H + @B@", "@A@ => @B@"]
    func_str = "@CROSS: @A@ = A B C\n@B@ = X Y Z\n"
    cross_rctn_block = Reaction_block(rctn_list=rctn_list,
                                      func_str=func_str)

    rctn_list = ["E + @A => E + H + @B ! @C", "@A => @B + @C"]
    zip_func_str = "@ZIP: @A = A B C\n@B = X Y Z\n@C = D E F"
    zip_rctn_block = Reaction_block(rctn_list=rctn_list,
                                    func_str=zip_func_str)
    rctn_list = ["a => b", "a*b = c"]
    where_func_str = "@WHERE: a = 1\n b = 2 \n c=(a+b)"
    where_rctn_block = Reaction_block(rctn_list=rctn_list,
                                      func_str=where_func_str)
