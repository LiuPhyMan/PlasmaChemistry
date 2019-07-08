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

str_to_match = r"""
E + H(2) => E + @A          !   cs_path     %CS_PATH%/H(n)_to_H(m)/H(2)_to_@A.csv
    @A = H(3) H(4)
@END

E + H(3) => E + @A          !   cs_path     %CS_PATH%/H(n)_to_H(m)/H(3)_to_@A.csv
    @A = H(4)
@END


# ----------------------------------------------------------------------------------------------- #
#   H2 VV
#       H2(v) + H2(w+1) => H2(v+1) + H2(w)
#   Reference
#       Matveyev, A.A. and V.P. Silakov, Kinetic Processes in Highly-Ionized Nonequilibrium
#       Hydrogen Plasma. Plasma Sources Science & Technhology, 1995. 4(4): p. 606-617.

# ----------------------------------------------------------------------------------------------- #
H2(v{v}) + H2(v{w1}) => H2(v{v1}) + H2(v{w})    !    \
                ({v}+1)*({w}+1)*kVV0110_H2*(1.5-0.5*exp(-delta*dv))*exp(Delta_1*dv-Delta_2*dv**2)
    @WHERE: dv = {v} - {w}
            delta = 0.21*sqrt(Tgas/300)
            Delta_1 = 0.236*(Tgas/300)**0.25
            Delta_2 = 0.0572*(300/Tgas)**(1/3)
    @LAMBDA: lambda x: [x.format(v=i, w=j, v1=i+1, w1=j+1).replace('H2(v0)', 'H2') \
                        for i in range(10) for j in range(14)]
@END
"""
# block_match = rctn_block_regexp.findall(str_to_match)

str_with_continue_line = r"""
  
a\
b\
 c
  d  
    \ e\
  f\
e
\
"""
continue_line = re.compile(r"""
^.+(?<!\\)$\n   # the precede line not end in '\'
(?:
    ^.+(?<=\\)$\n # the line end with '\'
)+        
^.+(?<!\\)$     # the following line not end in '\'
""", re.VERBOSE | re.MULTILINE)
merge_line_match = continue_line.findall(str_with_continue_line)
for _ in merge_line_match:
    print(_)
    print("NEXT...")

re.sub(r"[ ]+$", '', str_with_continue_line, flags=re.MULTILINE)
re.sub(r"^[ ]+", '', str_with_continue_line, flags=re.MULTILINE)
print(str_with_continue_line, end='%')


def _trim_lines(lines):
    r"""
    Examples
    --------
    '  ab'
    ' bc '
    '  ca'
    TO
    'ab'
    'bc'
    'ca'
    """
    assert isinstance(lines, str)
    trim_regexp = re.compile(r"""
    ^[ ]+ |
    [ ]+$
    """, re.VERBOSE | re.MULTILINE)
    return trim_regexp.sub("", lines)


def _remove_comment_line(lines):
    r"""
    Remove the comment lines.
    """
    comment_regexp = re.compile(r"""
    ^ [#].* $
    """, re.VERBOSE | re.MULTILINE)
    return comment_regexp.sub("", lines)


def _remove_blank_line(lines):
    r"""

    Parameters
    ----------
    lines

    Returns
    -------

    """
    blank_line_regexp = re.compile(r"""
    \n+
    """, re.VERBOSE | re.MULTILINE)
    return blank_line_regexp.sub("\n", lines)


def _merge_lines_end_in_backslash(lines):
    continuous_regexp = re.compile(r"""
    [ ]* \\ $\n
    ^[ ]*
    """, re.VERBOSE | re.MULTILINE)
    return continuous_regexp.sub(' ', lines)


def _find_all_rctn_block(lines):
    rctn_block_regexp = re.compile(r"""
    (?P<rctn>  ^.+!.*$) \n      # the reaction line
    (?P<block>              
        (?: .+(?<!@END)$ \n)*   # block_body
    )    
    (?P<end>^@END$)             # end of the block
    """, re.VERBOSE | re.MULTILINE)
    return rctn_block_regexp.findall(lines)

def treat_lines(lines):
    r"""
    Steps:
    1. trim each line.
    2. remove comment lines.
    3. remove blank line.
    4. merge line end in '\'.
    5. find all reaction block.
    """
    _lines = lines
    _lines = _trim_lines(_lines)
    _lines = _remove_comment_line(_lines)
    _lines = _remove_blank_line(_lines)
    _lines = _merge_lines_end_in_backslash(_lines)
    _rctn_block_list = _find_all_rctn_block(_lines)

