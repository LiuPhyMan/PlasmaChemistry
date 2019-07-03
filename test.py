#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20:09 2018/10/25

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import re
from plasmistry.io import read_reactionFile
from plasmistry import constants as const


def treat_where_or_lambda_cmds(origin_line, treat_cmd):
    _line = origin_line
    if treat_cmd.startswith("@WHERE"):
        _args = re.fullmatch(r"@WHERE\s*:\s*(?P<cmds>(?:[^\n]+\n)+)", treat_cmd)
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


def treat_multi_cmds(origin_line, _cmds):
    treat_cmds = [_.strip() for _ in _cmds]
    treat_cmds = re.findall(r"(?P<cmds>@(?:WHERE|LAMBDA)\s*:\s*[^@]+)", '\n'.join(treat_cmds))
    _line = origin_line
    for _cmd in treat_cmds:
        _line = treat_where_or_lambda_cmds(_line, _cmd)
    return _line


if __name__ == "__main__":
    with open("_rctn_list\H2.inp") as f:
        lines = ''.join(f.readlines())
        lines = re.sub('\\\\\s*\n\s*', ' ', lines)  # merge the lines end with \
        lines = re.sub('\s*\n\s*', '\n', lines)     # trip the lines
        lines = lines.split('\n')

    # temp = read_reactionFile('_rctn_list\H2.inp', start_line=123, end_line=131)

    # a = temp['reaction_info']
    # b = temp['pre_exec_list']
    # with open("_rctn_list/CO2_chemistry.gum") as f:
    #     lines = ''.join(f.readlines())
    #
    # a = re.findall(r"Reaction\s*{[^{}]+{[^{}]+}[^{}]+}", lines)
    # for _ in a:
    #     print(re.findall(r"Format[^\n]+\n", _))
