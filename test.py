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

treat_cmds = []
with open('_rctn_list\H2.inp') as f:
    for i, _line in enumerate(f):
        if i+1 == 117:
            origin_line = _line.strip()
        if 118 <= i + 1 <= 124:
            treat_cmds.append(_line)

treat_cmds = [_.strip() for _ in treat_cmds]
_total_str = '\n'.join(treat_cmds)
treat_cmds = re.findall(r"(?P<cmds>@(?:WHERE|LAMBDA)\s*:\s*[^@]+)", _total_str)


def treat_line(origin_line, treat_cmd):
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


_line = origin_line
for _cmd in treat_cmds:
    _line = treat_line(_line, _cmd)
# _treated_line = treat_line(origin_line, temp[0])
# _treated_line = treat_line(_treated_line, temp[1])
