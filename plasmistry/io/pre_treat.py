#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  9:09 2019/7/10

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import re


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


def _remove_comments(lines):
    r"""
    Remove the comments.
    """
    assert isinstance(lines, str)
    comment_regexp = re.compile(r"""
    [#].* $   # a line with '#' inside.
    """, re.VERBOSE | re.MULTILINE)
    return comment_regexp.sub("", lines)


def _remove_blank_line(lines):
    r"""
    Remove the blank lines.
    """
    assert isinstance(lines, str)
    blank_line_regexp = re.compile(r"""
    \n{2,}
    """, re.VERBOSE | re.MULTILINE)
    return blank_line_regexp.sub("\n", lines)


def _merge_lines_end_in_backslash(lines):
    r"""
    Merge the line end in '\' with the following one.
    """
    assert isinstance(lines, str)
    continuous_regexp = re.compile(r"""
    [ ]* \\ $\n     # the line end with '\'
    ^[ ]*           # the following line
    """, re.VERBOSE | re.MULTILINE)
    return continuous_regexp.sub(' ', lines)


# ----------------------------------------------------------------------------------------------- #
def treat_lines(lines):
    r"""
    Steps:
    1. remove comments.
    2. trim each line.
    3. remove blank line.
    4. merge line end in '\'.
    5. find all reaction block.
    """
    assert isinstance(lines, str)
    _lines = lines
    _lines = _remove_comments(_lines)
    _lines = _trim_lines(_lines)
    _lines = _remove_blank_line(_lines)
    _lines = _merge_lines_end_in_backslash(_lines)
    return _lines
