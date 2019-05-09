# -*- coding: utf-8 -*-
"""
regular expressions to match tuples from user inputs
kindly provided by
Luis Carlos Garcia Peraza Herrera <luis.herrera.14@ucl.ac.uk>
"""
from __future__ import unicode_literals

import re

INT = r'(?:[-+]?\d+)'
FLOAT = r'(?:[-+]?\d*\.\d+|' + INT + r')'
LITERAL = r'(?:[-_a-zA-Z0-9 \\:\.]+)'
COMMA = r'(?:[,])'
LEFT_PARENTHESIS = r'(?:\()'
RIGHT_PARENTHESIS = r'(?:\))'
LEFT_BRACKET = r'(?:[{])'
RIGHT_BRACKET = r'(?:[}])'
OPTIONAL_BLANK = r'(?:[ \t\r\n]?)'
OPTIONAL_BLANKS = r'(?:' + OPTIONAL_BLANK + r'+)'
OR = r'|'

TUPLE = \
    r'(?:' \
    + r'(?:' \
    + r'(?:' \
    + OPTIONAL_BLANKS + FLOAT + OPTIONAL_BLANKS + COMMA \
    + r')*' \
    + OPTIONAL_BLANKS + FLOAT + OPTIONAL_BLANKS \
    + r')' \
    + OR \
    + r'(?:' \
    + r'(?:' \
    + OPTIONAL_BLANKS + LITERAL + OPTIONAL_BLANKS + COMMA \
    + r')*' \
    + OPTIONAL_BLANKS + LITERAL + OPTIONAL_BLANKS \
    + r')' \
    + r')'

STATEMENT = \
    r'^' \
    + LEFT_PARENTHESIS + r'(' + TUPLE + r')?' + RIGHT_PARENTHESIS + r'$' \
    + OR \
    + r'^' + LEFT_BRACKET + r'(' + TUPLE + r')?' + RIGHT_BRACKET + r'$' \
    + OR \
    + r'^(' + TUPLE + r')?$'


def match_array(string_input, type_str):
    """
    matching input string to a tuple of elements in `type_str` type

    :param string_input:
    :param type_str:
    :return:
    """
    regex = re.compile(STATEMENT)
    matched_str = regex.match(string_input)
    if matched_str:
        filtered_groups = [matched for matched in matched_str.groups()
                           if matched is not None]
        if not filtered_groups:
            return ()
        try:
            values = [v.strip() for v in filtered_groups[0].split(',')]
        except IndexError:
            raise ValueError(
                "unrecognised input string {}".format(string_input))
        if type_str == 'int':
            return tuple(int(val) for val in values)
        if type_str == 'float':
            return tuple(float(val) for val in values)
        if type_str == 'str':
            return tuple(values)
        raise ValueError("unknown array type_str {}".format(string_input))
    raise ValueError("invalid parameter {}".format(string_input))
