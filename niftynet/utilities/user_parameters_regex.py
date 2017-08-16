INT_PATTERN = r'([-+]?\d+)'
FLOAT_PATTERN = r'([-+]?\d*.\d+|\d+)'
LITERAL_PATTERN = r'([a-zA-Z0-9]+)'

COMMA_PATTERN = r'(?:[,])'

LEFT_PARENTHESIS_PATTERN = r'(?:[(])'
RIGHT_PARENTHESIS_PATTERN = r'(?:[(])'
LEFT_BRACKET_PATTERN = r'(?:[{])'
RIGHT_BRACKET_PATTERN = r'(?:[}])'

TUPLE_PATTERN =                                                          \
    r'(?:(?:' + FLOAT_PATTERN + COMMA_PATTERN + r')?' + FLOAT_PATTERN +     \
    r')|(?:' +                                                               \
    r'(?:' + INT_PATTERN + COMMA_PATTERN + r')?' + INT_PATTERN +         \
    r')|(?:' +                                                               \
    r'(?:' + LITERAL_PATTERN + COMMA_PATTERN + r')?' + LITERAL_PATTERN + \
    r')'
    #r'|' +                                                               \
STATEMENT_PATTERN =                                                        \
    LEFT_PARENTHESIS_PATTERN + TUPLE_PATTERN + RIGHT_PARENTHESIS_PATTERN + \
    r'|' +                                                                 \
    LEFT_BRACKET_PATTERN + TUPLE_PATTERN + RIGHT_BRACKET_PATTERN +         \
    r'|' +                                                                 \
    TUPLE_PATTERN

import re
string_to_match = ['()', '(32, 32)', '(a, s)', '(32.0,32.0)', '32.0', ')']

var = re.compile(STATEMENT_PATTERN)
for ex in string_to_match:
    if var.match(ex):
        print var.match(ex).groups()
