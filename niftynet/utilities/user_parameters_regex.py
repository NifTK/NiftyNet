# regular expressions
# by 	Luis Carlos Garcia Peraza Herrera <luis.herrera.14@ucl.ac.uk>
#
INT_PATTERN = r'(?:[-+]?\d+)'
FLOAT_PATTERN = r'(?:[-+]?\d*\.\d+|' + INT_PATTERN + r')'
LITERAL_PATTERN = r'(?:[a-zA-Z0-9]+)'
COMMA_PATTERN = r'(?:[,])'
LEFT_PARENTHESIS_PATTERN = r'(?:\()'
RIGHT_PARENTHESIS_PATTERN = r'(?:\))'
LEFT_BRACKET_PATTERN = r'(?:[{])'
RIGHT_BRACKET_PATTERN = r'(?:[}])'
OPTIONAL_BLANK_PATTERN = r'(?:[ \t\r\n]?)'
OPTIONAL_BLANKS_PATTERN = r'(?:' + OPTIONAL_BLANK_PATTERN + r'+)'
OR_PATTERN = r'|'

TUPLE_PATTERN = \
    r'(?:' \
    r'(?:' \
    + r'(?:' \
    + OPTIONAL_BLANKS_PATTERN + FLOAT_PATTERN + OPTIONAL_BLANKS_PATTERN + COMMA_PATTERN \
    + r')*' \
    + OPTIONAL_BLANKS_PATTERN + FLOAT_PATTERN + OPTIONAL_BLANKS_PATTERN \
    + r')' \
    + OR_PATTERN \
    + r'(?:' \
    + r'(?:' \
    + OPTIONAL_BLANKS_PATTERN + LITERAL_PATTERN + OPTIONAL_BLANKS_PATTERN + COMMA_PATTERN \
    + r')*' \
    + OPTIONAL_BLANKS_PATTERN + LITERAL_PATTERN + OPTIONAL_BLANKS_PATTERN \
    + r')' \
    + r')'

STATEMENT_PATTERN = \
    r'^' + LEFT_PARENTHESIS_PATTERN + r'(' + TUPLE_PATTERN + r')?' + RIGHT_PARENTHESIS_PATTERN + r'$' \
    + OR_PATTERN \
    + r'^' + LEFT_BRACKET_PATTERN + r'(' + TUPLE_PATTERN + r')?' + RIGHT_BRACKET_PATTERN + r'$' \
    + OR_PATTERN \
    + r'^(' + TUPLE_PATTERN + r')?$'

import re

# strings_to_match = ['(32, 32)', '(a, s)', '(32.0,32.0)', '32.0']
strings_to_match = [
    '2.0, ( 6.0, 9.0',
    '{32.0   , 32.0}',
    '{   32.0, 32.0}',
    'a, c, b, f, d, e',
    '(), ()',
    '{), (}',
    '(),',
    '()',
    '{}',
    '32, (32),',
    '32, (),',
    '32',
    '({)',
    '(()',
    '',
    '32, 32',
    '32,',
    '-32',
    '-32.0, a',
    '-a, 32.0',
    '(-32,a)',
    '-32.0',
    '(-a)',
    '(-32.0, 10.0, 2.99987, 5.6, 3.5, 5.6)',
]
regex = re.compile(STATEMENT_PATTERN)

for strval in strings_to_match:
    matched_str = regex.match(strval)
    if matched_str:
        filtered_groups = filter(None, matched_str.groups())
        if filtered_groups:
            values = [v.strip() for v in filtered_groups[0].split(',')]
        else:
            values = []
        print('String:', strval, 'Matched!', values)
    # print('String:', strval, 'Matched!')
    else:
        print('String:', strval, 'Error!')
