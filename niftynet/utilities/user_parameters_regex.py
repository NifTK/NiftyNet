# regular expressions to match tuples from user inputs
# kindly provided by
# Luis Carlos Garcia Peraza Herrera <luis.herrera.14@ucl.ac.uk>

import re

INT = r'(?:[-+]?\d+)'
FLOAT = r'(?:[-+]?\d*\.\d+|' + INT + r')'
LITERAL = r'(?:[a-zA-Z0-9]+)'
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
    regex = re.compile(STATEMENT)
    matched_str = regex.match(string_input)
    if matched_str:
        filtered_groups = filter(None, matched_str.groups())
        if not filtered_groups:
            return ()
        values = [v.strip() for v in list(filtered_groups)[0].split(',')]
        if type_str == 'int':
            return tuple(map(int, values))
        if type_str == 'float':
            return tuple(map(float, values))
        if type_str == 'str':
            return tuple(values)
        else:
            raise ValueError("unknown array type {}".format(string_input))
    else:
        raise ValueError("invalid parameter {}".format(string_input))


if __name__ == "__main__":
# TODO add expected outputs, move to unit tests folder
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
    regex = re.compile(STATEMENT)

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
