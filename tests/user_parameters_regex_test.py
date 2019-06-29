from __future__ import unicode_literals

import re

import tensorflow as tf

from niftynet.utilities.user_parameters_regex import STATEMENT
from tests.niftynet_testcase import NiftyNetTestCase

class UserParameterRegexTest(NiftyNetTestCase):
    def run_match(self, string_to_match, expected_output):
        regex = re.compile(STATEMENT)
        matched_str = regex.match(string_to_match)
        if matched_str:
            filtered_groups = list(filter(None, matched_str.groups()))
            if filtered_groups:
                values = [v.strip() for v in filtered_groups[0].split(',')]
                self.assertEqual(values, expected_output)
        else:
            self.assertEqual(expected_output, False)

    def test_cases(self):
        self.run_match('c:\program files', [u'c:\\program files'])
        self.run_match('2.0, ( 6.0, 9.0', False)
        self.run_match('{   32.0, 32.0}', [u'32.0', u'32.0'])
        self.run_match('a, c, b, f, d, e', [u'a', u'c', u'b', u'f', u'd', u'e'])
        self.run_match('(), ()', False)
        self.run_match('{), (}', False)
        self.run_match('(),', False)
        self.run_match('()', False)
        self.run_match('{}', False)
        self.run_match('32, (32),', False)
        self.run_match('32, (),', False)
        self.run_match('32', [u'32'])
        self.run_match('({)', False)
        self.run_match('(()', False)
        self.run_match('', False)
        self.run_match('32, 32', [u'32', u'32'])
        self.run_match('32,', False)
        self.run_match('-32', [u'-32'])
        self.run_match('-32.0, a', [u'-32.0', 'a'])
        self.run_match('-a, 32.0', [u'-a', '32.0'])
        self.run_match('(-32,a)', [u'-32', u'a'])
        self.run_match('-32.0', [u'-32.0'])
        self.run_match('(-a)', [u'-a'])
        self.run_match('(-32.0, 10.0, 2.99987, 5.6, 3.5, 5.6)',
                       [u'-32.0', u'10.0', u'2.99987', u'5.6', u'3.5', u'5.6'])


if __name__ == "__main__":
    tf.test.main()
