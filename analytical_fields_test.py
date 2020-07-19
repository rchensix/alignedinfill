# Ruiqi Chen
# July 8, 2020

'''
This module tests analytical fields
'''

import unittest

import matplotlib.pyplot as plt
import numpy as np

import analytical_fields

class TestAnalyticalFields(unittest.TestCase):
    def test_circular_field(self):
        # Check circular tensile field
        s = analytical_fields.plate_with_hole_tension_stress(np.array([0, 1, 0]))
        s_golden = 3000
        match_digits = 12
        self.assertAlmostEqual(s[0], s_golden, match_digits)

if __name__ == '__main__':
    unittest.main()