# Ruiqi Chen
# July 3, 2020

'''
Test FEA data reading and interpolation.
'''

import unittest

import matplotlib.pyplot as plt
import numpy as np

import numerical_fields

class TestNumericalFields(unittest.TestCase):
    def test_meshio(self):
        result_path = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\FEM\three_point_bend_stress.vtu'
        sxx, syy, sxy = numerical_fields.rbf_stress_interpolator(result_path)
        x, y = np.meshgrid(np.linspace(-60e-3, 60e-3, 121), np.linspace(-12.5e-3, 12.5e-3, 26))
        sxx_grid = sxx(x, y, np.zeros_like(x))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.imshow(sxx_grid)
        plt.show()

if __name__ == '__main__':
    unittest.main()