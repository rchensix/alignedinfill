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
        x, y = np.meshgrid(np.linspace(-60e-3, 60e-3, 61), np.linspace(-12.5e-3, 12.5e-3, 13))
        sxx_grid = sxx(x, y, np.zeros_like(x))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.contourf(x, y, sxx_grid)
        plt.show()

    def test_meshio_vonmises(self):
        result_path = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\FEM\three_point_bend_stress.vtu'
        sxx, syy, sxy = numerical_fields.rbf_stress_interpolator(result_path)
        x, y = np.meshgrid(np.linspace(-60e-3, 60e-3, 61), np.linspace(-12.5e-3, 12.5e-3, 13))
        sxx_grid = sxx(x, y, np.zeros_like(x))
        syy_grid = syy(x, y, np.zeros_like(x))
        sxy_grid = sxy(x, y, np.zeros_like(x))
        vonmises_grid = np.sqrt(sxx_grid**2 - sxx_grid*syy_grid + syy_grid**2 + 3*sxy_grid**2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        contour = ax.contourf(x, y, vonmises_grid, cmap='jet')
        plt.colorbar(contour)
        plt.show()

if __name__ == '__main__':
    unittest.main()