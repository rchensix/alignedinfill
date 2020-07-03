# Ruiqi Chen
# March 11, 2020

'''
This module tests compliance matrix rotation and how it affects strain energy.
'''

import sys
import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

directory = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\Code\alignedinfill'
sys.path.insert(1, directory)
from analytical_fields import uniaxial_tension_stress
from elasticity import constants_to_compliance_matrix, rotate_str_vector_z, strain_energy_density

class TestRotatedMaterialStrainEnergy(unittest.TestCase):
    def test_rotated_material_strain_energy(self):
        theta = np.linspace(0, np.pi, 181)
        compliance = constants_to_compliance_matrix(3500, 2400, 1900, 0.33, 0.30, 0.33, 800, 700, 750)
        u = np.zeros_like(theta)
        for i in range(theta.size):
            stress = np.array([1, 0, 0, 0, 0, 0])
            rotated_stress = rotate_str_vector_z(stress, -theta[i])
            u[i] = strain_energy_density(compliance, rotated_stress)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(theta*180/np.pi, u)
        plt.show()

if __name__ == '__main__':
    unittest.main()