# Ruiqi Chen
# March 11, 2020

'''
This module tests preferred fiber orientation on analytical stress fields.
'''

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import Callable

directory = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\Code\alignedinfill'
sys.path.insert(1, directory)
from analytical_fields import plate_with_hole_tension_stress, uniaxial_tension_stress
from elasticity import constants_to_compliance_matrix, rotate_str_vector_z, strain_energy_density
from optimization import brute_force_1d

# Defines a strain energy density based on material properties and stress
def objective_functor(compliance_matrix: np.ndarray, stress_vector: np.ndarray) -> np.ndarray:
    def f(angle_rad: np.ndarray) -> np.ndarray:
        result = np.zeros_like(angle_rad)
        for i in range(result.size):
            stress_rotated = rotate_str_vector_z(stress_vector, -angle_rad[i])  # note the negative sign
            u = strain_energy_density(compliance_matrix, stress_rotated)
            result[i] = u
        return result
    return f

def fiber_orientation_optimizer_test():
    xvec = np.linspace(-3, 3, 20)
    yvec = np.linspace(-3, 3, 20)
    xlist = []
    ylist = []
    ulist = []
    vlist = []
    compliance = constants_to_compliance_matrix(3300, 2400, 1900, 0.33, 0.30, 0.33, 1000, 900, 850)
    # loop over points in a grid
    # find optimal orientation at every point
    theta_domain = np.linspace(-np.pi/2, np.pi/2, 181)
    for x in np.nditer(xvec):
        for y in np.nditer(yvec):
            if np.sqrt(x**2 + y**2) < 1: continue
            stress = plate_with_hole_tension_stress(np.array([x, y]))
            # stress = uniaxial_tension_stress(np.array([x, y]))
            func = objective_functor(compliance, stress)
            u, theta = brute_force_1d(func, theta_domain)
            xlist.append(x)
            ylist.append(y)
            ulist.append(u*np.cos(theta))
            vlist.append(u*np.sin(theta))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.quiver(xlist, ylist, ulist, vlist)
    plt.show()

def main():
    fiber_orientation_optimizer_test()

if __name__ == '__main__':
    main()