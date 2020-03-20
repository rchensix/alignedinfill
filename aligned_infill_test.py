# Ruiqi Chen
# March 13, 2020

'''
This module tests generating aligned infills
'''

import numpy as np
import sys

directory = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\Code\alignedinfill'
sys.path.insert(1, directory)
from aligned_infill import GridSpec, AlignedInfillGenerator
from analytical_fields import plate_with_hole_tension_stress
from elasticity import constants_to_compliance_matrix, rotate_str_vector_z, strain_energy_density
from optimization import brute_force_1d
from visualization import plot_field

def aligned_infill_test():
    gridspec = GridSpec(-3, 3, 101, -3, 3, 101)
    def square_boundary_with_hole(point: np.ndarray) -> bool:
        x = point[0]
        y = point[1]
        return x >= -3 and x <= 3 and y >= -3 and y <= 3 and np.sqrt(x**2 + y**2) >= 1
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
    def alignment_field(points: np.ndarray) -> np.ndarray:
        if points.shape == (3,): 
            pts = points.copy().reshape((1, -1))
            result = np.zeros(3)
        else:
            pts = points
            result = np.zeros_like(pts)
        compliance = constants_to_compliance_matrix(3300, 2400, 1900, 0.33, 0.30, 0.33, 1000, 900, 850)
        # loop over points
        # find optimal orientation at every point
        theta_domain = np.linspace(-np.pi/2, np.pi/2, 91)
        for i in range(pts.shape[0]):
            pt = pts[i]
            if not square_boundary_with_hole(pt): 
                if points.shape == (3,): 
                    return np.zeros(3)
                else:
                    result[i, :] = np.zeros(3)
                continue
            stress = plate_with_hole_tension_stress(pt)
            # stress = uniaxial_tension_stress(np.array([x, y]))
            func = objective_functor(compliance, stress)
            u, theta = brute_force_1d(func, theta_domain)
            if pts.shape[0] == 1:
                result = np.array([u*np.cos(theta), u*np.sin(theta), 0])
            else:
                result[i] = np.array([u*np.cos(theta), u*np.sin(theta), 0])
        return result
    step_size = 0.02
    separation = 0.25
    generator = AlignedInfillGenerator(gridspec, square_boundary_with_hole, alignment_field, separation, step_size, 1000)
    generator.generate()
    generator.plot(debug_mode_on=True)

def main():
    aligned_infill_test()

if __name__ == '__main__':
    main()