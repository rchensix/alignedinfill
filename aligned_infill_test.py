# Ruiqi Chen
# March 13, 2020

'''
This module tests generating aligned infills
'''

import os
import sys
import unittest

import numpy as np

directory = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\Code\alignedinfill'
sys.path.insert(1, directory)
from aligned_infill import GridSpec, AlignedInfillGenerator
from analytical_fields import cantilever_vertical_load_stress, plate_with_hole_tension_stress, simply_supported_uniform_load_stress
from elasticity import constants_to_compliance_matrix, rotate_str_vector_z, strain_energy_density, str_vector_to_matrix
import numerical_fields
from optimization import brute_force_1d
from visualization import plot_field

class TestAlignedInfill(unittest.TestCase):
    def test_aligned_infill(self):
        gridspec = GridSpec(-3, 3, 41, -3, 3, 41)
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
                    result = np.array([u*np.cos(theta), u*np.sin(theta), 0]) # /u
                else:
                    result[i] = np.array([u*np.cos(theta), u*np.sin(theta), 0]) # /u
            return result
        magic_pt = np.array([-1.04707, 0.48628, 0])
        stress = plate_with_hole_tension_stress(magic_pt)
        step_size = 0.02
        separation = 0.25
        generator = AlignedInfillGenerator(gridspec, square_boundary_with_hole, alignment_field, separation, step_size, 1000)
        seeds = np.array([[0, 1, 0], 
                        [0, -1, 0],
                        [0, 1.5, 0],
                        [0, 2, 0],
                        [0, 2.5, 0],
                        [-2.9, 0.25, 0],
                        [-2.9, 0.75, 0],
                        [-2.9, 1.25, 0],
                        [-1, 0, 0]
                        ])
        generator.generate()
        generator.plot()

    def test_aligned_infill_cantilever(self):
        gridspec = GridSpec(0, 24, 81, -3, 3, 20)
        def rectangular_boundary(point: np.ndarray) -> bool:
            x = point[0]
            y = point[1]
            return x >= 0 and x <= 24 and y >= -3 and y <= 3
        def alignment_field(points: np.ndarray) -> np.ndarray:
            if points.shape == (3,): 
                pts = points.copy().reshape((1, -1))
            else:
                pts = points
            # align with principal stress direction
            stress = cantilever_vertical_load_stress(pts)
            eps = 1e-3  # prevent division by zero
            theta = 0.5*np.arctan2(2*stress[:, 3], stress[:, 0] - stress[:, 1] + eps)
            if points.shape == (3,):
                result = np.linalg.norm(stress)*np.array([np.cos(theta[0]), np.sin(theta[0]), 0])
            else:
                result = np.zeros(points.shape)
                result[:, 0] = np.linalg.norm(stress)*np.cos(theta)
                result[:, 1] = np.linalg.norm(stress)*np.sin(theta)
            return result
        step_size = 0.02
        separation = 0.25
        generator = AlignedInfillGenerator(gridspec, rectangular_boundary, alignment_field, separation, step_size, 1000)
        generator.generate()
        generator.plot()

    def test_aligned_infill_simply_supported_uniform(self):
        gridspec = GridSpec(-12, 12, 81, -3, 3, 20)
        def rectangular_boundary(point: np.ndarray) -> bool:
            x = point[0]
            y = point[1]
            return x >= -12 and x <= 12 and y >= -3 and y <= 3
        def alignment_field(points: np.ndarray) -> np.ndarray:
            if points.shape == (3,): 
                pts = points.copy().reshape((1, -1))
            else:
                pts = points
            # align with principal stress direction
            stress = simply_supported_uniform_load_stress(pts)
            eps = 1e-3  # prevent division by zero
            theta = 0.5*np.arctan2(2*stress[:, 3], stress[:, 0] - stress[:, 1] + eps)
            if points.shape == (3,):
                result = np.linalg.norm(stress)*np.array([np.cos(theta[0]), np.sin(theta[0]), 0])
            else:
                result = np.zeros(points.shape)
                result[:, 0] = np.linalg.norm(stress)*np.cos(theta)
                result[:, 1] = np.linalg.norm(stress)*np.sin(theta)
            return result
        step_size = 0.02
        separation = 0.25
        generator = AlignedInfillGenerator(gridspec, rectangular_boundary, alignment_field, separation, step_size, 1000)
        generator.generate()
        generator.plot()

    def test_aligned_infill_hole_in_tensile_field_principal_stress(self):
        length = 7
        width = 5
        num_grid_pts_x = 40
        num_grid_pts_y = 56
        gridspec = GridSpec(-length/2, length/2, num_grid_pts_x, -width/2, width/2, num_grid_pts_y)
        def square_boundary_with_hole(point: np.ndarray) -> bool:
            x = point[0]
            y = point[1]
            return x >= -length/2 and x <= length/2 and y >= -width/2 and y <= width/2 and np.sqrt(x**2 + y**2) >= 1
        def alignment_field(points: np.ndarray) -> np.ndarray:
            if points.shape == (3,): 
                pts = points.copy().reshape((1, -1))
            else:
                pts = points
            # align with principal stress direction
            stress = plate_with_hole_tension_stress(pts)
            eps = 1e-3  # prevent division by zero
            theta = 0.5*np.arctan2(2*stress[:, 3], stress[:, 0] - stress[:, 1] + eps)
            if points.shape == (3,):
                result = np.linalg.norm(stress)*np.array([np.cos(theta[0]), np.sin(theta[0]), 0])
            else:
                result = np.zeros(points.shape)
                result[:, 0] = np.linalg.norm(stress)*np.cos(theta)
                result[:, 1] = np.linalg.norm(stress)*np.sin(theta)
            return result
        step_size = 0.02
        separation = 0.25
        generator = AlignedInfillGenerator(gridspec, square_boundary_with_hole, alignment_field, separation, step_size, 1000)
        seeds = np.array([
            [0, 1.0, 0],
            [0, -1.0, 0]
        ])
        generator.generate(seeds)
        for i, streamline in enumerate(generator.streamlines):
            out_path = os.path.join(directory, 'hole_in_plate', 'streamline{}.npy'.format(i))
            np.save(out_path, streamline)
        generator.plot()

    def test_aligned_infill_fem(self):
        # Read fem result of 3 point bending
        result_path = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\FEM\three_point_bend_stress.vtu'
        sxx, syy, sxy = numerical_fields.rbf_stress_interpolator(result_path)
        gridspec = GridSpec(-60e-3, 60e-3, 121, -12.5e-3, 12.5e-3, 26)
        def rectangular_boundary(point: np.ndarray) -> bool:
            x = point[0]
            y = point[1]
            return x >= -60e-3 and x <= 60e-3 and y >= -12.5e-3 and y <= 12.5e-3
        def alignment_field(points: np.ndarray) -> np.ndarray:
            if points.shape == (3,): 
                pts = points.copy().reshape((1, -1))
            else:
                pts = points
            # align with principal stress direction
            stress = np.zeros((pts.shape[0], 6))
            stress[:, 0] = sxx(pts[:, 0], pts[:, 1], pts[:, 2])
            stress[:, 1] = syy(pts[:, 0], pts[:, 1], pts[:, 2])
            stress[:, 3] = sxy(pts[:, 0], pts[:, 1], pts[:, 2])
            eps = 1e-3  # prevent division by zero
            theta = 0.5*np.arctan2(2*stress[:, 3], stress[:, 0] - stress[:, 1] + eps)
            if points.shape == (3,):
                result = np.linalg.norm(stress)*np.array([np.cos(theta[0]), np.sin(theta[0]), 0])
            else:
                result = np.zeros(points.shape)
                result[:, 0] = np.linalg.norm(stress)*np.cos(theta)
                result[:, 1] = np.linalg.norm(stress)*np.sin(theta)
            return result
        step_size = 0.02
        separation = 0.25
        generator = AlignedInfillGenerator(gridspec, rectangular_boundary, alignment_field, separation, step_size, 1000)
        seeds = np.array([
            [1e-6, -12.49e-3, 0],
        ])
        generator.generate(seeds)
        generator.plot()

if __name__ == '__main__':
    unittest.main()