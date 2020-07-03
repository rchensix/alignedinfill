# Ruiqi Chen
# March 9, 2020

'''
This module tests generating streamlines from an analytical vector field.
'''

import sys
from typing import Callable, List
import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

directory = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\Code\alignedinfill'
sys.path.insert(1, directory)
from analytical_fields import plate_with_hole_shear_displacement, center_zero_field
from path_planner import generate_streamline
from visualization import plot_field, plot_streamline

def bounding_box_stop(point: np.ndarray) -> bool:
    if point.size == 2:
        x, y = point
        z = 0
    elif point.size == 3:
        x, y, z = point
    else:
        raise Exception('point dimension must be 2 or 3')
    xmin, ymin, zmin = (-10, -10, -10)
    xmax, ymax, zmax = (10, 10, 10)
    return x <= xmin or x >= xmax or y <= ymin or y >= ymax or z <= zmin or z >= zmax

class TestAnalyticalFieldStreamline(unittest.TestCase):
    def test_plate_with_hole_shear_displacement(self):
        field = plate_with_hole_shear_displacement
        # field = center_zero_field
        stop_condition = bounding_box_stop
        start = np.array([9, 8.7, 0], dtype=np.float64)
        direction = False
        streamline = generate_streamline(field, start, direction, stop_condition, step_size=0.1, max_steps=1000)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        x, y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
        points = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        plot_streamline(streamline, ax)
        plot_field(field, points, ax)
        plt.show()

if __name__ == '__main__':
    unittest.main()