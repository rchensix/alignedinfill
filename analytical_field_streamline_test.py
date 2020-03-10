# Ruiqi Chen
# March 9, 2020

'''
This module tests generating streamlines from an analytical vector field.
'''

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from typing import Callable, List

directory = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\Code\alignedinfill'
sys.path.insert(1, directory)
from path_planner import generate_streamline

# Takes a array input (n x 2) or (n x 3) representing position [x, y, z] in every row and returns array of same size as input
# with every row representing a vector
# See Barber "Elasticity" pp. 128 for solution.
def plate_with_hole_shear_displacement(point: np.ndarray) -> np.ndarray:
    a = 1  # hole radius
    s = 1  # tensile stress in xx direction
    e = 100  # young's modulus of material
    nu = 0.3  # poisson ratio of material
    eps = 1e-6  # prevents overflow at origin

    if (point.shape == (2,)):
        x, y = point
    elif (point.shape == (3,)):
        x, y, z = point
    elif (len(point.shape) == 2 and point.shape[1] == 2):
        x = point[:, 0]
        y = point[:, 1]
    elif (len(point.shape) == 2 and point.shape[1] == 3):
        x = point[:, 0]
        y = point[:, 1]
        z = point[:, 2]
    else:
        raise Exception('Bad input shape.')
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    ur = s/e*((1 + nu)*r + 4*a**2/(r + eps) - (1 + nu)*a**4/(r**3 + eps))*np.sin(2*theta)
    ut = s/e*((1 + nu)*r + 2*(1 - nu)*a**2/(r + eps) + (1 + nu)*a**4/(r**3 + eps))*np.cos(2*theta)
    u = ur*np.cos(theta) - ut*np.sin(theta)
    v = ur*np.sin(theta) + ut*np.cos(theta)

    if (point.shape == (2,)):
        return np.array([u, v])
    elif (point.shape == (3,)):
        return np.array([u, v, 0])
    else:
        result = np.zeros(point.shape)
        result[:, 0] = u
        result[:, 1] = v
        return result

# field with zero vectors in center band, otherwise equal to point itself
def zero_field(point: np.ndarray) -> np.ndarray:
    if (point.shape == (2,)):
        x, y = point
        u = 0 if x > -2 and x < 2 and y > -2 and y < 2 else x
        v = 0 if x > -2 and x < 2 and y > -2 and y < 2 else y
    elif (point.shape == (3,)):
        x, y, z = point
        u = 0 if x > -2 and x < 2 and y > -2 and y < 2 else x
        v = 0 if x > -2 and x < 2 and y > -2 and y < 2 else y
    elif (len(point.shape) == 2 and point.shape[1] == 2):
        x = point[:, 0]
        y = point[:, 1]
        u = x
        v = y
        u[np.logical_and(np.logical_and(np.logical_and(x > -2, x < 2), y > -2), y < 2)] = 0
        v[np.logical_and(np.logical_and(np.logical_and(x > -2, x < 2), y > -2), y < 2)] = 0
    elif (len(point.shape) == 2 and point.shape[1] == 3):
        x = point[:, 0]
        y = point[:, 1]
        z = point[:, 2]
        u = x
        v = y
        u[np.logical_and(np.logical_and(np.logical_and(x > -2, x < 2), y > -2), y < 2)] = 0
        v[np.logical_and(np.logical_and(np.logical_and(x > -2, x < 2), y > -2), y < 2)] = 0
    else:
        raise Exception('Bad input shape.')

    if (point.shape == (2,)):
        return np.array([u, v])
    elif (point.shape == (3,)):
        return np.array([u, v, 0])
    else:
        result = np.zeros(point.shape)
        result[:, 0] = u
        result[:, 1] = v
        return result

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

def plot_streamline(streamline: np.ndarray, ax: matplotlib.axes.Axes):
    # for now, only plot 2D
    x = streamline[:, 0]
    y = streamline[:, 1]
    ax.plot(x, y)

def plot_field(field: Callable[[np.ndarray], np.ndarray], points: np.ndarray, ax: matplotlib.axes.Axes):
    # for now, only plot 2D
    vec = field(points)
    if len(points.shape) == 1:
        x = points[0]
        y = points[1]
        u = vec[0]
        v = vec[1]
    else:
        x = points[:, 0]
        y = points[:, 1]
        u = vec[:, 0]
        v = vec[:, 1]
    ax.quiver(x, y, u, v)

def main():
    # field = plate_with_hole_shear_displacement
    field = zero_field
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
    main()