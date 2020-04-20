# Ruiqi Chen
# March 30, 2020

'''
Tools for reading FEA data and performing interpolation.
'''

import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.interpolate
from typing import Tuple

def rbf_stress_interpolator(path: str, function: str='gaussian') -> Tuple:
    mesh = meshio.read(path)
    pts = mesh.points
    data = mesh.point_data
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    sxx = data['Stress_tensor,_x_component']
    syy = data['Stress_tensor,_y_component']
    sxy = data['Stress_tensor,_xy_component']
    return scipy.interpolate.Rbf(x, y, z, sxx, function=function), \
           scipy.interpolate.Rbf(x, y, z, syy, function=function), \
           scipy.interpolate.Rbf(x, y, z, sxy, function=function) 

def meshio_test():
    sxx, syy, sxy = rbf_stress_interpolator(r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\FEM\three_point_bend_stress.vtu')
    x, y = np.meshgrid(np.linspace(-60e-3, 60e-3, 121), np.linspace(-12.5e-3, 12.5e-3, 26))
    sxx_grid = sxx(x, y, np.zeros_like(x))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.imshow(sxx_grid)
    plt.show()

def main():
    meshio_test()

if __name__ == '__main__':
    main()