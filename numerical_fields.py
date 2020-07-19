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

def main():
    pass

if __name__ == '__main__':
    main()