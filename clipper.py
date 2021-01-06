# Ruiqi Chen
# March 27, 2020

'''
Tools for polygon testing and insetting polygons for GCode generation.
'''

# import sys
from typing import List, Union
import warnings

from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import pyclipper

# directory = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\Code\alignedinfill'
# sys.path.insert(1, directory)

# polygon is (N, 2) or (N, 3) ndarray where N is number of vertices. polygon is assumed to be closed. 
#    Any 3D data is discarded.
# point is (2,), (3,), (M, 2), or (M, 3) ndarray. Any 3D data is ignored.
def point_in_polygon(polygon: np.ndarray, point: np.ndarray) -> Union[bool, np.ndarray]:
    if polygon.shape[1] == 2:
        path = Path(polygon, closed=True)
    elif polygon.shape[1] == 3:
        warnings.warn('3D polygon not supported. Flattening to 2D.')
        path = Path(polygon[:, :2], closed=True)
    else:
        raise Exception('polygon shape must be either (N, 2) or (N, 3)')
    if point.shape == (2,):
        return path.contains_point(point)
    elif point.shape == (3,):
        warnings.warn('point contains 3D data; ignoring z component.')
        return path.contains_point(point[:2])
    elif point.shape[1] == 2:
        return path.contains_points(point)
    elif point.shape[1] == 3:
        warnings.warn('point contains 3D data; ignoring z component.')
        return path.contains_points(point[:, :2])
    else:
        raise Exception('point shape must be either (2,), (3,), (M, 2), or (M, 3)')

def inset_polygon(polygon: np.ndarray, inset_amount: float) -> List:
    assert inset_amount > 0, "inset_amount must be greater than 0 but is currently {}".format(inset_amount)
    if polygon.shape[1] == 3:
        warnings.warn('polygon contains 3D data; ignoring z component.')
        polygon = np.copy(polygon)[:, :2]
    assert polygon.shape[1] == 2, "polygon must be 2D and have shape (n, 2) where n is number of vertices"
    polygon_scaled = pyclipper.scale_to_clipper(polygon)
    inset_scaled = pyclipper.scale_to_clipper(-inset_amount)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(polygon_scaled, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solution_scaled = pco.Execute(inset_scaled)
    solution = pyclipper.scale_from_clipper(solution_scaled)
    return solution

def main():
    pass

if __name__ == '__main__':
    main()
