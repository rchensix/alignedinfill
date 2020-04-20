# Ruiqi Chen
# March 27, 2020

'''
Tools for insetting polygons for GCode generation.
'''

import matplotlib.pyplot as plt
import numpy as np
import pyclipper
import sys
from typing import List

directory = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\Code\alignedinfill'
sys.path.insert(1, directory)
from visualization import plot_polygon 

def inset_polygon(polygon: np.ndarray, inset_amount: float) -> List:
    assert inset_amount > 0, "inset_amount must be greater than 0 but is currently {}".format(inset_amount)
    assert polygon.shape[1] == 2, "polygon must be 2D and have shape (n, 2) where n is number of vertices"
    polygon_scaled = pyclipper.scale_to_clipper(polygon)
    inset_scaled = pyclipper.scale_to_clipper(-inset_amount)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(polygon_scaled, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solution_scaled = pco.Execute(inset_scaled)
    solution = pyclipper.scale_from_clipper(solution_scaled)
    return solution

def inset_test():
    dumbbell = np.array([
        [3, 3],
        [3, -3],
        [0.5, -3],
        [0.5, -0.5],
        [-0.5, -0.5],
        [-0.5, -3],
        [-3, -3],
        [-3, 3],
        [-0.5, 3],
        [-0.5, 0.5],
        [0.5, 0.5],
        [0.5, 3]
    ])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    plot_polygon(dumbbell, ax)
    for inset_amount in [0.5, 1, 1.5]:
        solution = inset_polygon(dumbbell, inset_amount)
        for sol in solution:
            plot_polygon(np.array(sol), ax)
    plt.show()

def main():
    inset_test()

if __name__ == '__main__':
    main()
