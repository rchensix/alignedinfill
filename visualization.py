# Ruiqi Chen
# March 11, 2020

import matplotlib
import numpy as np
from typing import Callable

'''
Tools for visualizing data, plotting etc.
'''

def plot_streamline(streamline: np.ndarray, ax: matplotlib.axes.Axes):
    # for now, only plot 2D
    x = streamline[:, 0]
    y = streamline[:, 1]
    ax.plot(x, y, '-')

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