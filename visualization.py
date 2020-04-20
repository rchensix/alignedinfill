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

def plot_polygon(polygon: np.ndarray, ax: matplotlib.axes.Axes):
    # for now, only plot 2D
    num_points = polygon.shape[0]
    x = np.zeros(num_points + 1)  # add 1 to close the polygon
    y = np.zeros(num_points + 1)  # add 1 to close the polygon
    x[0:num_points] = polygon[:, 0]
    y[0:num_points] = polygon[:, 1]
    x[-1] = x[0]
    y[-1] = y[0]
    ax.plot(x, y)