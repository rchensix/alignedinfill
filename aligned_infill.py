# Ruiqi Chen
# March 12, 2020

'''
This module contains the main functions and classes for generating aligned infills.
'''

import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import Callable, List, Tuple

directory = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\Code\alignedinfill'
sys.path.insert(1, directory)
from path_planner import generate_streamline
from visualization import plot_field, plot_streamline

class GridSpec:
    def __init__(self, xmin: float, xmax: float, num_pts_x: int, ymin: float, ymax: float, num_pts_y: int, zmin: float=0, zmax: float=0, num_pts_z: int=0):
        self.xmin = xmin
        self.xmax = xmax
        self.num_pts_x = num_pts_x
        self.ymin = ymin
        self.ymax = ymax
        self.num_pts_y = num_pts_y
        self.zmin = zmin
        self.zmax = zmax
        self.num_pts_z = num_pts_z

# boundary: returns true if in bounds
class AlignedInfillGenerator:
    def __init__(self, grid_spec: GridSpec, in_bounds: Callable[[np.ndarray], bool], alignment_field: Callable[[np.ndarray], np.ndarray], separation: float, step_size: float=0.01, max_steps: int=10000):
        self.streamlines = []
        self.grid_spec = grid_spec
        self.in_bounds = in_bounds
        self.alignment_field = alignment_field
        self.separation = separation
        self.step_size = step_size
        self.max_steps = max_steps
        assert self.grid_spec.num_pts_x > 1, 'grid_spec must have at least 2 points in x'
        assert self.grid_spec.num_pts_y > 1, 'grid_spec must have at least 2 points in y'
        self.spacing_x = (self.grid_spec.xmax - self.grid_spec.xmin)/(self.grid_spec.num_pts_x - 1)
        self.spacing_y = (self.grid_spec.ymax - self.grid_spec.ymin)/(self.grid_spec.num_pts_y - 1)

    def index_to_world(self, index: int) -> np.ndarray:
        i, j = self.index_to_tuple(index)
        return np.array([self.grid_spec.xmin + i*self.spacing_x, self.grid_spec.ymin + j*self.spacing_y, 0])

    def world_to_index(self, point: np.ndarray) -> int:
        assert point.shape == (2,) or point.shape == (3,), 'point shape must be either (2,) or (3,)'
        x = point[0]
        y = point[1]
        i = int(np.round((x - self.grid_spec.xmin)/self.spacing_x))  
        j = int(np.round((y - self.grid_spec.ymin)/self.spacing_y)) 
        return self.tuple_to_index(i, j)
    
    def tuple_to_index(self, i: int, j: int) -> int:
        return i + j*self.grid_spec.num_pts_x
    
    def index_to_tuple(self, index: int) -> Tuple[int, int]:
        return int(index % self.grid_spec.num_pts_x), int(index // self.grid_spec.num_pts_x)

    def is_index_inbounds(self, index: int) -> bool:
        return index >= 0 and index < self.grid_spec.num_pts_x * self.grid_spec.num_pts_y
    
    def generate(self) -> List[np.ndarray]:
        self.streamlines.clear()
        # Create a BooleanGrid of visited spots
        self.visited = np.full(self.grid_spec.num_pts_x*self.grid_spec.num_pts_y, False, dtype=bool)
        # Create array of points to evaluate PointField
        self.pts = np.zeros((self.grid_spec.num_pts_x*self.grid_spec.num_pts_y, 3))
        for j in range(self.grid_spec.num_pts_y):
            for i in range(self.grid_spec.num_pts_x):
                index = self.tuple_to_index(i, j)
                self.pts[index] = self.index_to_world(index)
        self.vectors = self.alignment_field(self.pts)
        self.magnitude = np.linalg.norm(self.vectors, axis=1)
        sorted_indices = np.flip(np.argsort(self.magnitude))
        for i in range(sorted_indices.size):
            index = sorted_indices[i]
            if self.is_valid_start_point(index):
                line1 = generate_streamline(self.alignment_field, self.pts[index], True, self.stop_streamline_condition, self.step_size, self.max_steps)
                line2 = generate_streamline(self.alignment_field, self.pts[index], False, self.stop_streamline_condition, self.step_size, self.max_steps)
                # merge streamlines into one
                line = np.zeros((line1.shape[0] + line2.shape[0] - 1, 3))
                line[:line1.shape[0], :] = np.flip(line1, axis=0)
                line[line1.shape[0]:, :] = line2[1:]
                self.visited[index] = True
                self.mark_line_visited(line)
                self.streamlines.append(line)
    
    def stop_streamline_condition(self, point: np.ndarray) -> bool:
        # stop generating streamline if the point goes
        # -out of grid
        # -out of bounds
        # -comes too close to another streamline
        index = self.world_to_index(point)
        return not self.is_index_inbounds(index) or not self.in_bounds(point) or self.visited[index] or not self.is_separated(index)   

    # Marks all grid points in a rectangle defined by points in the line as visited
    # Rectangle is defined as follows:
    #  C ---- P1 ---- D
    #  |              |
    #  |              |        t
    #  |              |        ^
    #  |              |        |
    #  A ---- P0 ---- B  n <---|
    def mark_line_visited(self, line: np.ndarray):
        for i in range(line.shape[0] - 1):
            # Calculate points A, B, C, D from P0, P1
            p0 = line[i]
            p1 = line[i + 1]
            # Avoid numerical errors if points are too close together
            tol = 1e-12
            if (p1 - p0).dot(p1 - p0) < tol: continue
            t = (p1 - p0)/np.linalg.norm(p1 - p0)
            n = np.array([-t[1], t[0], 0])
            search_step_size_approx = np.min(np.array([self.spacing_x/2, self.spacing_y/2]))
            num_search_steps = int(self.separation*2/search_step_size_approx) + 1
            u = np.linspace(-1, 1, num_search_steps)
            for i in range(num_search_steps):
                a = p0 + n*u[i]*self.separation
                c = p1 + n*u[i]*self.separation
                ind_a = self.world_to_index(a)
                ind_c = self.world_to_index(c)
                if self.is_index_inbounds(ind_a): self.visited[ind_a] = True
                if self.is_index_inbounds(ind_c): self.visited[ind_c] = True

    def is_valid_start_point(self, index: int):
        # start point is valid if it is
        # -in bounds
        # -not visited
        # -separated from other visited spots by at least self.separation
        return self.is_index_inbounds(index) and self.in_bounds(self.pts[index]) and (~self.visited[index]) and self.is_separated(index)
    
    def is_separated(self, index: int):
        indices = self.radius_search(index, self.separation)
        for index in indices:
            if self.visited[index]: return False
        return True
    
    # returns list of indices within given radius of index
    def radius_search(self, index: int, radius: float) -> List[int]:
        indices = []
        i, j = self.index_to_tuple(index)
        search_radius_x = int(radius/self.spacing_x)
        search_radius_y = int(radius/self.spacing_y)
        for di in range(-search_radius_x, search_radius_x + 1):
            for dj in range(-search_radius_y, search_radius_y + 1):
                index_new = self.tuple_to_index(i + di, j + dj)
                if self.is_index_inbounds(index_new):
                    pt1 = self.pts[index_new]
                    pt0 = self.pts[index]
                    if np.linalg.norm(pt1 - pt0) <= radius: indices.append(index_new)
        return indices

    def plot(self, infill_on: bool=True, boundary_on: bool=True, alignment_field_on: bool=True, debug_mode_on: bool=False):
        if (not infill_on) and (not boundary_on) and (not alignment_field_on): return
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        if infill_on:
            for streamline in self.streamlines:
                plot_streamline(streamline, ax)
        if boundary_on:
            # TODO: plot boundary
            pass
        if alignment_field_on:
            # TODO: plot alignment field
            pass
        if debug_mode_on:
            # Visited squares plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.imshow(self.visited.reshape((self.grid_spec.num_pts_x, -1)))
            # Magnitude plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.imshow(self.magnitude.reshape((self.grid_spec.num_pts_x, -1)))
            # Separate alignment field plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            plot_field(self.alignment_field, self.pts, ax)
        plt.show()
