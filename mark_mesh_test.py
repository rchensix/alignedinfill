import os
from typing import List, Tuple
import unittest

import meshio
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

import mark_mesh

kTestDir = 'Code/alignedinfill/test_data'

def _CreateQuiverPlot(pts_directions: List[Tuple[np.ndarray, float]], out_path: str):
    x = list()
    y = list()
    u = list()
    v = list()
    for pt_direction in pts_directions:
        pt, direction = pt_direction
        x.append(pt[0])
        y.append(pt[1])
        u.append(np.cos(direction))
        v.append(np.sin(direction))
    ax = plt.figure().add_subplot(111)
    ax.quiver(x, y, u, v)
    plt.savefig(out_path)
    plt.close()

class TestMarkMesh(unittest.TestCase):
    def DISABLED_test_basic(self):
        region = np.array([[0, 0],
                           [1, 0],
                           [1, 1],
                           [0, 1]])
        inset_spacing = 0.2
        sample_spacing = 0.1
        dedup = 1e-6
        pts_directions = mark_mesh.GetRegionDirections(region, inset_spacing, sample_spacing,
                                                       dedup)
        out_path = os.path.join(kTestDir, 'basic_mark_mesh_quiver.png')
        _CreateQuiverPlot(pts_directions, out_path)

    def DISABLED_test_hole_in_plate(self):
        kHoleInPlateDir = 'Code/alignedinfill/hole_in_plate'
        region0 = np.load(os.path.join(kHoleInPlateDir, 'region0.npy'))
        region1 = np.load(os.path.join(kHoleInPlateDir, 'region1.npy'))
        region2 = np.load(os.path.join(kHoleInPlateDir, 'region2.npy'))
        region3 = np.load(os.path.join(kHoleInPlateDir, 'region3.npy'))
        inset_spacing = 0.5
        sample_spacing = 0.1
        dedup = 1e-6
        inset_start = inset_spacing / 2
        pts_directions = mark_mesh.GetRegionDirections(region0, inset_spacing, sample_spacing,
                                                       dedup, inset_start)
        pts_directions += mark_mesh.GetRegionDirections(region1, inset_spacing, sample_spacing,
                                                        dedup, inset_start)
        pts_directions += mark_mesh.GetRegionDirections(region2, inset_spacing, sample_spacing,
                                                        dedup, inset_start)
        pts_directions += mark_mesh.GetRegionDirections(region3, inset_spacing, sample_spacing,
                                                        dedup, inset_start)
        out_path = os.path.join(kTestDir, 'hole_in_plate_quiver.png')
        _CreateQuiverPlot(pts_directions, out_path)

    def test_hole_in_plate_mesh(self):
        # Import and plot mesh
        in_path = 'Code/alignedinfill/hole_in_plate/hole_in_plate.vtk'
        mesh = meshio.read(in_path)
        mesh_pts = mesh.points
        # The mesh is in meters so let's convert to inches
        kMeterToInch = 39.3700787
        mesh_pts *= kMeterToInch
        triangles = mesh.cells[0][1]
        ax = plt.figure(figsize=(24, 24)).add_subplot(111)
        ax.set_aspect('equal')
        ax.triplot(mesh_pts[:, 0], mesh_pts[:, 1], triangles)
        # Get infill points and directions to build kdtree
        kHoleInPlateDir = 'Code/alignedinfill/hole_in_plate'
        region0 = np.load(os.path.join(kHoleInPlateDir, 'region0.npy'))
        region1 = np.load(os.path.join(kHoleInPlateDir, 'region1.npy'))
        region2 = np.load(os.path.join(kHoleInPlateDir, 'region2.npy'))
        region3 = np.load(os.path.join(kHoleInPlateDir, 'region3.npy'))
        kMmToInch = 1/25.4
        # Make these values large for now to prevent running out of memory
        inset_spacing = 10 * kMmToInch
        sample_spacing = 10 * kMmToInch
        dedup = 1e-6 * kMmToInch
        inset_start = inset_spacing / 2
        pts_directions = mark_mesh.GetRegionDirections(region0, inset_spacing, sample_spacing,
                                                       dedup, inset_start)
        pts_directions += mark_mesh.GetRegionDirections(region1, inset_spacing, sample_spacing,
                                                        dedup, inset_start)
        pts_directions += mark_mesh.GetRegionDirections(region2, inset_spacing, sample_spacing,
                                                        dedup, inset_start)
        pts_directions += mark_mesh.GetRegionDirections(region3, inset_spacing, sample_spacing,
                                                        dedup, inset_start)
        pts = np.zeros((len(pts_directions), 2))
        directions = np.zeros(len(pts_directions))
        for i, pt_direction in enumerate(pts_directions):
            pts[i, 0] = pt_direction[0][0]
            pts[i, 1] = pt_direction[0][1]
            directions[i] = pt_direction[1]
        kdtree = scipy.spatial.KDTree(pts)
        # Query triangle centroids for closest point and direction
        x = list()
        y = list()
        u = list()
        v = list()
        for i in range(triangles.shape[0]):
            p, q, r = triangles[i]
            centroid = (mesh_pts[p] + mesh_pts[q] + mesh_pts[r]) / 3
            if centroid.shape[0] == 3: centroid = centroid[:2]
            _, idx = kdtree.query(centroid)
            direction = directions[idx]
            x.append(centroid[0])
            y.append(centroid[1])
            u.append(np.cos(direction))
            v.append(np.sin(direction))
        # Make quiver plot
        ax.quiver(x, y, u, v)
        # Dump plot
        out_path = os.path.join(kTestDir, 'marked_mesh.png')
        plt.savefig(out_path)
        plt.close()

if __name__ == '__main__':
    unittest.main()