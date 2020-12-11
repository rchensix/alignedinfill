from typing import Tuple
import unittest

import numpy as np
import meshio

def LoadComsolMesh(in_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(in_path, 'r') as f:
        for i, line in enumerate(f):
            if '# Nodes' in line:
                node_start = i + 1  # inclusive
            elif '# Elements' in line:
                node_end = i  # exclusive
                elem_start = i + 1  # inclusive
    print(node_start)
    print(node_end)
    nodes = np.loadtxt(in_path, skiprows=node_start, max_rows=node_end-node_start)
    elems = np.loadtxt(in_path, skiprows=elem_start)
    return nodes, elems

class TestMeshing(unittest.TestCase):
    def test_hole_in_plate(self):
        in_path = 'Code/alignedinfill/hole_in_plate/hole_in_plate_mesh.txt'
        out_path = 'Code/alignedinfill/hole_in_plate/hole_in_plate.vtk'
        pts, elems = LoadComsolMesh(in_path)
        m = meshio.Mesh(pts, [('triangle', elems)])
        m.write(out_path)
    
    def test_simple_plate(self):
        in_path = 'Code/alignedinfill/hole_in_plate/simple_plate.txt'
        out_path = 'Code/alignedinfill/hole_in_plate/simple_plate.vtk'
        pts, elems = LoadComsolMesh(in_path)
        m = meshio.Mesh(pts, [('triangle', elems)])
        m.write(out_path)

if __name__ == '__main__':
    unittest.main()