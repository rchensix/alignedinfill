import os
import unittest

import numpy as np

import gcode

class TestGcodeGenerator(unittest.TestCase):
    def test_hole_in_plate(self):
        # Import polygons
        kInchToMm = 25.4
        polygons = list()
        hole_in_plate_dir = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\Code\alignedinfill\hole_in_plate'
        hole_in_plate_polygon_dir = os.path.join(hole_in_plate_dir, 'polygons')
        filelist = os.listdir(hole_in_plate_polygon_dir)
        for f in filelist:
            in_path = os.path.join(hole_in_plate_polygon_dir, f)
            if os.path.isfile(in_path) and '.npy' in f:
                polygons.append(np.load(in_path) * kInchToMm)  # convert from inch to mm)
        num_layers = 2
        layer_height_mm = 0.5
        origin_offset_xbed_mm = 290/2
        origin_offset_ybed_mm = 275/2
        out_path = os.path.join(hole_in_plate_dir, 'hole_in_plate_alignedinfill.gcode')
        plot = True
        verbose = True  # debugging
        gcode.generate_2d_gcode(polygons, num_layers, layer_height_mm, origin_offset_xbed_mm,
                                origin_offset_ybed_mm, out_path, plot, verbose)

if __name__ == '__main__':
    unittest.main()