import unittest

import matplotlib.pyplot as plt
from shapely.geometry import box

import part_segmenter
import visualization

class PartSegmenterTest(unittest.TestCase):
    def test_three_point_bend(self):
        part = box(-0.06, -0.0125, 0.06, 0.0125)
        fem_result = r'FEM/three_point_bend_stress.vtu'
        grid_spacing = 2e-3 # 0.5e-3
        min_dist = 6e-3
        max_num_lines = 3
        streamlines, debug_x, debug_y = part_segmenter.SegmentPart(part, fem_result, grid_spacing,
                                                                   min_dist, max_num_lines,
                                                                   debug_mode=True)
        # Plot results
        ax = plt.figure().add_subplot(111)
        ax.set_aspect('equal')
        for sl in streamlines:
            visualization.plot_streamline(sl, ax)
        ax.scatter(debug_x, debug_y)
        plt.show()

if __name__ == '__main__':
    unittest.main()