# Ruiqi Chen
# July 3, 2020

'''
Test polygon insetting.
'''

import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

import clipper
from visualization import plot_polygon 

class TestClipper(unittest.TestCase):
    def DISABLED_test_inset(self):
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
        insets = [0.2, 0.5, 1, 1.5]
        num_solutions = [1, 2, 2, 0]
        for inset_amount, num_sol in zip(insets, num_solutions):
            solution = clipper.inset_polygon(dumbbell, inset_amount)
            self.assertEqual(len(solution), num_sol)
            for sol in solution:
                plot_polygon(np.array(sol), ax)
        plt.show()

    def test_hole_in_plate(self):
        directory = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\Aligned Infills\Code\alignedinfill\hole_in_plate'
        streamline0 = np.load(os.path.join(directory, 'streamline0.npy'))
        streamline1 = np.load(os.path.join(directory, 'streamline1.npy'))
        # Region 0: top half
        region0 = np.zeros((streamline0.shape[0] + 4, 3))
        region0[0:-4] = streamline0
        region0[-4:] = np.array([[-2.5, 0.13291783, 0],
                                 [-2.5, 2.5, 0],
                                 [2.5, 2.5, 0],
                                 [2.5, 0.13291783, 0]])
        # Region 1: bottom half
        region1 = np.zeros((streamline1.shape[0] + 4, 3))
        region1[0:-4] = streamline1
        region1[-4:] = np.array([[-2.5, -0.13291783, 0],
                                 [-2.5, -2.5, 0],
                                 [2.5, -2.5, 0],
                                 [2.5, -0.13291783, 0]])   
        # Region 2: left half
        region2 = np.zeros((90 + 50 + 90 + 2, 3))
        region2[0] = np.array([-2.5, -0.13291738, 0])
        region2[1:91] = streamline1[-1:-91:-1]
        arc2 = np.zeros((50, 3))
        theta = np.linspace(220*np.pi/180, 140*np.pi/180, 50)
        arc2[:, 0] = 1.02*np.cos(theta)
        arc2[:, 1] = 1.02*np.sin(theta)
        region2[91:141] = arc2
        region2[141:231] = streamline0[-90::1]
        region2[231] = np.array([-2.5, 0.13291738, 0])
        # Region 3: right half
        region3 = np.zeros((90 + 50 + 90 + 2, 3))
        region3[0] = np.array([2.5, -0.13291738, 0])
        region3[1:91] = streamline1[:90]
        arc3 = np.zeros((50, 3))
        theta = np.linspace(-40*np.pi/180, 40*np.pi/180, 50)
        arc3[:, 0] = 1.02*np.cos(theta)
        arc3[:, 1] = 1.02*np.sin(theta)
        region3[91:141] = arc3
        region3[141:231] = streamline0[89::-1]
        region3[231] = np.array([2.5, 0.13291738, 0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        plot_polygon(region0, ax)
        plot_polygon(region1, ax)
        plot_polygon(region2, ax)
        plot_polygon(region3, ax)
        np.save(os.path.join(directory, 'region0.npy'), region0)
        np.save(os.path.join(directory, 'region1.npy'), region1)
        np.save(os.path.join(directory, 'region2.npy'), region2)
        np.save(os.path.join(directory, 'region3.npy'), region3)
        # plt.show()
        insets = np.arange(0.25, 2, 0.25)
        # nozzle_diam_inch = 0.019685
        # insets = np.arange(nozzle_diam_inch/2, 2, nozzle_diam_inch)
        counter = 0
        for inset in insets:
            for region in [region0, region1, region2, region3]:
                solution = clipper.inset_polygon(region, inset)
                for sol in solution:
                    plot_polygon(np.array(sol), ax)
                    out_path = os.path.join(directory, 'polygons', '{}.npy'.format(counter))
                    # np.save(out_path, np.array(sol))
                    counter += 1
        plt.show()

    def DISABLED_test_simply_supported_beam(self):
        region = np.array([[0, 0, 0],
                           [-0.25, 0, 0],
                           [-0.25, 5, 0],
                           [-30, 5, 0],
                           [-30, -5, 0],
                           [30, -5, 0],
                           [30, 5, 0],
                           [0.25, 5, 0],
                           [0.25, 0, 0]])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        plot_polygon(region, ax)
        insets = np.arange(0.25, 2, 0.25)
        for inset in insets:
            solution = clipper.inset_polygon(region, inset)
            for sol in solution:
                plot_polygon(np.array(sol), ax)
        plt.show()

if __name__ == '__main__':
    unittest.main()