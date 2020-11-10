import sys
from typing import List

import numpy as np

pyprint_directory = r'D:\OneDrive - Leland Stanford Junior University\Research\Projects\PyPrint\Code'
sys.path.insert(1, pyprint_directory)
import pyprint

def generate_2d_gcode(polygons: List[np.ndarray], num_layers: int, layer_height_mm: float,
                      origin_offset_xbed_mm: float, origin_offset_ybed_mm: float, out_path: str,
                      plot: bool, verbose: bool=False):
    if polygons[0].shape[1] == 3:
        polygons = [np.copy(pg[:, 2]) for pg in polygons]  # only keep x and y
    assert polygons[0].shape[1] == 2, 'all polygons must be shape (n, 2)'
    offset_vector = np.array([[origin_offset_xbed_mm, origin_offset_ybed_mm]])
    polygons = [pg + offset_vector for pg in polygons]  # offset all polygons
    num_polygons = len(polygons)
    GENERATOR_VERSION = 1.0
    # start a new print job
    job = pyprint.FusedFilamentFabricationJob()
    # write job parameters to gcode file for better tracking/accountability
    job.comment('ALIGNED INFILL GCODE GENERATOR')
    job.comment('JOB PARAMETERS')
    job.comment('GENERATOR_VERSION: ' + str(GENERATOR_VERSION))
    job.comment('NOZZLE_DIAMETER_MM: ' + str(job.printer_config['nozzle_diameter_mm']))
    job.comment('LAYER_HEIGHT_MM: ' + str(layer_height_mm))
    job.comment('ORIGIN_OFFSET_XBED_MM: ' + str(origin_offset_xbed_mm))
    job.comment('ORIGIN_OFFSET_YBED_MM: ' + str(origin_offset_ybed_mm))
    job.heatup_initial([205], 60)
    job.prime_nozzle_initial()
    job.purge_nozzle_initial()
    initial_height_mm = 0.8*layer_height_mm
    current_height_mm = initial_height_mm
    # Loop over every layer
    for i in range(num_layers):
        if verbose: print('Generating layer {} of {}'.format(i + 1, num_layers))
        job.comment('Layer: ' + str(i + 1) + ' of ' + str(num_layers))
        # maybe set fan
        if i == 1:
            job.set_fan_on(127, 'half power')
        elif i == 2:
            job.set_fan_on(255, 'full power')
        # Loop over every polygon
        for j, polygon in enumerate(polygons):
            if verbose: print('Generating polygon {} of {}'.format(j + 1, num_polygons))
            # Move nozzle to starting point
            starting_point = np.array([polygon[0, 0], polygon[0, 1], current_height_mm])
            job.retract(2, 1500)
            job.move(starting_point, layer_height_mm)
            job.extrude(2)
            for k in range(1, polygon.shape[0]):  # skip first point
                job.move_extrude(np.array([polygon[k, 0], polygon[k, 1], current_height_mm]), layer_height_mm)
        # Go to next layer
        current_height_mm += layer_height_mm
    # Pre-finishing code
    job.move_rel(0, 0, 10, comment='move head out of the way vertically')
    job.move_rel(0, 50, 0, comment='move head out of the way laterally')
    job.finish()
    job.write_gcode(out_path)
    if plot: job.plot()

def main():
    pass

if __name__ == '__main__':
    main()