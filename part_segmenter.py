from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon

import clipper
import elasticity
import numerical_fields
import path_planner

def GenerateAlignmentFieldFromFemResult(fem_result: str) -> Callable[[np.ndarray], np.ndarray]:
    # Get RBF interpolation of fem result
    sxx, syy, sxy = numerical_fields.rbf_stress_interpolator(fem_result, 'gaussian')
    # Create alignment functor
    def f(points: np.ndarray):
        if points.shape == (3,): 
            pts = points.copy().reshape((1, -1))
        else:
            pts = points
        # align with principal stress direction
        stress = np.zeros((pts.shape[0], 6))
        stress[:, 0] = sxx(pts[:, 0], pts[:, 1], pts[:, 2])
        stress[:, 1] = syy(pts[:, 0], pts[:, 1], pts[:, 2])
        stress[:, 3] = sxy(pts[:, 0], pts[:, 1], pts[:, 2])
        eps = 1e-3  # prevent division by zero
        theta = 0.5*np.arctan2(2*stress[:, 3], stress[:, 0] - stress[:, 1] + eps)
        if points.shape == (3,):
            result = np.linalg.norm(stress)*np.array([np.cos(theta[0]), np.sin(theta[0]), 0])
        else:
            result = np.zeros(points.shape)
            result[:, 0] = np.linalg.norm(stress)*np.cos(theta)
            result[:, 1] = np.linalg.norm(stress)*np.sin(theta)
        return result
    return f

def SegmentPart(part: Polygon, fem_result: str, grid_spacing: float,
                min_distance_between_streamlines: float=-1,
                max_num_streamlines: Union[int, None]=None,
                debug_mode: bool=False) -> List[np.ndarray]:
    """Given a part and fem result for the part, generates streamlines from max stress points.
    Points are separated by at least min_distance_between_streamlines. Returns list of
    disconnected streamlines.
    """
    # Get RBF interpolation of fem result
    sxx, syy, sxy = numerical_fields.rbf_stress_interpolator(fem_result, 'gaussian')
    # Generate sampling grid
    x_min, y_min, x_max, y_max = part.bounds
    num_pts_x = int(np.ceil((x_max - x_min)/grid_spacing))
    num_pts_y = int(np.ceil((y_max - y_min)/grid_spacing))
    xvec = np.linspace(x_min, x_max, num_pts_x)
    yvec = np.linspace(y_min, y_max, num_pts_y)
    sampled_pt_stresses = list()
    for x in xvec:
        for y in yvec:
            pt = (x, y, 0)
            # Make sure pt is inside part
            if not part.contains(Point(x, y)): continue
            # Assume z stresses are zero (plane stress)
            stress_vec = np.array([sxx(*pt), syy(*pt), 0, sxy(*pt), 0, 0])
            s_vonmises = elasticity.von_mises_stress(stress_vec)
            sampled_pt_stresses.append((pt, s_vonmises))
    # Plot contour of stress field
    # if debug_mode:
    #     ax = plt.figure().add_subplot(111)
    #     x, y = np.meshgrid(xvec, yvec)
    #     sxx_grid = sxx(x, y, np.zeros_like(x))
    #     syy_grid = syy(x, y, np.zeros_like(x))
    #     sxy_grid = sxy(x, y, np.zeros_like(x))
    #     vonmises_grid = np.sqrt(sxx_grid**2 - sxx_grid*syy_grid + syy_grid**2 + 3*sxy_grid**2)
    #     contour = ax.contourf(x, y, vonmises_grid, cmap='jet')
    #     plt.colorbar(contour)
    #     plt.show()
    # Sort sampled points
    sorted_pts = sorted(sampled_pt_stresses, key=lambda item: item[1], reverse=True)
    # Generate streamlines
    field = GenerateAlignmentFieldFromFemResult(fem_result)
    def stop_cond(pt: np.ndarray) -> bool:
        return not part.contains(Point(pt[0], pt[1]))
    step_size = grid_spacing / 10
    streamlines = list()
    if debug_mode:
        debug_x = list()
        debug_y = list()
    for pt in sorted_pts:
        point = pt[0]
        # Make sure starting point is far away from any other streamlines
        too_close = False
        if min_distance_between_streamlines > 0:
            for sl in streamlines:
                # Convert sl to shapely linestring
                sl_pts = list()
                for i in range(sl.shape[0]):
                    sl_pts.append((sl[i, 0], sl[i, 1]))
                linestr = LineString(sl_pts)
                if linestr.distance(Point(point[0], point[1])) < min_distance_between_streamlines:
                    too_close = True
                    break
            if too_close: continue  # skip this point
        if debug_mode:
            debug_x.append(point[0])
            debug_y.append(point[1])
        line1 = path_planner.generate_streamline(field, np.array(point), True, stop_cond, step_size)
        line2 = path_planner.generate_streamline(field, np.array(point), False, stop_cond, step_size)
        # Join line1 and line2
        line = np.zeros((line1.shape[0] + line2.shape[0] - 1, 3))
        for i in range(line1.shape[0] - 1):
            line[i] = line1[-(i + 1)]
        line[line1.shape[0] - 1:, :] = line2
        streamlines.append(line)
        if max_num_streamlines is not None and len(streamlines) == max_num_streamlines: break
    if debug_mode:
        return streamlines, debug_x, debug_y
    else:
        return streamlines

def main():
    pass

if __name__ == '__main__':
    main()