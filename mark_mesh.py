from typing import Dict, List, Tuple

import numpy as np

import clipper

def _SampleLine(pt0: np.ndarray, pt1: np.ndarray, sample_spacing: float, offset: float=0) -> \
                List[np.ndarray]:
    """This samples a line between pt0 and pt1 and potentially offsets pt0 and pt1 in the result to
    prevent double counting.
    """
    diff = pt1 - pt0
    dist = np.sqrt(diff.dot(diff))
    num_pts = int(np.ceil(dist / sample_spacing)) + 1
    pts = list()
    for i in range(num_pts):
        if i == 0:
            if offset == 0:
                pts.append(pt0)
            else:
                pts.append(pt0 * (1 - offset) + pt1 * offset)
        elif i == num_pts - 1:
            if offset == 0:
                pts.append(pt1)
            else:
                pts.append(pt0 * (offset) + pt1 * (1 - offset))
        else:
            theta = i / (num_pts - 1)
            pts.append(pt0 * (1 - theta) + pt1 * theta)
    return pts

def GetRegionDirections(region: np.ndarray, inset_spacing: float, sample_spacing: float,
                        deduplicate_offset: float=0, inset_start: float=1e-6) -> \
                        List[Tuple[np.ndarray, float]]:
    inset_val = inset_start
    pts_directions = list()
    while True:
        solution = clipper.inset_polygon(region, inset_val)
        if len(solution) == 0: break
        for sol in solution:
            sol_array = np.array(sol).squeeze()
            num_pts = sol_array.shape[0]
            for i in range(num_pts):
                pt0 = sol_array[i]
                if i == num_pts - 1:
                    pt1 = sol_array[0]
                else:
                    pt1 = sol_array[i + 1]
                sampled_pts = _SampleLine(pt0, pt1, sample_spacing, deduplicate_offset)
                diff = pt1 - pt0
                angle = np.arctan2(diff[1], diff[0])
                for pt in sampled_pts: pts_directions.append((pt, angle))
        inset_val += inset_spacing
    return pts_directions

def main():
    pass

if __name__ == '__main__':
    main()