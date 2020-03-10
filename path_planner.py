# Ruiqi Chen
# March 9, 2020

'''
Purpose of this module is to develop algorithms for path planning, given a vector field that
represents preferential path directions and magnitude of some quantity (eg. strain energy density).
'''

import numpy as np
from typing import Callable

# Create a streamline from a vector field and start point. direction=true means follow the vector 
#   at initial point. false means go opposite direction.
# step_size determines magnitude of step size for update
# stop_condition is a callable function that returns true when to stop (will either stop if 
#   stop_condition is true or if max_steps is reached)
# returns an array of points that represent streamline
def generate_streamline(field: Callable[[np.ndarray], np.ndarray], start: np.ndarray, 
                        direction: bool, stop_condition: Callable[[np.ndarray], bool], 
                        step_size: float=0.01, max_steps: int=1000) -> np.ndarray:
    curr_pt = start     
    streamline_list = [] 
    num_steps = 0 
    tol = 1e-12
    v = field(curr_pt)
    if not direction:
        v = -v
    assert np.linalg.norm(v) > tol, 'cannot start at a stationary point'
    while((not stop_condition(curr_pt)) and (num_steps != max_steps)):
        # add current point to list
        print(v)
        streamline_list.append(curr_pt.copy())
        # take a step
        curr_pt += step_size*v/np.linalg.norm(v)
        num_steps += 1
        # determine direction based on current direction (avoid drastic changes)
        vnew = field(curr_pt)
        if (np.linalg.norm(vnew) < tol):
            print('stationary point encountered at', curr_pt, 'continuing previous trajectory')
            continue
        if (vnew.dot(v) > 0):
            v = vnew
        else:
            v = -vnew
    result = np.zeros((len(streamline_list), start.size))
    for i, point in enumerate(streamline_list):
        result[i, :] = point
    return result