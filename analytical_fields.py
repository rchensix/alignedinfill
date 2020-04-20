# Ruiqi Chen
# March 11, 2020

'''
Several analytical stress, strain, and displacement fields (along with some manually created ones) are given here.
Mostly used for testing and validation purposes.
'''

import numpy as np

def plate_with_hole_tension_stress(point: np.ndarray) -> np.ndarray:
    s = 1000
    if (point.shape == (2,)):
        x, y = point
    elif (point.shape == (3,)):
        x, y, z = point
    elif (len(point.shape) == 2 and point.shape[1] == 2):
        x = point[:, 0]
        y = point[:, 1]
    elif (len(point.shape) == 2 and point.shape[1] == 3):
        x = point[:, 0]
        y = point[:, 1]
        z = point[:, 2]
    else:
        raise Exception('Bad input shape.')
    sxx = s*(2*x**8 + x**6*(-5 + 8*y**2) + y**4*(3 + y**2 + 2*y**4) + x**2*y**2*(-18 + 13*y**2 + 8*y**4) + x**4*(3 + 7*y**2 + 12*y**4))/(2*(x**2 + y**2)**4)
    syy = s*(x**6 - 9*x**2*y**2*(-2 + y**2) + 3*y**4*(-1 + y**2) - x**4*(3 + 11*y**2))/(2*(x**2 + y**2)**4)
    sxy = s*x*y*(-5*x**4 - 2*x**2*(-3 + y**2) + 3*y**2*(-2 + y**2))/(x**2 + y**2)**4
    szz = 0
    syz = 0
    sxz = 0
    if (point.shape == (2,) or point.shape == (3,)):
        return np.array([sxx, syy, szz, sxy, syz, sxz])
    else:
        result = np.zeros((point.shape[0], 6))
        result[:, 0] = sxx
        result[:, 1] = syy
        result[:, 2] = szz
        result[:, 3] = sxy
        result[:, 4] = syz
        result[:, 5] = sxz
        return result

def cantilever_vertical_load_stress(point: np.ndarray) -> np.ndarray:
    s = 1000
    b = 3
    if (point.shape == (2,)):
        x, y = point
    elif (point.shape == (3,)):
        x, y, z = point
    elif (len(point.shape) == 2 and point.shape[1] == 2):
        x = point[:, 0]
        y = point[:, 1]
    elif (len(point.shape) == 2 and point.shape[1] == 3):
        x = point[:, 0]
        y = point[:, 1]
        z = point[:, 2]
    else:
        raise Exception('Bad input shape.')
    sxx = s/(2*b**3)*x*y
    syy = 0
    sxy = 3*s/(4*b**3)*(b**2 - y**2)
    szz = 0
    syz = 0
    sxz = 0
    if (point.shape == (2,) or point.shape == (3,)):
        return np.array([sxx, syy, szz, sxy, syz, sxz])
    else:
        result = np.zeros((point.shape[0], 6))
        result[:, 0] = sxx
        result[:, 1] = syy
        result[:, 2] = szz
        result[:, 3] = sxy
        result[:, 4] = syz
        result[:, 5] = sxz
        return result

def simply_supported_uniform_load_stress(point: np.ndarray) -> np.ndarray:
    s = 1000
    a = 12
    b = 3
    if (point.shape == (2,)):
        x, y = point
    elif (point.shape == (3,)):
        x, y, z = point
    elif (len(point.shape) == 2 and point.shape[1] == 2):
        x = point[:, 0]
        y = point[:, 1]
    elif (len(point.shape) == 2 and point.shape[1] == 3):
        x = point[:, 0]
        y = point[:, 1]
        z = point[:, 2]
    else:
        raise Exception('Bad input shape.')
    sxx = -s*y/(20*b**3)*(-6*b**2 + 15*a**2 - 15*x**2 + 10*y**2)
    syy = s/(4*b**3)*(-2*b**3 - 3*y*b**2 + y**3)
    sxy = -3*s*x/(4*b**3)*(-b**2 + y**2)
    szz = 0
    syz = 0
    sxz = 0
    if (point.shape == (2,) or point.shape == (3,)):
        return np.array([sxx, syy, szz, sxy, syz, sxz])
    else:
        result = np.zeros((point.shape[0], 6))
        result[:, 0] = sxx
        result[:, 1] = syy
        result[:, 2] = szz
        result[:, 3] = sxy
        result[:, 4] = syz
        result[:, 5] = sxz
        return result

def uniaxial_tension_stress(point: np.ndarray) -> np.ndarray:
    s = 1
    if (point.shape == (2,)):
        x, y = point
    elif (point.shape == (3,)):
        x, y, z = point
    elif (len(point.shape) == 2 and point.shape[1] == 2):
        x = point[:, 0]
        y = point[:, 1]
    elif (len(point.shape) == 2 and point.shape[1] == 3):
        x = point[:, 0]
        y = point[:, 1]
        z = point[:, 2]
    else:
        raise Exception('Bad input shape.')
    if (point.shape == (2,) or point.shape == (3,)):
        return np.array([s, 0, 0, 0, 0, 0])
    else:
        result = np.zeros((point.shape[0], 6))
        result[:, 0] = s
        return result    

# Takes a array input (n x 2) or (n x 3) representing position [x, y, z] in every row and returns array of same size as input
# with every row representing a vector
# See Barber "Elasticity" pp. 128 for solution.
def plate_with_hole_shear_displacement(point: np.ndarray) -> np.ndarray:
    a = 1  # hole radius
    s = 1  # tensile stress in xx direction
    e = 100  # young's modulus of material
    nu = 0.3  # poisson ratio of material
    eps = 1e-6  # prevents overflow at origin

    if (point.shape == (2,)):
        x, y = point
    elif (point.shape == (3,)):
        x, y, z = point
    elif (len(point.shape) == 2 and point.shape[1] == 2):
        x = point[:, 0]
        y = point[:, 1]
    elif (len(point.shape) == 2 and point.shape[1] == 3):
        x = point[:, 0]
        y = point[:, 1]
        z = point[:, 2]
    else:
        raise Exception('Bad input shape.')
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    ur = s/e*((1 + nu)*r + 4*a**2/(r + eps) - (1 + nu)*a**4/(r**3 + eps))*np.sin(2*theta)
    ut = s/e*((1 + nu)*r + 2*(1 - nu)*a**2/(r + eps) + (1 + nu)*a**4/(r**3 + eps))*np.cos(2*theta)
    u = ur*np.cos(theta) - ut*np.sin(theta)
    v = ur*np.sin(theta) + ut*np.cos(theta)

    if (point.shape == (2,)):
        return np.array([u, v])
    elif (point.shape == (3,)):
        return np.array([u, v, 0])
    else:
        result = np.zeros(point.shape)
        result[:, 0] = u
        result[:, 1] = v
        return result

# field with zero vectors in center band, otherwise equal to point itself
def center_zero_field(point: np.ndarray) -> np.ndarray:
    if (point.shape == (2,)):
        x, y = point
        u = 0 if x > -2 and x < 2 and y > -2 and y < 2 else x
        v = 0 if x > -2 and x < 2 and y > -2 and y < 2 else y
    elif (point.shape == (3,)):
        x, y, z = point
        u = 0 if x > -2 and x < 2 and y > -2 and y < 2 else x
        v = 0 if x > -2 and x < 2 and y > -2 and y < 2 else y
    elif (len(point.shape) == 2 and point.shape[1] == 2):
        x = point[:, 0]
        y = point[:, 1]
        u = x
        v = y
        u[np.logical_and(np.logical_and(np.logical_and(x > -2, x < 2), y > -2), y < 2)] = 0
        v[np.logical_and(np.logical_and(np.logical_and(x > -2, x < 2), y > -2), y < 2)] = 0
    elif (len(point.shape) == 2 and point.shape[1] == 3):
        x = point[:, 0]
        y = point[:, 1]
        z = point[:, 2]
        u = x
        v = y
        u[np.logical_and(np.logical_and(np.logical_and(x > -2, x < 2), y > -2), y < 2)] = 0
        v[np.logical_and(np.logical_and(np.logical_and(x > -2, x < 2), y > -2), y < 2)] = 0
    else:
        raise Exception('Bad input shape.')

    if (point.shape == (2,)):
        return np.array([u, v])
    elif (point.shape == (3,)):
        return np.array([u, v, 0])
    else:
        result = np.zeros(point.shape)
        result[:, 0] = u
        result[:, 1] = v
        return result

def analytical_fields_test():
    tol = 1e-12
    # Check circular tensile field
    s = plate_with_hole_tension_stress(np.array([0, 1, 0]))
    s_golden = 3
    assert np.abs(s[0] - s_golden) < tol, 'plate_with_hole_tension_stress test exceeds tolerance of {}'.format(tol)
    print('passed plate_with_hole_tension_stress test')
    print('passed all analytical_fields tests')

def main():
    analytical_fields_test()

if __name__ == '__main__':
    main()