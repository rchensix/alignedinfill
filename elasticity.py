# Ruiqi Chen
# March 10, 2020

import numpy as np

def constants_to_compliance_matrix(exx: float, eyy: float, ezz: float, nuxy: float, nuyz: float, nuxz: float, gxy: float, gyz: float, gxz: float) -> np.ndarray:
    assert exx > 0, 'modulus must be greater than zero'
    assert eyy > 0, 'modulus must be greater than zero'
    assert ezz > 0, 'modulus must be greater than zero'
    assert gxy > 0, 'modulus must be greater than zero'
    assert gyz > 0, 'modulus must be greater than zero'
    assert gxz > 0, 'modulus must be greater than zero'
    return np.array([[1/exx, -nuxy/exx, -nuxz/exx, 0, 0, 0], 
                     [-nuxy/exx, 1/eyy, -nuyz/eyy, 0, 0, 0],
                     [-nuxz/exx, -nuyz/eyy, 1/ezz, 0, 0, 0],
                     [0, 0, 0, 1/(2*gxy), 0, 0],
                     [0, 0, 0, 0, 1/(2*gyz), 0],
                     [0, 0, 0, 0, 0, 1/(2*gxz)]])

# Converts stress or tensorial strain vector into matrix.
# Coordinate order is expected to be xx, yy, zz, xy, yz, xz
def str_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    assert vector.shape == (6,), 'input vector must be shape (6,), not ' + vector.shape
    return np.array([[vector[0], vector[3], vector[5]],
                     [vector[3], vector[1], vector[4]],
                     [vector[5], vector[4], vector[2]]])

# Converts stress or tensorial strain matrix into vector.
def str_matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    tol = 1e-12
    assert matrix.shape == (3, 3), 'input matrix must be shape (3, 3), not ' + matrix.shape
    for i in range(3):
        for j in range(3):
            if i == j: continue
            assert matrix[i][j] - matrix[j][i] < tol, 'matrix is not symmetric within a tolerance of ' + tol
    return np.array([matrix[0][0], matrix[1][1], matrix[2][2], matrix[0][1], matrix[1][2], matrix[0][2]])

# Transforms stress or tensorial strain vector by rotating about z axis by angle_rad (positive is CCW)
def rotate_str_vector_z(str_vector: np.ndarray, angle_rad: float) -> np.ndarray:
    mat = str_vector_to_matrix(str_vector)
    # We are rotating coordinate system, not a vector in a fixed coordinate system.
    rot = np.array([[np.cos(angle_rad), np.sin(angle_rad), 0],
                    [-np.sin(angle_rad), np.cos(angle_rad), 0],
                    [0, 0, 1]])
    return str_matrix_to_vector(rot @ mat @ np.transpose(rot))

# Transforms compliance matrix by rotating about z axis (positive CCW)
# See Section 3.2.11 of http://solidmechanics.org/text/Chapter3_2/Chapter3_2.htm
def rotate_compliance_matrix_z(compliance: np.ndarray, angle_rad: float) -> np.ndarray:
    # TODO: fix this, the indices in the link above are not the same as the ones in this file
    raise Exception('Implementation is wrong, do not use')
    assert compliance.shape == (6,6), 'compliance matrix shape must be (6, 6), not {}'.format(compliance.shape)
    c = np.cos(-angle_rad)
    s = np.sin(-angle_rad)
    kinv = np.array([[c**2, s**2, 0, 0, 0, 2*c*s],
                    [s**2, c**2, 0, 0, 0, -2*c*s],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, c, s, 0],
                    [0, 0, 0, -s, c, 0],
                    [-c*s, c*s, 0, 0, 0, c**2 - s**2]])
    return np.transpose(kinv) @ compliance @ kinv

def strain_energy_density(compliance_matrix: np.ndarray, stress_vector: np.ndarray) -> float:
    strain_vector = compliance_matrix @ stress_vector
    u = 0.5*(strain_vector.dot(stress_vector) + strain_vector[3:].dot(stress_vector[3:]))
    return u

def von_mises_stress(stress_vector: np.ndarray) -> float:
    return np.sqrt(0.5*((stress_vector[0] - stress_vector[1])**2 + \
                        (stress_vector[1] - stress_vector[2])**2 + \
                        (stress_vector[2] - stress_vector[0])**2 + \
                        6*stress_vector[4]**2 + \
                        6*stress_vector[5]**2 + \
                        6*stress_vector[6]**2))

def elasticity_test():
    # some test values
    tol = 1e-12
    exx = 4
    eyy = 5
    ezz = 6
    vxy = 0.2
    vyz = 0.3
    vxz = 0.4
    gxy = 1
    gyz = 2
    gxz = 3
    # test compliance matrix creation
    s = constants_to_compliance_matrix(exx, eyy, ezz, vxy, vyz, vxz, gxy, gyz, gxz)
    s_golden = np.array([[0.25, -0.05, -0.1, 0, 0, 0],
                         [-0.05, 0.2, -0.06, 0, 0, 0],
                         [-0.1, -0.06, 1/6, 0, 0, 0],
                         [0, 0, 0, 0.5, 0, 0],
                         [0, 0, 0, 0, 0.25, 0],
                         [0, 0, 0, 0, 0, 1/6]])
    assert np.linalg.norm(s - s_golden) < tol, 'constants_to_compliance_matrix test exceeds tolerance of {}'.format(tol)                     
    print('passed constants_to_compliance_matrix test')
    # test vector and matrix conversions
    sigma_vec = np.array([-2, -1, 0, 1, 2, 3])
    sigma_mat = str_vector_to_matrix(sigma_vec)
    sigma_mat_golden = np.array([[-2, 1, 3],
                                 [1, -1, 2],
                                 [3, 2, 0]])
    assert np.linalg.norm(sigma_mat - sigma_mat_golden) < tol, 'str_vector_to_matrix test exceeds tolerance of {}'.format(tol)
    print('passed str_vector_to_matrix test')   
    sigma_vec_golden = sigma_vec.copy()
    sigma_vec_new = str_matrix_to_vector(sigma_mat)
    assert np.linalg.norm(sigma_vec_new - sigma_vec_golden) < tol, 'str_matrix_to_vector test exceeds tolerance of {}'.format(tol)      
    print('passed str_matrix_to_vector test')     
    # test rotation by rotating by 90 degrees, then by 69 degrees
    sigma_vec_rot90 = rotate_str_vector_z(sigma_vec, np.pi/2)
    sigma_vec_rot90_golden = np.array([-1, -2, 0, -1, -3, 2])
    assert np.linalg.norm(sigma_vec_rot90 - sigma_vec_rot90_golden) < tol, 'rotate_str_vector_z 90 test exceeds tolerance of {}'.format(tol)
    sigma_vec_rot69 = rotate_str_vector_z(sigma_vec, 69*np.pi/180)    
    sigma_vec_rot69_golden = np.array([-0.4592969809024448, 
                                       -2.540703019097555,
                                       0,
                                       -0.4085795222979652,
                                       -2.084005380401005,
                                       2.942264701630304])      
    assert np.linalg.norm(sigma_vec_rot69 - sigma_vec_rot69_golden) < tol, 'rotate_str_vector_z 69 test exceeds tolerance of {}'.format(tol)  
    print('passed rotate_str_vector_z test')
    # Compute strain energy
    u = strain_energy_density(s, sigma_vec)
    u_golden = 7
    assert np.abs(u - u_golden) < tol, 'strain_energy_density test exceeds tolerance of {}'.format(tol)
    print('passed strain_energy_density test')
    print('passed all elasticity tests')                                 

def main():
    elasticity_test()

if __name__ == '__main__':
    main()
