import numpy as np
from .basis_functions import linear_quad_grad
from .gauss_quad import quad_points
# from numba import autojit
# @autojit
def single_element(E,nu,h, order = 1):

    gauss_points, gauss_weights = quad_points(order)
    Jacobian = np.eye(2)*(2/h)
    detJ = np.linalg.det(Jacobian)
    K_element = np.zeros((8,8))
    gauss_points, gauss_weights = quad_points(order)
    D = (E/((1+nu)*(1-2*nu)))*np.array([[1.0-nu, nu, 0],
                                    [nu, 1.0-nu, 0],
                                    [0, 0, 1.0-2.0*nu]])
    for i in range(len(gauss_weights)):
        B = linear_quad_grad(gauss_points[i,:])
        K_element += ((B.dot(D)).dot(B.transpose()))*gauss_weights[i]*detJ


    return K_element     
