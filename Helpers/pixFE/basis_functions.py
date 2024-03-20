import numpy as np
# from numba import autojit
# @autojit
def linear_quad(X):
    '''
    Function to calculate linear finite element basis function for unit quad. 
    As argument accepts X a 2d poisiton in the space [-1,1]
    returns a vector of values of each of the 4 basis functions
    '''
    N = np.zeros(4)
    N[0] = ((1 - X[0])*(1 - X[1]))*0.25
    N[1] = ((1 + X[0])*(1 - X[1]))*0.25
    N[2] = ((1 + X[0])*(1 + X[1]))*0.25
    N[3] = ((1 - X[0])*(1 + X[1]))*0.25
    return N
# @autojit
def linear_quad_grad(X):
    '''
    Function to calculate gradients of linear finite element basis function for unit quad. 
    As argument accepts X a 2d poisiton in the space [-1,1]
    returns the strain displacement matrix (Commonly callled B, where K_element= Bt.D.B)
    '''
    B = np.zeros((8,3))
    B[0,0] = (( -1)*(1 - X[1]))*0.25
    B[1,1] = ((1 - X[0])*(-1))*0.25
    B[0,2] = B[1,1]
    B[1,2] = B[0,0] 
    B[2,0] =   (( +1)*(1 - X[1]))*0.25
    B[3,1] =   ((1 + X[0])*(-1))*0.25
    B[2,2] = B[3,1]
    B[3,2] = B[2,0] 
    B[4,0] = (( +1)*(1 + X[1]))*0.25
    B[5,1] = ((1 + X[0])*(+1))*0.25
    B[4,2] = B[5,1]
    B[5,2] = B[4,0]
    B[6,0] = ((-1)*(1 + X[1]))*0.25
    B[7,1] = ((1 - X[0])*(+1))*0.25
    B[6,2] = B[7,1]
    B[7,2] = B[6,0] 
    return B