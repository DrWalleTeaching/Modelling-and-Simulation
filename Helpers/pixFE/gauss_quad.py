import numpy as np

def quad_points(order):
    if order == 1:
        return np.array([[0,0]]), np.array([2])

    if order == 2:
        X=np.array([
            [-0.57735,-0.57735],
            [0.57735,-0.57735],
            [0.57735,0.57735],
            [-0.57735,0.57735],
        ])
        W=np.ones(4)
        return X, W

    if order == 3:
        X=np.array([
        [0,0],
        [-0.774597,-0.774597],
        [0.774597,-0.774597],
        [0.774597,0.774597],
        [-0.774597,0.774597],
        ])
        W = np.ones(5)
        W*=0.555556
        W[0] = 0.888889