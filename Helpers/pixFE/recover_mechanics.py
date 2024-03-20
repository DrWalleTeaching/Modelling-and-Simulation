
import numpy as np
from .basis_functions import linear_quad_grad
from .gauss_quad import quad_points
# from numba import autojit
#@autojit
def mechanics(I,u,h):
    node_map_size = np.array(I.shape)+1
    N_nodes = np.prod(node_map_size)
    node_map = np.arange(np.prod(node_map_size)).reshape(node_map_size)
    dof = np.zeros(8,dtype=int)
    E=1
    nu=0.3
    gauss_points, gauss_weights = quad_points(2)
    D = (E/((1+nu)*(1-2*nu)))*np.array([[1.0-nu, nu, 0],
                                    [nu, 1.0-nu, 0],
                                    [0, 0, 1.0-2.0*nu]])

    #B = linear_quad_grad((0,0))*(2/h)
    StrainEnergy = np.zeros(I.shape)
    e_xx = np.zeros(I.shape)
    e_yy = np.zeros(I.shape)
    e_xy = np.zeros(I.shape)
    s_xx = np.zeros(I.shape)
    s_yy = np.zeros(I.shape)
    s_xy = np.zeros(I.shape)
    
    for x in range((I.shape[0])):
        for y in range(I.shape[1]):
            #node 0
            dof[0] = node_map[x,y]
            dof[1] = node_map[x,y]+N_nodes
            #node 1
            dof[2] = node_map[x+1,y]
            dof[3] = node_map[x+1,y]+N_nodes
            #node 2
            dof[4] = node_map[x+1,y+1]
            dof[5] = node_map[x+1,y+1]+N_nodes        
            #node 3
            dof[6] = node_map[x,y+1]
            dof[7] = node_map[x,y+1]+N_nodes  
            u_local = u[np.ix_(dof)]
            strain=np.zeros(3)
            for i in range(len(gauss_weights)):
                B = linear_quad_grad(gauss_points[i,:])
                strain += (B.transpose()*(2/h)).dot(u_local)
            strain/=len(gauss_weights)
            stress = D.dot(strain)
            SED = 0.5*strain*stress
            e_xx[x,y] = strain[0]
            e_yy[x,y] = strain[1]
            e_xy[x,y] = strain[2]
            s_xx[x,y] = stress[0]
            s_yy[x,y] = stress[1]
            s_xy[x,y] = stress[2]
            StrainEnergy[x,y]=np.sum(SED)

    return StrainEnergy,e_xx,e_yy,e_xy,s_xx,s_yy,s_xy