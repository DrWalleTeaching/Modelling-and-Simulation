from scipy.sparse import lil_matrix,csr_matrix,coo_matrix
import numpy as np
from .single_element import single_element
import itertools
# from numba import autojit

def stiffness_image(I,nu=0.3, h=0.0105,order=1):
    node_map_size = np.array(I.shape)+1
    N_nodes = np.prod(node_map_size)
    node_map = np.arange(np.prod(node_map_size)).reshape(node_map_size)
    #K_global = lil_matrix((2*N_nodes,2*N_nodes))
    K_element = single_element(1,nu,h,order=order)*I.shape[0]*h
    dof = np.zeros(8)
    
    rows = []#np.empty( shape=(0) )

    cols = []#np.empty( shape=(0) )

    vals = []#np.empty( shape=(0) )
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
            
            K_loc = I[x,y]*K_element
            #idx =np.array(np.ix_(dof,dof))
            #print(idx)
            idx = np.meshgrid(dof,dof)
            #idx_flat = np.reshape(idx,8*8)
            rows.append(np.array(idx[0].flatten()))

            cols.append(np.array(idx[1].flatten()))

            vals.append(K_loc.flatten())
    rows = np.concatenate(rows).flatten()
    cols = np.concatenate(cols).flatten()
    vals = np.concatenate(vals).flatten()

    K_global=coo_matrix((vals, (rows, cols)), shape=(2*N_nodes,2*N_nodes))
    return(K_global.tocsr())
# @autojit
def applyBC(A,BC,I,f):
    '''
    This function applies the BC to the matrix
    '''
    
    if BC:
        fixedDofs=np.zeros(len(BC['values_y']),dtype=int)
        bc_x=BC['x'].astype(int)
        bc_y=BC['y'].astype(int)
        node_map_size = np.array(I.shape)+1
        node_map_y = np.arange(np.prod(node_map_size)).reshape(node_map_size)
        node_map_x = np.arange(np.prod(node_map_size),2*np.prod(node_map_size)).reshape(node_map_size)
        #print('xdof',bc_x)
        #print('ydof',bc_y)
        fixedDofs = np.concatenate([node_map_x[bc_x,bc_y],node_map_y[bc_x,bc_y]])
        #bcwt=A.diagonal().sum()/A.shape[0]
        values = np.concatenate([BC['values_x'],BC['values_y']])
        for i in fixedDofs:
            _, J = A[i,:].nonzero()   # ignored the first return value, which is just [i,...,i]
            for j in J:
                if j!=i:
                    A[i,j] = 0.0
            A[i,i] = 1.0

        f[fixedDofs]=values

    return A,f
