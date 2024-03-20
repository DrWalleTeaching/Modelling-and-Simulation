import numpy as np
from .recover_mechanics import mechanics
from .construct_global_K import stiffness_image, applyBC
from scipy.sparse.linalg import spsolve
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import cos,sin,pi
# from numba import autojit
from .timer import Timer
import copy
def compress(I,percentage,h=1.8):
    '''
    Function which compresses an image along the y axis. Arguments are:
    I : image in MPa
    percentage : amount of compression
    h : the pixel width in mm

    '''
    t = Timer()
    t.start('Generating Matrix')
    K=stiffness_image(I,h=h,order=2)
    Korg = copy.copy(K)
    t.stop('Generating Matrix')
    t.start('Calculation of BC')
    node_map_size = np.array(I.shape)+1
    node_map = np.arange(np.prod(node_map_size)).reshape(node_map_size)
    dofx =np.copy(node_map)
    dofy = np.arange(np.prod(node_map_size),2*np.prod(node_map_size)).reshape(node_map_size)
    f=np.zeros(K.shape[0])
    height = I.shape[1]*h
    disp = -percentage*height

    BC={}
    BC['y'] = np.concatenate([np.arange(0,I.shape[1]+1),np.arange(0,I.shape[1]+1)])
    BC['x'] = np.concatenate([np.zeros(len(node_map[0,:].flatten()),dtype=int), I.shape[0]*np.ones(len(node_map[0,:].flatten()))])
    BC['values_y'] = np.zeros(len(BC['y']))
    BC['values_x'] = np.concatenate([np.zeros(len(node_map[0,:].flatten()),dtype=int), disp*np.ones(len(node_map[0,:].flatten()))])
    t.stop('Calculation of BC')
    t.start('Applying essential BC')
    K,f=applyBC(K,BC,I,f)
    t.stop('Applying essential BC')
    t.start('Solving K*u=f')
    u = spsolve(K,f)
    f=Korg.dot(u)
    t.stop('Solving K*u=f')
    t.start('Post-processing')

    F_temp = f[dofy[:,0].flatten()]
    F=np.sum(F_temp)
    #rescale because of image being to thin
    #F/=-4.05942893898e-10
    #F*=10

    #recover mechanics
    SED,e_xx,e_yy,e_xy,s_xx,s_yy,s_xy= mechanics(I,u,h)
    strains={'xx':e_xx,'yy':e_yy,'xy':e_xy}
    stresses={'xx':s_xx,'yy':s_yy,'xy':s_xy}
    t.stop('Post-processing')
    t_total=0
    for key,values in t.elapsed_times.items():
        print(key,':',values,'s')
        t_total+=values
    print('Total Time :',t_total)
    return SED,strains,stresses


def bend(I,angle,h=0.0105,center_x=None):
    '''
    Function which bends an image applying a fixed rotation to the top and bottom faces in y. Arguments are:
    I : image in MPa
    angle : amount of compression
    h : the pixel width in mm

    '''

    t = Timer()
    #generate stiffness matrix
    t.start('Generating Matrix')
    K=stiffness_image(I,h=h,order=2)
    Korg = copy.copy(K)
    t.stop('Generating Matrix')
    t.start('Calculation of BC')



    #if no center given assume 50% of x
    if center_x is None:
        center_x=(I.shape[0]/2)*h
    angle = pi*angle/180.0 #get angle in radians
    R = np.array([[cos(angle), -sin(angle)],
                  [sin(angle), cos(angle)]]) #rotation matrix

    #rotate top of image
    top_x =np.arange(I.shape[0]+1)*h - center_x
    top_y =np.zeros(I.shape[0]+1)
    disp_top_x=np.zeros(top_x.shape)
    disp_top_y=np.zeros(top_x.shape)
    for i in range(len(top_x)):

        Pt = np.asarray([top_x[i], top_y[i] ],dtype=float)
        tPt = R.dot(Pt)
        disp = tPt -Pt
        disp_top_x[i]=disp[0]
        disp_top_y[i]=disp[1]
        #print('top',disp)

    #equal and opposite rotation of bottom
    angle*=-1
    R = np.array([[cos(angle), -sin(angle)],
                  [sin(angle), cos(angle)]])
    bot_x =np.arange(I.shape[0]+1)*h - center_x
    bot_y = np.zeros(I.shape[0]+1)
    disp_bot_x=np.zeros(top_x.shape)
    disp_bot_y=np.zeros(top_x.shape)
    for i in range(len(top_x)):
        Pt = np.asarray([bot_x[i], bot_y[i] ])
        tPt = R.dot(Pt)
        disp = tPt -Pt
        disp_bot_x[i]=disp[0]
        disp_bot_y[i]=disp[1]
        #print('bot',disp)

    node_map_size = np.array(I.shape)+1
    node_map = np.arange(np.prod(node_map_size)).reshape(node_map_size)

    dofx =np.copy(node_map)
    dofy = np.arange(np.prod(node_map_size),2*np.prod(node_map_size)).reshape(node_map_size)
    f=np.zeros(K.shape[0])


    BC={}
    BC['x'] = np.concatenate([np.arange(I.shape[0]+1),np.arange(I.shape[0]+1)])
    BC['y'] = np.concatenate([top_y,bot_y+I.shape[1]])
    BC['values_x'] = np.concatenate([disp_top_x,disp_bot_x])
    BC['values_y'] = np.concatenate([disp_top_y,disp_bot_y])
    t.stop('Calculation of BC')
    t.start('Applying essential BC')
    K,f=applyBC(K,BC,I,f)
    t.stop('Applying essential BC')
    t.start('Solving K*u=f')
    u=spsolve(K,f)
    t.stop('Solving K*u=f')
    t.start('Post-processing')
    f=Korg.dot(u)
    f*=1e-10
    #f*=10

    f_top = f[dofy[:,0].flatten()]
    Moment = 0
    for i in range(len(top_x)):
        Moment+=top_x[i]*f_top[i]

    SED,e_xx,e_yy,e_xy= mechanics(I,u,h)
    strains={'xx':e_xx,'yy':e_yy,'xy':e_xy}
    t.stop('Post-processing')
    t_total=0
    for key,values in t.elapsed_times.items():
        print(key,':',values,'s')
        t_total+=values
    print('Total Time :',t_total)
    return SED,strains
