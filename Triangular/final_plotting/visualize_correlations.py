import numpy as np

import sys
sys.path.append("..")
from interactions import buildlattice_alltoall_primitive_vector

default_p1 = (1., 0.)
default_p2 = (-1./2, np.sqrt(3)/2)

def get_k_mesh(L):

    yscale = (2*np.pi/np.sqrt(3)) / (L/2)
    xscale = (2*np.pi/3) / (L/2)
    xpoints = []

    xpoints.append(0)

    nums = int(L)
    for num in range(1,2*nums+1):
        xpoints.append(num * xscale)
        xpoints.append(-1 * num * xscale)

    ypoints_bottom = np.arange(-2*np.pi/np.sqrt(3),0,yscale/2)
    ypoints_top = np.arange(2*np.pi/np.sqrt(3),0,-yscale/2)
    y0 = np.array([0])
    ypoints = np.concatenate((ypoints_top, y0, ypoints_bottom[::-1]))

    num_x = len(xpoints)
    all_x_points = []
    all_y_points = []
    for y_point_i,y_point in enumerate(ypoints):
        if y_point_i %2 == 0:
            all_x_points.append(np.array(xpoints))
            all_y_points.append([y_point]*num_x)
        else:
            all_x_points.append(np.array(xpoints)+xscale/2)
            all_y_points.append([y_point]*num_x)

    return np.array(all_x_points).flatten(),np.array(all_y_points).flatten()

def get_k_mesh_hex(L):

    yscale = (2*np.pi/np.sqrt(3)) / (L/2)
    xscale = (2*np.pi/3) / (L/2)
    xpoints = []


    numx = int(L)
    for num in range(1,numx+1)[::-1]:
        xpoints.append(num * xscale)
    xpoints.append(0)
    for num in range(1,numx+1):
        xpoints.append(-1 * num * xscale)

    ypoints_bottom = np.arange(-2*np.pi/np.sqrt(3),0,yscale/2)
    ypoints_top = np.arange(2*np.pi/np.sqrt(3),0,-yscale/2)
    y0 = np.array([0])
    ypoints = np.concatenate((ypoints_top, y0, ypoints_bottom[::-1]))
    
    midpoint = int(len(xpoints)/2)
    all_x_points = np.zeros((1))
    all_y_points = np.zeros((1))
    for y_point_i,y_point in enumerate(ypoints):
        if y_point_i % 2 == 0:
            if y_point_i <= L:
                x_len_i = 1 + numx + y_point_i
                half = int(x_len_i/2)
                xpoints_i = xpoints[midpoint-half:midpoint+half+1]
                num_xpoints_i = len(xpoints_i)
            else:
                x_len_i = 1 + numx + 2*L - y_point_i 
                half = int(x_len_i/2)
                xpoints_i = xpoints[midpoint-half:midpoint+half+1]
                num_xpoints_i = len(xpoints_i)

            all_x_points = np.concatenate((all_x_points,np.array(xpoints_i)),axis=0)
            all_y_points = np.concatenate((all_y_points,np.array([y_point]*num_xpoints_i)),axis=0)
        else:
            if y_point_i <= L:
                x_len_i = 1 + numx + y_point_i 
                half = int(x_len_i/2)
                xpoints_i = xpoints[midpoint-(half-1):midpoint+half+1]
                num_xpoints_i = len(xpoints_i)
            else:
                x_len_i = 1 + numx + 2*L-1 - y_point_i 
                half = int(x_len_i/2)
                xpoints_i = xpoints[midpoint-half:midpoint+half+2]
                num_xpoints_i = len(xpoints_i)

            all_x_points = np.concatenate((all_x_points,np.array(xpoints_i)+xscale/2),axis=0)
            all_y_points = np.concatenate((all_y_points,np.array([y_point]*num_xpoints_i)),axis=0)

    return all_x_points[1:],all_y_points[1:]


def calculate_tri_kspace_correlations(L: int, Sij: np.ndarray, kx_points: np.ndarray, ky_points: np.ndarray, 
                                      p1=default_p1, p2=default_p2,
                               periodic=False, reorder=False, boundary_size=0):
    
    L_bulk = L - (2*boundary_size)
    Sk = np.zeros((len(kx_points)), dtype=complex)
    interactions_r = buildlattice_alltoall_primitive_vector(L, p1, p2, periodic=periodic, reorder=reorder, boundary_size=boundary_size)

    for idx, r in interactions_r.items():
        Sk += np.exp(1j * (r[0] * kx_points + r[1] * ky_points)) * Sij[idx]
        if idx[0] != idx[1]:
            Sk += np.exp(-1j * (r[0] * kx_points + r[1] * ky_points)) * Sij[idx]

    Sk /= L_bulk ** 2

    return np.real(Sk)