import triangle as tr
import twoDcell.tri_functions as trf
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
from two_dimensional_cell.cell_geometries import geometries

radius = 0.6
L = 9
init_noise = 1e-2
A0 = 1
x = trf.hexagonal_lattice(int(L), int(np.ceil(L)), noise=init_noise, A=A0)
x += 1e-3
np.argsort(x.max(axis=1))

x = x[np.argsort(x.max(axis=1))[:int(L ** 2 / A0)]]



@jit(nopython=True)
def get_edge_masks(x,L,buffer):
    """
    Given a buffer distance (e.g. 2*radius), determine which cells lie in various regions within the buffer.

    On an edge (i.e. less than the length of the buffer from one of the edges of the domain)

    On a corner (i.e. within a corner box of the domain, of size (buffer x buffer))
    """
    rad_count_lb = (x/buffer).astype(np.int64)
    rad_count_rt = ((L-x)/buffer).astype(np.int64)
    on_edge_lb = rad_count_lb == 0
    on_edge_rt = rad_count_rt == 0
    on_edge = (on_edge_lb[:,0],on_edge_rt[:,0],on_edge_lb[:,1],on_edge_rt[:,1])
    on_corner_lb = on_edge_lb[:,0]*on_edge_lb[:,1]
    on_corner_rb = on_edge_rt[:,0]*on_edge_lb[:,1]
    on_corner_lt = on_edge_lb[:,0]*on_edge_rt[:,1]
    on_corner_rt = on_edge_rt[:,0]*on_edge_rt[:,1]
    on_corner = (on_corner_lb,on_corner_rb,on_corner_lt,on_corner_rt)
    on_boundary = on_edge_lb[:,0] + on_edge_lb[:,1] + on_edge_rt[:,0] + on_edge_rt[:,1]
    return on_boundary,on_edge,on_corner#,on_mid

corner_shifts = np.array(((L,L),
                          (-L,L),
                          (L,-L),
                          (-L,-L)))
mid_shifts = np.array(((L,0),
                       (-L,0),
                       (0,L),
                       (0,-L)))


@jit(nopython=True)
def extend_domain(x,L,radius,N,corner_shifts,mid_shifts):
    """
    Given the above categorization, duplicate cells such that an extended periodic domain is established.

    This enlarges the domain from (L x L) to ([L+2*buffer] x [L+2*buffer])

    New positions are 'y'.

    """
    # x_x,x_y = x[:,0],x[:,1]

    on_boundary,on_edge,on_corner = get_edge_masks(x,L,radius)
    n_on_edge = np.zeros((4),dtype=np.int64)
    n_on_corner = np.zeros((4),dtype=np.int64)
    for i in range(4):
        n_on_edge[i] = on_edge[i].sum()
        n_on_corner[i] = on_corner[i].sum()
    n_on_edge_all = n_on_edge.sum()
    n_on_corner_all = n_on_corner.sum()
    n_replicated = n_on_edge_all + n_on_corner_all
    y = np.zeros((N+n_replicated,2),dtype=np.float64)
    y[:N] = x
    M = N
    if on_boundary.any():
        for i, (oc,shft) in enumerate(zip(on_corner,corner_shifts)):
            k = n_on_corner[i]
            if k>0:
                y[M:M + k] = x[oc] + shft
                # idx = np.nonzero(oc)[0]
                # y[N+n_edge:N+n_edge+k,0] = x_x.take(idx)+shft[0]
                # y[N+n_edge:N+n_edge+k,1] = x_y.take(idx)+shft[1]

                M += k
        for i, (om,shft) in enumerate(zip(on_edge,mid_shifts)):
            k = n_on_edge[i]
            if k>0:
                y[M:M + k] = x[om] + shft

                # idx = np.nonzero(om)[0]
                # y[N+n_edge:N+n_edge+k,0] = x_x.take(idx)+shft[0]
                # y[N+n_edge:N+n_edge+k,1] = x_y.take(idx)+shft[1]
                M += k
    return y,M

buffer = 2*radius
boundary_points = np.array(((L+buffer,0),
                       (-L-buffer,0),
                       (0,L+buffer),
                       (0,-L-buffer)))

@jit(nopython=True)
def add_boundary_points(y,M,boundary_points,n_boundary_points):
    """
    Boundary points are included to ensure that all edges within the true tissue are represented twice in the triangulation,

    once forward, once reverse.
    """
    z = np.zeros((M+n_boundary_points,2),dtype=np.float64)
    z[:M] = y
    z[M:] = boundary_points
    return z, M+n_boundary_points

@jit(nopython=True)
def get_non_ghost_tri_mask(tri,N):
    return tri < N


N = x.shape[0]
y, M = extend_domain(x,L,radius,N,corner_shifts,mid_shifts)
y,M = add_boundary_points(y,M,boundary_points,4)
triangulation = tr.triangulate({"vertices":y})
tri = triangulation["triangles"]
non_ghost_tri_mask = get_non_ghost_tri_mask(tri,N)

t0 = time.time()
for i in range(int(4000)):
    y, M = extend_domain(x, L, radius, N, corner_shifts, mid_shifts)
    y, M = add_boundary_points(y, M, boundary_points, 4)
    triangulation = tr.triangulate({"vertices": y})
    tri = triangulation["triangles"]
    non_ghost_tri_mask = get_non_ghost_tri_mask(tri, N)
t1= time.time()

print(t1-t0)

"""
Now how to use it: 

The principle here is that duplicated cells act as 'ghosts'. 
Thus, one can calculate the forces acting on the real (and ghost) cells. Then just ignore the ghost ones. 



"""

# geom = geometries(y, np.ones((M))*radius, V, tri_list, n_v, n_b)