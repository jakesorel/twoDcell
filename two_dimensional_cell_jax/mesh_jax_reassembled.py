"""
Overview
--------

x,R --> perioself domain extension (y,R_extended) (duplicating corners and edges, introducing ghost particles)
(y,R_extended) --> triangulation (tri)
triangulation --> neighbourhood

This triangulation contains 'real', 'duplicated', and 'ghost' instances
Proposal: 'forces' are calculated wrt real instances only (i.e. inselfes where tri<n_c)
Further: 'perimeters' etc need only be calculated for real instances.
"""


import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import grad as jgrad
from jax import grad, vmap

from jax import jit
import jax
import jax
from functools import partial
from scipy import optimize

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)
import time
import numba as nb
import two_dimensional_cell.power_triangulation as pt
import two_dimensional_cell.tri_functions as trf


class Mesh:
    def __init__(self,x,R,L,buffer=None):
        self.x = x
        self.R = R
        self.L = L
        if buffer is None:
            self.buffer = np.max(R)*3


    def triangulate(self):
        self.n_c = self.x.shape[0]
        self.y, self.n_C,self.idxs = extend_domain(self.x, self.L, self.R, self.n_c, self.corner_shifts, self.mid_shifts)
        self.y, self.n_C = add_boundary_points(self.y, self.n_C, self.boundary_points, 4)
        self.idxs = np.concatenate((self.idxs,(-1,-1,-1,-1)))
        self.R_extended = self.R.take(self.idxs)
        self.tri, self.vs, self.n_v = pt.get_power_triangulation(self.y,self.R_extended)
        self.tri_per = trf.tri_call(self.idxs,self.tri)
        self.neigh = trf.get_neighbours(self.tri)
        self.neighm1 = self.neigh == - 1
        self.non_ghost_tri_mask = get_non_ghost_tri_mask(self.tri, self.n_c)
        self.non_ghost_tri_idx = np.array(np.nonzero(self.non_ghost_tri_mask)).T
    def tri_format(self):
        self.tx = trf.tri_call3(self.y, self.tri)
        self.tR = trf.tri_call(self.R_extended,self.tri)

class ReducedMesh:
    def __init__(self,tx,tR):
        self.tx,self.tR = tx,tR



        # self.rj = trf.roll3(self.tx,1)
        # self.rk = trf.roll3(self.tx,-1)
        # self.Rj = trf.roll(self.tR,1)
        # self.Rk = trf.roll(self.tR,-1)
        # self.vn = trf.tri_call3(self.vs, self.neigh)
        # self.vp1 = trf.roll3(self.vn, 1)
        # self.vm1 = trf.roll3(self.vn, -1)

@jit
def _get_circumcenter_i(txi,tRi):
    (xi,yi),(xj,yj),(xk,yk) = txi
    Ri,Rj,Rk = tRi
    denom = (2. * (xk * (-yi + yj) + xj * (yi - yk) + xi * (-yj + yk)))
    vx = (xj ** 2 * yi - xk ** 2 * yi + Rk ** 2 * (yi - yj) + xk ** 2 * yj - yi ** 2 * yj + yi * yj ** 2 + (
                    Ri - xi) * (Ri + xi) * (yj - yk) - (xj ** 2 - yi ** 2 + yj ** 2) * yk + (
                            -yi + yj) * yk ** 2 + Rj ** 2 * (-yi + yk))
    vy = (xi ** 2 * xj - xi * xj ** 2 + Rk ** 2 * (-xi + xj) + Rj ** 2 * (
                    xi - xk) - xi ** 2 * xk + xj ** 2 * xk + xi * xk ** 2 - xj * xk ** 2 + Ri ** 2 * (
                            -xj + xk) + xj * yi ** 2 - xk * yi ** 2 - xi * yj ** 2 + xk * yj ** 2 + (
                            xi - xj) * yk ** 2)
    return jnp.array((vx,vy))/denom

@jit
def _get_circumcenter(tx,tR):
    return vmap(_get_circumcenter_i)(tx,tR)

@jit
def _tri_format(self):
    self["vs"] = _get_circumcenter(self["tx"],self["tR"])
    self["ri"] = self["tx"]
    self["rj"] = jnp.roll(self["tx"], -1,1)
    self["rk"] = jnp.roll(self["tx"], 1,1)
    self["Ri"] = self["tR"]
    self["Rj"] = jnp.roll(self["tR"], -1,1)
    self["Rk"] = jnp.roll(self["tR"], 1,1)
    self["vn"] = self["vs"][self["neigh"]] ##BEWARE THERE ARE -1s in the neighbourhood matrix.
    self["vn"] = self["vn"]*1/jnp.expand_dims(self["neigh"]>-1,2) ##CAN REMOVE THIS LATER. TO CATCH EXCEPTIONS
    self["vp1"] = jnp.roll(self["vn"], -1,1)
    self["vm1"] = jnp.roll(self["vn"], 1,1)
    return self

@jit
def disp23(x, y):
    return jnp.expand_dims(x, 1) - y

@jit
def disp33(x, y):
    return x - y

@jit
def tnorm(x):
    """
    Calculate the L1 norm of a set of vectors that are given in triangulated form:

    (nv x 3 x 2) ->> (nv x 3)
    :param x:
    :return:
    """
    return jnp.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)


@jit
def _get_displacements(self):
    self["v_x"] = disp23(self["vs"], self["tx"])
    self["lv_x"] = tnorm(self["v_x"])
    self["v_vp1"] = disp23(self["vs"], self["vp1"])
    self["lp1"] = tnorm(self["v_vp1"])
    self["v_vm1"] = disp23(self["vs"], self["vm1"])
    self["lm1"] = tnorm(self["v_vm1"])
    self["vp1_x"] = disp33(self["vp1"], self["tx"])
    self["lvp1_x"] = tnorm(self["vp1_x"])
    self["vm1_x"] = disp33(self["vm1"], self["tx"])
    self["lvm1_x"] = tnorm(self["vm1_x"])
    self["vp1_vm1"] = disp33(self["vp1"], self["vm1"])
    self["rik"] = self["ri"] - self["rk"]
    self["rij"] = self["ri"] - self["rj"]
    self["lrik"] = tnorm(self["rik"])
    self["lrij"] = tnorm(self["rij"])

    ##determine if all of these are needed at some point

    return self


@jit
def __classify_edges(lv_x, lvm1_x, tR, Rj,lrij):
    V_in = (lv_x < tR)
    V_out = (lvm1_x < tR)
    no_touch = (2 * (tR ** 2 + Rj ** 2) / lrij ** 2 - ((tR ** 2 - Rj ** 2) ** 2) / lrij ** 4 - 1) < 0
    return V_in, V_out, no_touch

@jit
def _classify_edges(self):
    self["V_in_j"], self["V_out_j"], self["no_touch_j"] = __classify_edges(self["lv_x"], self["lvm1_x"], self["tR"],self["Rj"], self["lrij"])
    self["V_in_k"], self["V_out_k"], self["no_touch_k"] = __classify_edges(self["lv_x"], self["lvp1_x"], self["tR"],self["Rk"], self["lrik"])
    return self

@jit
def _get_angles(self):
    vs3 = jnp.stack((self["vs"],self["vs"],self["vs"]),1)
    self["ttheta_j"], self["hm_j"], self["hp_j"] = _ttheta(self["ri"],self["rj"],
                                                           self["Ri"],self["Rj"],
                                                           self["lrij"],
                                                           self["V_in_j"],self["V_out_j"],
                                                           self["no_touch_j"],
                                                           vs3,self["vm1"])


    self["ttheta_k"], self["hm_k"], self["hp_k"] = _ttheta(self["ri"],self["rk"],
                                                           self["Ri"],self["Rk"],
                                                           self["lrik"],
                                                           self["V_in_k"],self["V_out_k"],
                                                           self["no_touch_k"],
                                                           self["vp1"],vs3)

    # self.classification_correction()
    self["tphi_j"] = _tvecangle(self["vm1_x"], self["v_x"])
    self["tphi_k"] = _tvecangle(self["v_x"], self["vp1_x"])

    self["tpsi_j"] = self["tphi_j"] - self["ttheta_j"]
    self["tpsi_k"] = self["tphi_k"] - self["ttheta_k"]
    return self

@jit
def _tintersections_i(ri, rj, Ri, Rj, nrij):
    # ri,rj,Ri,Rj,nrij = geom.tS,_roll3(geom.tS,1),geom.tR,_roll(geom.tR),geom.nrij
    a = 0.5 * (ri + rj) + jnp.expand_dims((Ri ** 2 - Rj ** 2) / (2 * nrij ** 2),1) * (rj - ri)
    b = jnp.expand_dims(0.5 * jnp.sqrt(2 * (Ri ** 2 + Rj ** 2) / nrij ** 2 - ((Ri ** 2 - Rj ** 2) ** 2) / nrij ** 4 - 1),1) * jnp.column_stack((rj[:,1] - ri[:,1], ri[:,0] - rj[:,0]))
    pos1 = a - b
    pos2 = a + b
    return pos1, pos2

@jit
def _tintersections(ri,rj,Ri,Rj,lrij):
    return jax.vmap(_tintersections_i)(ri,rj,Ri,Rj,lrij)



@jit
def _ttheta(ri,rj,Ri,Rj,lrij,V_in, V_out, no_touch, _hp_j,_hm_j):

    _V_in, _V_out = jnp.expand_dims(V_in,2),jnp.expand_dims(V_out,2)
    _touch = jnp.expand_dims(~no_touch,2)
    h_CCW, h_CW = _tintersections(ri, rj, Ri, Rj, lrij)
    hp_j = (_touch)*(_hp_j*_V_in + h_CCW*~_V_in)
    hm_j = (_touch)*(_hm_j*_V_out + h_CW*~_V_out)

    ttheta = _tvecangle(hm_j - ri, hp_j - ri)
    return ttheta, hm_j, hp_j

@jit
def _tvecangle(a, b):
    """
    Signed angle between two (triangle form) sets of vectors
    :param a:
    :param b:
    :return:
    """
    return jnp.arctan2(tcross(a, b), tdot(a, b))

@jit
def tcross(A, B):
    """
    Cross product of two triangulated vectors, each of shape nv x 3 x 2
    :param A:
    :param B:
    :return:
    """
    return A[:, :, 0] * B[:, :, 1] - A[:, :, 1] * B[:, :, 0]


@jit
def tdot(A, B):
    """
    Dot product of two triangulated vectors, each of shape nv x 3 x 2
    :param A:
    :param B:
    :return:
    """
    return A[:, :, 0] * B[:, :, 0] + A[:, :, 1] * B[:, :, 1]



@nb.jit(nopython=True)
def get_edge_masks(x, L, buffer):
    """
    Given a buffer distance (e.g. 2*radius), determine which cells lie in various regions within the buffer.

    On an edge (i.e. less than the length of the buffer from one of the edges of the domain)

    On a corner (i.e. within a corner box of the domain, of size (buffer x buffer))
    """
    rad_count_lb = (x / buffer).astype(np.int64)
    rad_count_rt = ((L - x) / buffer).astype(np.int64)
    on_edge_lb = rad_count_lb == 0
    on_edge_rt = rad_count_rt == 0
    on_edge = (on_edge_lb[:, 0], on_edge_rt[:, 0], on_edge_lb[:, 1], on_edge_rt[:, 1])
    on_corner_lb = on_edge_lb[:, 0] * on_edge_lb[:, 1]
    on_corner_rb = on_edge_rt[:, 0] * on_edge_lb[:, 1]
    on_corner_lt = on_edge_lb[:, 0] * on_edge_rt[:, 1]
    on_corner_rt = on_edge_rt[:, 0] * on_edge_rt[:, 1]
    on_corner = (on_corner_lb, on_corner_rb, on_corner_lt, on_corner_rt)
    on_boundary = on_edge_lb[:, 0] + on_edge_lb[:, 1] + on_edge_rt[:, 0] + on_edge_rt[:, 1]
    return on_boundary, on_edge, on_corner  # ,on_mid

@nb.jit(nopython=True)
def make_shifts(L):
    corner_shifts = np.array(((L, L),
                              (-L, L),
                              (L, -L),
                              (-L, -L)))
    mid_shifts = np.array(((L, 0),
                           (-L, 0),
                           (0, L),
                           (0, -L)))
    return corner_shifts,mid_shifts

@nb.jit(nopython=True)
def extend_domain(x, L, R, N, corner_shifts, mid_shifts):
    """
    Given the above categorization, duplicate cells such that an extended perioself domain is established.

    This enlarges the domain from (L x L) to ([L+2*buffer] x [L+2*buffer])

    New positions are 'y'.

    """
    x_x,x_y = x[:,0],x[:,1]

    radius = np.max(R)
    on_boundary, on_edge, on_corner = get_edge_masks(x, L, radius*3)
    n_on_edge = np.zeros((4), dtype=np.int64)
    n_on_corner = np.zeros((4), dtype=np.int64)
    for i in range(4):
        n_on_edge[i] = on_edge[i].sum()
        n_on_corner[i] = on_corner[i].sum()
    n_on_edge_all = n_on_edge.sum()
    n_on_corner_all = n_on_corner.sum()
    n_replicated = n_on_edge_all + n_on_corner_all
    y = np.zeros((N + n_replicated, 2), dtype=np.float64)
    idxs = np.zeros((N+n_replicated),dtype=np.int64)
    idxs[:N] = np.arange(N)
    y[:N] = x
    M = N
    if on_boundary.any():
        for i, (oc, shft) in enumerate(zip(on_corner, corner_shifts)):
            k = n_on_corner[i]
            if k > 0:
                y[M:M + k] = x[oc] + shft
                idx = np.nonzero(oc)[0]
                # y[M:M + k,0] = x_x.take(idx)+shft[0]
                # y[M:M + k,1] = x_y.take(idx)+shft[1]
                idxs[M:M+k] = idx
                M += k
        for i, (om, shft) in enumerate(zip(on_edge, mid_shifts)):
            k = n_on_edge[i]
            if k > 0:
                y[M:M + k] = x[om] + shft

                idx = np.nonzero(om)[0]
                # y[M:M + k,0] = x_x.take(idx)+shft[0]
                # y[M:M + k,1] = x_y.take(idx)+shft[1]
                idxs[M:M+k] = idx
                M += k
    return y, M,idxs

@nb.jit(nopython=True)
def make_boundary_points(L,buffer):
    mn,mx = -L-buffer,2*L+buffer
    boundary_points = np.array(((mn,mn),
                                (mn,mx),
                                (mx,mn),
                                (mx,mx)))
    return boundary_points

@nb.jit(nopython=True)
def add_boundary_points(y, M, boundary_points, n_boundary_points):
    """
    Boundary points are included to ensure that all edges within the true tissue are represented twice in the triangulation,

    once forward, once reverse.
    """
    z = np.zeros((M + n_boundary_points, 2), dtype=np.float64)
    z[:M] = y
    z[M:] = boundary_points
    return z, M + n_boundary_points

@nb.jit(nopython=True)
def get_non_ghost_tri_mask(tri, N):
    return tri < N

#######


msh = sim.t.mesh
tx,tR,neigh = msh.tx.copy(),msh.tR.copy(),msh.neigh.copy()
self = {"tx":tx,"tR":tR,"neigh":neigh}
self = _tri_format(self)
self = _get_displacements(self)
self = _classify_edges(self)
self = _get_angles(self)

print(np.nanmax(self["hm_j"]-msh.hm_j))

a = self["hp_k"]

"""
seems like hm_k/hp_k has issues
Don't know whether this is due to classification correction or not
Not sure what classification correction does. By memory, this is when there is some dodgy partitioning owing to squished triangulation
But can't rememember...  

UPDATE: 
Classification correction had a bug in the original code -- it did not update certain cell properties (theta, phi etc) 
Implement classification correction here and see if the values now match up. 
From some subsetting, it seems like this is the cause (i.e. no-touch matrices are not updated here). 


From my understanding, theoretically, classification correction deals with the following edge case: 
- In principle, there may be scenarios where the radius of one cell 


Also note: 
sim_TEST seems to run mostly fine. However, there have been cases where the radius suddenly springs up before returning down, suggesting some error in the (e.g. area) calculation 
Not sure whether this is an error in calculating the area etc due to some edge case, or an error in the jacobian calcualtion

"""