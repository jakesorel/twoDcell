import _pickle as cPickle
import bz2
import pickle

import numpy as np
import triangle as tr
from numba import jit
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist

import triangle as tr
import two_dimensional_cell.tri_functions as trf
import two_dimensional_cell.power_triangulation as pt
import numpy as np

from numba import jit
import matplotlib.pyplot as plt
import time

class Mesh:
    def __init__(self,x,R,L,buffer=None,tri=None,fill=True, id=None, name=None, load=None, run_options=None):
        assert run_options is not None, "Specify run options"

        if id is None:
            self.id = {}
        else:
            self.id = id
        self.run_options = run_options
        self.name = name

        self.x = x
        self.R = R
        self.L = L
        if buffer is None:
            self.buffer = np.max(R)*2
        self.corner_shifts,self.mid_shifts = make_shifts(self.L)
        self.boundary_points = make_boundary_points(self.L,self.buffer)


        if load is not None:
            self.load(load)
        elif tri is not None:
            self.tri = tri
            self.n_v = self.tri.shape[0]
            self.neigh = trf.get_neighbours(self.tri)
            self.update_from_tri()
        elif fill:
            self.n_c = self.x.shape[0]
            self.update_x(self.x)

    def triangulate(self):
        self.n_c = self.x.shape[0]
        self.y, self.n_C,self.idxs = extend_domain(self.x, self.L, self.R, self.n_c, self.corner_shifts, self.mid_shifts)
        self.y, self.n_C = add_boundary_points(self.y, self.n_C, self.boundary_points, 4)
        self.idxs = np.concatenate((self.idxs,(-1,-1,-1,-1)))
        self.R_extended = self.R.take(self.idxs)
        self.tri, self.vs, self.n_v = pt.get_power_triangulation(self.y,self.R_extended)
        # triangulation = tr.triangulate({"vertices": self.y},"n")
        # self.tri = triangulation["triangles"]
        self.tri_per = trf.tri_call(self.idxs,self.tri)
        # self.neigh = triangulation["neighbors"]
        self.neigh = trf.get_neighbours(self.tri)
        self.neighm1 = self.neigh == - 1
        self.non_ghost_tri_mask = get_non_ghost_tri_mask(self.tri, self.n_c)

    def update(self):
        self.triangulate()
        self.tri_format()
        self.get_displacements()
        self.classify_edges()
        self.get_angles()
        self.classification_correction()
        self.get_touch_mats()

        self.get_circle_intersect_distances()
        self.get_A()
        self.get_P()
        self.get_l_interface()

    def update_x(self, x):
        self.x = x
        self.triangulate()
        self.tri_format()
        self.get_displacements()
        self.classify_edges()
        self.get_angles()
        self.classification_correction()

        self.get_touch_mats()
        self.get_circle_intersect_distances()
        self.get_A()
        self.get_P()
        self.get_l_interface()

    def update_from_tri(self):
        self.tri_format()
        self.get_displacements()
        self.classify_edges()
        self.get_angles()
        self.classification_correction()

        self.get_touch_mats()

        self.get_circle_intersect_distances()
        self.get_A()
        self.get_P()
        self.get_l_interface()

    def save(self, name, id=None, dir_path="", compressed=False):
        self.name = name
        if id is None:
            self.id = {}
        else:
            self.id = id
        if compressed:
            with bz2.BZ2File(dir_path + "/" + self.name + "_mesh" + '.pbz2', 'w') as f:
                cPickle.dump(self.__dict__, f)
        else:
            pikd = open(dir_path + "/" + self.name + "_mesh" + '.pickle', 'wb')
            pickle.dump(self.__dict__, pikd)
            pikd.close()

    def load(self, fname):
        if fname.split(".")[1] == "pbz2":
            fdict = cPickle.load(bz2.BZ2File(fname, 'rb'))
        else:
            pikd = open(fname, 'rb')
            fdict = pickle.load(pikd)
            pikd.close()
        if (self.run_options != fdict["run_options"]) and (self.run_options is not None):
            print("Specified run options do not match those from the loaded file. Proceeding...")
        self.__dict__ = fdict

    def get_vertices(self):
        """
        Get vertex locations, given cell centroid positions and triangulation. I.e. calculate the circumcentres of
        each triangle

        :return V: Vertex coordinates (nv x 2)
        """
        V = trf.circumcenter(self.tx)
        return V

    def tri_format(self):
        self.tx = trf.tri_call3(self.y, self.tri)
        self.rj = trf.roll3(self.tx,1)
        self.rk = trf.roll3(self.tx,-1)
        self.tR = trf.tri_call(self.R_extended,self.tri)
        # self.tR = np.ones((self.tx.shape[0],self.tx.shape[1]))*self.radius ##for now, enforce all radii are the same
        self.Rj = trf.roll(self.tR,1)
        self.Rk = trf.roll(self.tR,-1)
        # self.vs = self.get_vertices()
        self.vs3 = trf.triplicate(self.vs)
        self.vn = trf.tri_call3(self.vs, self.neigh)
        self.vp1 = trf.roll3(self.vn, 1)
        self.vm1 = trf.roll3(self.vn, -1)

    def get_displacements(self):
        self.v_x = disp23(self.vs, self.tx)
        self.lv_x = trf.tnorm(self.v_x)
        self.v_vp1 = disp23(self.vs, self.vp1)
        self.lp1 = trf.tnorm(self.v_vp1)
        self.v_vm1 = disp23(self.vs, self.vm1)
        self.lm1 = trf.tnorm(self.v_vm1)
        self.vp1_x = disp33(self.vp1, self.tx)
        self.lvp1_x = trf.tnorm(self.vp1_x)
        self.vm1_x = disp33(self.vm1, self.tx)
        self.lvm1_x = trf.tnorm(self.vm1_x)
        self.vp1_vm1 = disp33(self.vp1, self.vm1)

        self.rik = get_rik(self.tx)
        self.rij = get_rij(self.tx)
        self.lrik = trf.tnorm(self.rik)
        self.lrij = trf.tnorm(self.rij)

        # self.dij = get_dij(self.lrij, self.radius)
        # self.dik = get_dij(self.lrik, self.radius)

    def classify_edges(self):
        self.V_in_j, self.V_out_j, self.no_touch_j = _classify_edges(self.lv_x, self.lvm1_x, self.tR,self.Rj, self.lrij)
        self.V_in_k, self.V_out_k, self.no_touch_k = _classify_edges(self.lv_x, self.lvp1_x, self.tR,self.Rk, self.lrik)

    def get_angles(self):
        self.ttheta_j, self.hm_j, self.hp_j = _ttheta(self.V_in_j, self.V_out_j, self.no_touch_j, self.tR,
                                                      self.tx,
                                                      self.vs3,
                                                      self.lrij, self.vm1, dir=1)

        self.ttheta_k, self.hm_k, self.hp_k = _ttheta(self.V_in_k, self.V_out_k, self.no_touch_k, self.tR,
                                                      self.tx,
                                                      self.vp1,
                                                      self.lrik, self.vs3,  dir=-1)

        # self.classification_correction()
        self.tphi_j = _tvecangle(self.vm1_x, self.v_x)
        self.tphi_k = _tvecangle(self.v_x, self.vp1_x)

        self.tpsi_j = self.tphi_j - self.ttheta_j
        self.tpsi_k = self.tphi_k - self.ttheta_k

    def classification_correction(self):
        self.hp_j, self.hm_j, self.ttheta_j, self.no_touch_j = do_classification_correction(self.x, self.R,
                                                                                            self.ttheta_j,
                                                                                            self.hm_j,
                                                                                            self.hp_j,
                                                                                            self.no_touch_j,
                                                                                            self.V_in_j)
        self.hp_k, self.hm_k, self.ttheta_k, self.no_touch_k = do_classification_correction(self.x, self.R,
                                                                                            self.ttheta_k,
                                                                                            self.hm_k,
                                                                                            self.hp_k,
                                                                                            self.no_touch_k,
                                                                                            self.V_in_k)

    def get_touch_mats(self):

        self.no_touch_j_mat = trf.repeat_mat(self.no_touch_j)
        self.no_touch_j_vec = trf.repeat_vec(self.no_touch_j)
        self.no_touch_k_mat = trf.repeat_mat(self.no_touch_k)
        self.no_touch_k_vec = trf.repeat_vec(self.no_touch_k)

    def get_circle_intersect_distances(self):
        self.rihm_j, self.rihp_j = -disp33(self.hm_j, self.tx), -disp33(self.hp_j, self.tx)
        self.rihm_k, self.rihp_k = -disp33(self.hm_k, self.tx), -disp33(self.hp_k, self.tx)
        self.nrihp_j, self.nrihp_k, self.nrihm_j, self.nrihm_k = trf.tnorm(self.rihp_j), trf.tnorm(
            self.rihp_k), trf.tnorm(self.rihm_j), trf.tnorm(self.rihm_k)

        self.hmj_hpj = disp33(self.hm_j, self.hp_j)
        self.hmk_hpk = disp33(self.hm_k, self.hp_k)
        self.nhmj_hpj = trf.tnorm(self.hmj_hpj)
        self.nhmk_hpk = trf.tnorm(self.hmk_hpk)

    def get_P(self):
        self.tlP, self.tlC = get_lP(self.hp_j, self.hm_j), get_lC(self.tpsi_j, self.tR)
        # self.tP = self.tlP + self.tlC
        self._LP = trf.assemble_tri(self.tlP, self.tri)
        self._LC = trf.assemble_tri(self.tlC, self.tri)
        self.LP = self._LP[:self.n_c]
        self.LC = self._LC[:self.n_c]
        # self._P = self._LP + self._LC
        self.P = self.LP + self.LC

    def get_A(self):
        self.tAP, self.tAC = get_AP(self.hm_j, self.hp_j, self.tx), get_AC(self.tpsi_j, self.tR)
        self.tA = self.tAP + self.tAC
        self._A = trf.assemble_tri(self.tA, self.tri)
        self.A = self._A[:self.n_c]


    def get_l_interface(self):
        """
        A matrix of interface lengths between pairs of cells (contacting interfaces only).
        """
        ##This needs to be adjusted for periodic.
        self.l_int = None#coo_matrix((self.tlP.ravel(), (self.tri.ravel(), trf.roll(self.tri, -1).ravel())))


    def save(self, name, id=None, dir_path="", compressed=False):
        self.name = name
        if id is None:
            self.id = {}
        else:
            self.id = id
        if compressed:
            with bz2.BZ2File(dir_path + "/" + self.name + "_mesh" + '.pbz2', 'w') as f:
                cPickle.dump(self.__dict__, f)
        else:
            pikd = open(dir_path + "/" + self.name + "_mesh" + '.pickle', 'wb')
            pickle.dump(self.__dict__, pikd)
            pikd.close()

    def load(self, fname):
        if fname.split(".")[1] == "pbz2":
            fdict = cPickle.load(bz2.BZ2File(fname, 'rb'))
        else:
            pikd = open(fname, 'rb')
            fdict = pickle.load(pikd)
            pikd.close()
        if (self.run_options != fdict["run_options"]) and (self.run_options is not None):
            print("Specified run options do not match those from the loaded file. Proceeding...")
        self.__dict__ = fdict

@jit(nopython=True)
def disp33(x, y):
    return x - y

@jit(nopython=True)
def disp23(x, y):
    return np.expand_dims(x, 1) - y

@jit(nopython=True)
def disp32(x, y):
    return x - np.expand_dims(y, 1)

@jit(nopython=True)
def get_k2(tri, neigh):
    """
    To determine whether a given neighbouring pair of triangles needs to be re-triangulated, one considers the sum of
    the pair angles of the triangles associated with the cell centroids that are **not** themselves associated with the
    adjoining edge. I.e. these are the **opposite** angles.

    Given one cell centroid/angle in a given triangulation, k2 defines the column index of the cell centroid/angle in the **opposite** triangle

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: Neighbourhood matrix (n_v x 3) np.int32 array
    :return:
    """
    three = np.array([0, 1, 2])
    nv = tri.shape[0]
    k2s = np.empty((nv, 3), dtype=np.int32)
    for i in range(nv):
        for k in range(3):
            neighbour = neigh[i, k]
            k2 = ((neigh[neighbour] == i) * three).sum()
            k2s[i, k] = k2
    return k2s

@jit(nopython=True)
def _classify_edges(lv_x, lvm1_x, tR, Rj,lrij):
    """
    Classifies edges whether the
    """
    V_in = (lv_x < tR)
    V_out = (lvm1_x < tR)

    no_touch = (2 * (tR ** 2 + Rj ** 2) / lrij ** 2 - ((tR ** 2 - Rj ** 2) ** 2) / lrij ** 4 - 1) < 0


    return V_in, V_out, no_touch


@jit(nopython=True)
def get_rij(tx):
    """
    The displacement between the two centroids that flank the edge:

    v to vm1
    """
    return tx - trf.roll3(tx, 1)

@jit(nopython=True)
def get_rik(tx):
    """
    The displacement between the two centroids that flank the edge:

    v to vp1
    """
    return tx - trf.roll3(tx, -1)

@jit(nopython=True)
def get_dij(lrij, radius):
    return (lrij ** 2 + radius) / (2 * lrij)

# @jit(nopython=True)
# def _tintersections(ri, rj, radius, nrij):
#     ri, rj, nrij = ri.T, rj.T, nrij.T
#     a = 0.5 * (ri + rj)
#     b = 0.5 * np.sqrt(4 * (radius ** 2) / nrij ** 2 - 1) * np.stack((rj[1] - ri[1], ri[0] - rj[0]))
#     a, b = a.T, b.T
#     pos1 = a - b
#     pos2 = a + b
#     return pos1, pos2


@jit(nopython=True)
def _tintersections(ri, rj, Ri, Rj, nrij):
    # ri,rj,Ri,Rj,nrij = geom.tS,_roll3(geom.tS,1),geom.tR,_roll(geom.tR),geom.nrij
    ri, rj, Ri, Rj, nrij = ri.T, rj.T, Ri.T, Rj.T, nrij.T
    a = 0.5 * (ri + rj) + (Ri ** 2 - Rj ** 2) / (2 * nrij ** 2) * (rj - ri)
    b = 0.5 * np.sqrt(
        2 * (Ri ** 2 + Rj ** 2) / nrij ** 2 - ((Ri ** 2 - Rj ** 2) ** 2) / nrij ** 4 - 1) * np.stack(
        (rj[1] - ri[1], ri[0] - rj[0]))
    a, b = a.T, b.T
    pos1 = a - b
    pos2 = a + b
    return pos1, pos2


@jit(nopython=True)
def _ttheta(V_in, V_out, no_touch, tR, tS, tV, nrij, vj_neighbours, dir=1):
    """
    V_in, V_out, no_touch, tR, tS, tV, nrij, vj_neighbours = self.V_in_j, self.V_out_j, self.no_touch_j, self.radius,self.tx, self.vs3,self.lrij, self.vm1
    """
    V_in2, V_out2, no_touch2 = np.dstack((V_in, V_in)), np.dstack((V_out, V_out)), np.dstack((no_touch, no_touch))
    start, end = vj_neighbours.copy(), tV.copy()
    h_CCW, h_CW = _tintersections(tS, trf.roll3(tS, dir), tR, trf.roll(tR, dir), nrij)

    end = trf.replace_vec(end, ~V_in2, h_CCW)
    start = trf.replace_vec(start, ~V_out2, h_CW)
    end = trf.replace_val(end, no_touch2, 0)
    start = trf.replace_val(start, no_touch2, 0)

    ttheta = _tvecangle(start - tS, end - tS)
    return ttheta, start, end

@jit(nopython=True)
def _tvecangle(a, b):
    """
    Signed angle between two (triangle form) sets of vectors
    :param a:
    :param b:
    :return:
    """
    return np.arctan2(trf.tcross(a, b), trf.tdot(a, b))

@jit(nopython=True)
def numba_cdist(A, B):
    """
    Numba-ed version of scipy's cdist. for 2D only.
    with periodic bcs.
    """
    disp = np.expand_dims(A, 1) - np.expand_dims(B, 0)
    disp_x, disp_y = disp[..., 0], disp[..., 1]
    return disp_x ** 2 + disp_y ** 2

@jit(nopython=True)
def do_classification_correction(r, R, ttheta, hm, hp, no_touch, V_in_j, err=1e-7):
    """
    Deals with cases where two circles intersect within a 3rd cell

    LOOKS INCREDIBLY INEFFICIENT. Check whether it still works
    """

    touch_not_power_mask = (~V_in_j) * (~no_touch)
    if touch_not_power_mask.any():
        touch_not_power_mask_flat = touch_not_power_mask.ravel()
        hp_circ = np.column_stack(
            (hp[..., 0].ravel()[touch_not_power_mask_flat], hp[..., 1].ravel()[touch_not_power_mask_flat]))
        d = numba_cdist(hp_circ, r) - R ** 2
        false_cross = np.count_nonzero(d <= err, axis=1) > 2

        touch_not_power_mask_flat[touch_not_power_mask_flat] = false_cross
        touch_not_power_mask = touch_not_power_mask_flat.reshape(touch_not_power_mask.shape)

        no_touch = trf.replace_val(no_touch, touch_not_power_mask, True)
        no_touch_vec = np.dstack((no_touch, no_touch))
        hp = trf.replace_val(hp, no_touch_vec, 0)
        hm = trf.replace_val(hm, no_touch_vec, 0)
        ttheta = trf.replace_val(ttheta, no_touch, 0)

    return hp, hm, ttheta, no_touch

@jit(nopython=True)
def get_lP(start, end):
    return trf.tnorm(end - start)

@jit(nopython=True)
def get_lC(tpsi, tR):
    return tpsi * tR

@jit(nopython=True)
def get_AP(start, end, tx):
    return 0.5 * trf.tcross(start - tx, end - tx)

@jit(nopython=True)
def get_AC(tspi, tR):
    return 0.5 * tspi * tR ** 2


@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
def extend_domain(x, L, R, N, corner_shifts, mid_shifts):
    """
    Given the above categorization, duplicate cells such that an extended periodic domain is established.

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

@jit(nopython=True)
def make_boundary_points(L,buffer):
    mn,mx = -L-buffer,2*L+buffer
    boundary_points = np.array(((mn,mn),
                                (mn,mx),
                                (mx,mn),
                                (mx,mx)))
    return boundary_points

@jit(nopython=True)
def add_boundary_points(y, M, boundary_points, n_boundary_points):
    """
    Boundary points are included to ensure that all edges within the true tissue are represented twice in the triangulation,

    once forward, once reverse.
    """
    z = np.zeros((M + n_boundary_points, 2), dtype=np.float64)
    z[:M] = y
    z[M:] = boundary_points
    return z, M + n_boundary_points

@jit(nopython=True)
def get_non_ghost_tri_mask(tri, N):
    return tri < N

