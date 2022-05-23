import _pickle as cPickle
import bz2
import pickle

import numpy as np
import triangle as tr
from numba import jit
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist

import twoDcell.periodic_functions as per
import twoDcell.tri_functions as trf


class Mesh:
    """
    Mesh class
    ----------

    Deals with triangulation of a set of points, and calculates relevant geometries for use in force calculations.

    Triangulation algorithm takes some options, which can be tweaked for efficiency. Notably equiangulation.
    """

    def __init__(self, x=None, L=None,radius=None, tri=None,fill=True, id=None, name=None, load=None, run_options=None):
        assert run_options is not None, "Specify run options"

        if id is None:
            self.id = {}
        else:
            self.id = id
        self.run_options = run_options
        self.name = name
        self.x = x
        self.L = L
        self.radius = radius
        self.n_c = []
        self.n_v = []
        self.vs = []
        self.tri = []
        self.neigh = []
        self.k2s = []
        self.tx = []
        self.vs = []
        self.vn = []
        self.vp1 = []
        self.vm1 = []
        self.v_x = []
        self.lv_x = []
        self.v_vp1 = []
        self.lp1 = []
        self.v_vm1 = []
        self.lm1 = []
        self.vp1_x = []
        self.vm1_x = []
        self.vp1_vm1 = []
        self.A = []
        self.P = []
        self.A_components = []
        self.l_int = []

        self.grid_x, self.grid_y = np.mgrid[-1:2, -1:2]
        self.grid_x[0, 0], self.grid_x[1, 1] = self.grid_x[1, 1], self.grid_x[0, 0]
        self.grid_y[0, 0], self.grid_y[1, 1] = self.grid_y[1, 1], self.grid_y[0, 0]
        self.grid_xy = np.array([self.grid_x.ravel(), self.grid_y.ravel()]).T

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
        V = trf.circumcenter(self.tx, self.L)
        return V

    def _triangulate(self):
        """
        Calculates the periodic triangulation on the set of points x.

        Stores:
            self.n_v = number of vertices (int32)
            self.tri = triangulation of the vertices (nv x 3) matrix.
                Cells are stored in CCW order. As a convention, the first entry has the smallest cell id
                (Which entry comes first is, in and of itself, arbitrary, but is utilised elsewhere)
            self.vs = coordinates of each vertex; (nv x 2) matrix
            self.neigh = vertex ids (i.e. rows of self.vs) corresponding to the 3 neighbours of a given vertex (nv x 3).
                In CCW order, where vertex i {i=0..2} is opposite cell i in the corresponding row of self.tri
            self.neighbours = coordinates of each neighbouring vertex (nv x 3 x 2) matrix

        :param x: (nc x 2) matrix with the coordinates of each cell
        """

        # 1. Tile cell positions 9-fold to perform the periodic triangulation
        #   Calculates y from x. y is (9nc x 2) matrix, where the first (nc x 2) are the "true" cell positions,
        #   and the rest are translations
        y = trf.make_y(self.x, self.L * self.grid_xy)

        # 2. Perform the triangulation on y
        #   The **triangle** package (tr) returns a dictionary, containing the triangulation.
        #   This triangulation is extracted and saved as tri
        t = tr.triangulate({"vertices": y})
        tri = t["triangles"]

        # Del = Delaunay(y)
        # tri = Del.simplices
        n_c = self.x.shape[0]

        # 3. Find triangles with **at least one** cell within the "true" frame (i.e. with **at least one** "normal cell")
        #   (Ignore entries with -1, a quirk of the **triangle** package, which denotes boundary triangles
        #   Generate a mask -- one_in -- that considers such triangles
        #   Save the new triangulation by applying the mask -- new_tri
        tri = tri[(tri != -1).all(axis=1)]
        one_in = (tri < n_c).any(axis=1)
        new_tri = tri[one_in]

        # 4. Remove repeats in new_tri
        #   new_tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
        #   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
        #   cell ids, which are transformed by the mod function to account for periodicity. See function for more details
        n_tri = trf.remove_repeats(new_tri, n_c)

        # tri_same = (self.tri == n_tri).all()

        # 6. Store outputs
        self.n_v = n_tri.shape[0]
        self.tri = n_tri
        self.neigh = trf.get_neighbours(n_tri)

    def triangulate(self):
        if type(self.k2s) is list or not self.run_options["equiangulate"]:
            self._triangulate()
            self.k2s = get_k2(self.tri, self.neigh)
        else:
            tri, neigh, k2s, failed = re_triangulate(self.x, self.tri, self.neigh, self.k2s, self.tx, self.L, self.n_v, self.vs,max_runs=self.run_options["equi_nkill"])
            if failed:
                self._triangulate()
                self.k2s = get_k2(self.tri, self.neigh)
            else:
                self.tri, self.neigh, self.k2s = tri,neigh,k2s


    def tri_format(self):
        self.tx = trf.tri_call3(self.x, self.tri)
        self.vs = self.get_vertices()
        self.vs3 = trf.triplicate(self.vs)
        self.vn = trf.tri_call3(self.vs, self.neigh)
        self.vp1 = trf.roll3(self.vn,1)
        self.vm1 = trf.roll3(self.vn, -1)


    def get_displacements(self):
        self.v_x = disp23(self.vs, self.tx, self.L)
        self.lv_x = trf.tnorm(self.v_x)
        self.v_vp1 = disp23(self.vs, self.vp1, self.L)
        self.lp1 = trf.tnorm(self.v_vp1)
        self.v_vm1 = disp23(self.vs, self.vm1, self.L)
        self.lm1 = trf.tnorm(self.v_vm1)
        self.vp1_x = disp33(self.vp1, self.tx, self.L)
        self.lvp1_x = trf.tnorm(self.vp1_x)
        self.vm1_x = disp33(self.vm1, self.tx, self.L)
        self.lvm1_x = trf.tnorm(self.vm1_x)
        self.vp1_vm1 = disp33(self.vp1, self.vm1, self.L)

        self.rik = get_rik(self.tx,self.L)
        self.rij = get_rij(self.tx,self.L)
        self.lrik = trf.tnorm(self.rik)
        self.lrij = trf.tnorm(self.rij)

        self.dij = get_dij(self.lrij,self.radius)
        self.dik = get_dij(self.lrik,self.radius)

    def classify_edges(self):
        self.V_in_j, self.V_out_j, self.no_touch_j = _classify_edges(self.lv_x, self.lvm1_x, self.radius, self.lrij)
        self.V_in_k, self.V_out_k, self.no_touch_k = _classify_edges(self.lv_x, self.lvp1_x, self.radius, self.lrik)

    def get_angles(self):
        self.ttheta_j, self.hm_j, self.hp_j = _ttheta(self.V_in_j, self.V_out_j, self.no_touch_j, self.radius, self.tx,
                                                      self.vs3,
                                                      self.lrij, self.vm1,self.L,dir=1)
        self.ttheta_k, self.hm_k, self.hp_k = _ttheta(self.V_in_k, self.V_out_k, self.no_touch_k, self.radius, self.tx,
                                                      self.vp1,
                                                      self.lrik, self.vs3, self.L,dir=-1)

        # self.classification_correction()
        self.tphi_j = _tvecangle(self.vm1_x, self.v_x)
        self.tphi_k = _tvecangle(self.v_x, self.vp1_x)

        self.tpsi_j = self.tphi_j - self.ttheta_j
        self.tpsi_k = self.tphi_k - self.ttheta_k

    def classification_correction(self):
        self.hp_j,self.hm_j,self.ttheta_j,self.no_touch_j = do_classification_correction(self.x, self.radius,
                                                                                       self.ttheta_j, self.hm_j,
                                                                                       self.hp_j, self.no_touch_j,
                                                                                       self.V_in_j,self.L)
        self.hp_k,self.hm_k,self.ttheta_k,self.no_touch_k = do_classification_correction(self.x,self.radius,
                                                                                       self.ttheta_k, self.hm_k,
                                                                                       self.hp_k,self.no_touch_k,
                                                                                       self.V_in_k,self.L)


    def get_touch_mats(self):

        self.no_touch_j_mat = trf.repeat_mat(self.no_touch_j)
        self.no_touch_j_vec = trf.repeat_vec(self.no_touch_j)
        self.no_touch_k_mat = trf.repeat_mat(self.no_touch_k)
        self.no_touch_k_vec = trf.repeat_vec(self.no_touch_k)

    def get_circle_intersect_distances(self):
        self.rihm_j, self.rihp_j = -disp33(self.hm_j, self.tx,self.L), -disp33(self.hp_j, self.tx,self.L)
        self.rihm_k, self.rihp_k = -disp33(self.hm_k, self.tx,self.L), -disp33(self.hp_k, self.tx,self.L)
        self.nrihp_j, self.nrihp_k, self.nrihm_j, self.nrihm_k = trf.tnorm(self.rihp_j), trf.tnorm(self.rihp_k), trf.tnorm(self.rihm_j), trf.tnorm(self.rihm_k)

        self.hmj_hpj = disp33(self.hm_j, self.hp_j,self.L)
        self.hmk_hpk = disp33(self.hm_k, self.hp_k,self.L)
        self.nhmj_hpj = trf.tnorm(self.hmj_hpj)
        self.nhmk_hpk = trf.tnorm(self.hmk_hpk)

    def get_P(self):
        self.tlP, self.tlC = get_lP(self.hp_j, self.hm_j,self.L), get_lC(self.tpsi_j, self.radius)
        self.tP = self.tlP + self.tlC
        self.P = trf.assemble_tri(self.tP, self.tri)


    def get_A(self):
        self.tAP, self.tAC = get_AP(self.hm_j, self.hp_j, self.tx,self.L), get_AC(self.tpsi_j, self.radius)
        self.tA = self.tAP + self.tAC
        self.A = trf.assemble_tri(self.tA,self.tri)


    def get_l_interface(self):
        """
        A matrix of interface lengths between pairs of cells (contacting interfaces only).
        """
        self.l_int = coo_matrix((self.tlP.ravel(), (self.tri.ravel(), trf.roll(self.tri, -1).ravel())))


@jit(nopython=True)
def disp33(x, y, L):
    return per.per3(x - y, L, L)


@jit(nopython=True)
def disp23(x, y, L):
    return per.per3(np.expand_dims(x, 1) - y, L, L)


@jit(nopython=True)
def disp32(x, y, L):
    return per.per3(x - np.expand_dims(y, 1), L, L)


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
#
# @jit(nopython=True)
# def get_retriangulation_mask(angles,neigh,k2s,ntri):
#     neigh_angles = angles.take(neigh.ravel()*3 + k2s.ravel()).reshape(ntri,3)
#     mask = ((neigh_angles + angles) > np.pi)
#     return mask
#

@jit(nopython=True)
def get_retriangulation_mask(x,tri,lv_x,neigh,k2s,ntri,vs,L):
    d_cell = tri.take(neigh*3 + k2s).reshape(ntri,3)
    # rad_0 = per.per(tx[:,1] - vs,L,L)
    # rad_0 = np.sqrt(rad_0[:,0]**2 + rad_0[:,1]**2)
    xd = trf.tri_call3(x,d_cell)
    rad_d = per.per3(xd - np.expand_dims(vs,1),L,L)
    rad_d = np.sqrt(rad_d[...,0]**2 + rad_d[...,1]**2)
    mask = rad_d < np.expand_dims(lv_x,1)
    return mask



@jit(nopython=True)
def get_first_nonzero(flat_mask):
    i = 0
    while ~flat_mask[i]:
        i+=1
    return i


@jit(nopython=True)
def get_any_nonzero(flat_mask):
    i = int(np.random.random()*flat_mask.size)
    while ~flat_mask[i]:
        i = int(np.random.random()*flat_mask.size)
    return i




@jit(nopython=True)
def re_triangulate(x,_tri,_neigh,_k2s,tx0,L,ntri,vs0,max_runs=10):
    tri, neigh, k2s = _tri.copy(), _neigh.copy(), _k2s.copy()
    # lv_x = trf.tnorm(disp23(vs0, tx0, L))
    v_x = per.per(vs0-tx0[:,0],L,L)
    lv_x = np.sqrt(v_x[...,0]**2 + v_x[...,1]**2)

    mask = get_retriangulation_mask(x, tri, lv_x, neigh, k2s, ntri, vs0,L)
    continue_loop = mask.any()
    failed = False
    n_runs = 0
    if continue_loop:
        tx = tx0.copy()
        vs = vs0.copy()
        while (continue_loop):
            mask_flat = mask.ravel()
            q = get_first_nonzero(mask_flat)
            tri_0i, tri_0j = q//3,q%3
            quartet_info = get_quartet(tri,neigh,k2s,tri_0i,tri_0j)
            tri,neigh,k2s = update_mesh(quartet_info, tri, neigh, k2s)
            tx = tri_update(tx,quartet_info)

            tri_0i,tri_1i = quartet_info[0],quartet_info[2]
            tx_changed = np.stack((tx[tri_0i],tx[tri_1i]))
            vs_changed = trf.circumcenter(tx_changed,L)
            vs[tri_0i],vs[tri_1i] = vs_changed
            v_x_changed = per.per(vs_changed - tx_changed[:, 0], L, L)
            lv_x_changed = np.sqrt(v_x_changed[..., 0] ** 2 + v_x_changed[..., 1] ** 2)
            # lv_x_changed = trf.tnorm(disp23(vs_changed, tx_changed, L))
            lv_x[tri_0i],lv_x[tri_1i] = lv_x_changed
            mask = get_retriangulation_mask(x, tri, lv_x, neigh, k2s, ntri, vs, L)
            if n_runs > max_runs:
                failed = True
                continue_loop = False
            if not mask.any():
                continue_loop = False
            n_runs += 1
    return tri,neigh,k2s,failed
#
#
# @jit(nopython=True)
# def re_triangulate(x,_tri,_neigh,_k2s,L,ntri):
#     tri,neigh,k2s = _tri.copy(),_neigh.copy(),_k2s.copy()
#     angles = trf.tri_angles_periodic(x, tri, L)
#     neigh_angles = angles.take(neigh.ravel()*3 + k2s.ravel()).reshape(ntri,3)
#     interior_angles = neigh_angles + angles
#     mask = get_retriangulation_mask(angles,neigh,k2s,ntri)
#     n_runs = 0
#     continue_loop = mask.any()
#     failed = False
#     while (continue_loop):
#         mask_flat = mask.ravel()
#         q = get_any_nonzero(mask_flat)
#         # q = np.argmax(interior_angle.ravel())
#         tri_0i, tri_0j = q//3,q%3
#         quartet_info = get_quartet(tri,neigh,k2s,tri_0i,tri_0j)
#         tri_new, neigh_new, k2s_new = update_mesh(quartet_info, tri, neigh, k2s)
#         # trin, neighn, k2sn = update_mesh(quartet_info, tri, neigh, k2s)
#
#         angles_new = trf.tri_angles_periodic(x, tri_new, L)
#         neigh_angles_new = angles_new.take(neigh_new.ravel() * 3 + k2s_new.ravel()).reshape(ntri, 3)
#         interior_angles_new = neigh_angles_new + angles_new
#         mask_new = get_retriangulation_mask(angles_new, neigh_new, k2s_new, ntri)
#         if mask_new.sum()<mask.sum():
#         # if interior_angles_new.ravel()[mask_new.ravel()].sum()< interior_angles.ravel()[mask.ravel()].sum():
#             tri,neigh,k2s = tri_new,neigh_new,k2s_new
#             mask = mask_new
#             interior_angles = interior_angles_new.copy()
#         else:
#             failed = True
#             continue_loop = False
#         if not mask.any():
#             continue_loop = False
#         n_runs += 1
#     return tri,neigh,k2s,failed
#
# t0 = time.time()
# for i in range(int(1e4)):
#     re_triangulate(x, tri, neigh, k2s, L, ntri)
# t1= time.time()
# print(t1-t0)



@jit(nopython=True)
def get_quartet(tri,neigh,k2s,tri_0i,tri_0j):
    a,b,d = np.roll(tri[tri_0i],-tri_0j)
    tri_1i,tri_1j = neigh[tri_0i,tri_0j],k2s[tri_0i,tri_0j]
    c = tri[tri_1i,tri_1j]

    # quartet = np.array((a,b,c,d))

    tri0_da =(tri_0j+1)%3
    da_i = neigh[tri_0i,tri0_da]
    da_j = k2s[tri_0i,tri0_da]
    da = tri[da_i,da_j]

    tri0_ab =(tri_0j-1)%3
    ab_i = neigh[tri_0i,tri0_ab]
    ab_j = k2s[tri_0i,tri0_ab]
    ab = tri[ab_i,ab_j]


    tri1_cd =(tri_1j-1)%3
    cd_i = neigh[tri_1i,tri1_cd]
    cd_j = k2s[tri_1i,tri1_cd]
    cd = tri[cd_i,cd_j]

    tri1_bc =(tri_1j+1)%3
    bc_i = neigh[tri_1i,tri1_bc]
    bc_j = k2s[tri_1i,tri1_bc]
    bc = tri[bc_i,bc_j]

    return tri_0i,tri_0j,tri_1i,tri_1j,a,b,c,d,da,ab,bc,cd,da_i,ab_i,bc_i,cd_i,da_j,ab_j,bc_j,cd_j

@jit(nopython=True)
def tri_update(val,quartet_info):
    val_new = val.copy()
    tri_0i,tri_0j,tri_1i,tri_1j,a,b,c,d,da,ab,bc,cd,da_i,ab_i,bc_i,cd_i,da_j,ab_j,bc_j,cd_j = quartet_info
    val_new[tri_0i,(tri_0j-1)%3] = val[tri_1i,tri_1j]
    val_new[tri_1i,(tri_1j-1)%3] = val[tri_0i,tri_0j]
    return val_new




@jit(nopython=True)
def update_mesh(quartet_info,tri,neigh,k2s):
    """
    Update tri, neigh and k2. Inspect the equiangulation code for some inspo.
    :return:
    """

    tri_0i,tri_0j,tri_1i,tri_1j,a,b,c,d,da,ab,bc,cd,da_i,ab_i,bc_i,cd_i,da_j,ab_j,bc_j,cd_j = quartet_info

    neigh_new = neigh.copy()
    k2s_new = k2s.copy()

    ###SWAP C FOR A
    tri_new = tri_update(tri,quartet_info)

    neigh_new[tri_0i,tri_0j] = neigh[tri_1i,(tri_1j+1)%3]
    neigh_new[tri_0i,(tri_0j+1)%3] = neigh[bc_i,bc_j]
    neigh_new[tri_0i,(tri_0j+2)%3] = neigh[tri_0i,(tri_0j+2)%3]
    neigh_new[tri_1i,tri_1j] = neigh[tri_0i,(tri_0j+1)%3]
    neigh_new[tri_1i,(tri_1j+1)%3] = neigh[da_i,da_j]
    neigh_new[tri_1i,(tri_1j+2)%3] = neigh[tri_1i,(tri_1j+2)%3]

    k2s_new[tri_0i,tri_0j] = k2s[tri_1i,(tri_1j+1)%3]
    k2s_new[tri_0i,(tri_0j+1)%3] = k2s[bc_i,bc_j]
    k2s_new[tri_0i,(tri_0j+2)%3] = k2s[tri_0i,(tri_0j+2)%3]
    k2s_new[tri_1i,tri_1j] = k2s[tri_0i,(tri_0j+1)%3]
    k2s_new[tri_1i,(tri_1j+1)%3] = k2s[da_i,da_j]
    k2s_new[tri_1i,(tri_1j+2)%3] = k2s[tri_1i,(tri_1j+2)%3]


    neigh_new[bc_i,bc_j] = tri_0i
    k2s_new[bc_i,bc_j] = tri_0j
    neigh_new[da_i,da_j] = tri_1i
    k2s_new[da_i,da_j]= tri_1j

    return tri_new,neigh_new,k2s_new

@jit(nopython=True)
def _classify_edges(lv_x, lvm1_x, radius, lrij):
    """
    Classifies edges whether the
    """
    V_in = (lv_x < radius)
    V_out = (lvm1_x < radius)

    no_touch = (4 * (radius ** 2) / lrij ** 2 - 1) < 0

    return V_in, V_out, no_touch

@jit(nopython=True)
def get_rij(tx,L):
    """
    The displacement between the two centroids that flank the edge:

    v to vm1
    """
    return per.per3(tx - trf.roll3(tx,1),L,L)

@jit(nopython=True)
def get_rik(tx,L):
    """
    The displacement between the two centroids that flank the edge:

    v to vp1
    """
    return per.per3(tx - trf.roll3(tx,-1),L,L)


@jit(nopython=True)
def get_dij(lrij, radius):
    return (lrij ** 2 + radius) / (2 * lrij)



@jit(nopython=True)
def _tintersections(ri, rj, radius, nrij):
    ri, rj, nrij = ri.T, rj.T, nrij.T
    a = 0.5 * (ri + rj) 
    b = 0.5 * np.sqrt(4 * (radius ** 2) / nrij ** 2 - 1) * np.stack((rj[1] - ri[1], ri[0] - rj[0]))
    a, b = a.T, b.T
    pos1 = a - b
    pos2 = a + b
    return pos1, pos2


@jit(nopython=True)
def _ttheta(V_in, V_out, no_touch, radius, tx, tV, nrij, vj_neighbours, L,dir=1):
    V_in2, V_out2, no_touch2 = np.dstack((V_in, V_in)), np.dstack((V_out, V_out)), np.dstack((no_touch, no_touch))
    start, end = vj_neighbours.copy(), tV.copy()
    ##to satisfy periodic BCs
    start = tx + per.per3(start-tx,L,L)
    end = tx + per.per3(end-tx,L,L)
    rolled_tx = trf.roll3(tx, dir)
    rolled_tx = tx + per.per3(rolled_tx - tx,L,L)

    h_CCW, h_CW = _tintersections(tx, rolled_tx,radius, nrij)

    end = trf.replace_vec(end, ~V_in2, h_CCW)
    start = trf.replace_vec(start, ~V_out2, h_CW)
    end = trf.replace_val(end, no_touch2, 0)
    start = trf.replace_val(start, no_touch2, 0)

    ttheta = _tvecangle(start - tx, end - tx)

    start = per.mod3(start,L,L)
    end = per.mod3(end,L,L)

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
def numba_cdist(A,B,L):
    """
    Numba-ed version of scipy's cdist. for 2D only.
    with periodic bcs.
    """
    disp = per.per3(np.expand_dims(A,1) - np.expand_dims(B,0),L,L)
    disp_x,disp_y = disp[...,0],disp[...,1]
    disp_x = np.mod(disp_x+L/2,L)-L/2
    disp_y = np.mod(disp_x+L/2,L)-L/2
    return disp_x**2 + disp_y**2


@jit(nopython=True)
def do_classification_correction(r,radius,ttheta, hm, hp,no_touch,V_in_j,L,err=1e-7):
    """
    Deals with cases where two circles intersect within a 3rd cell

    LOOKS INCREDIBLY INEFFICIENT. Check whether it still works
    """

    touch_not_power_mask = (~V_in_j)*(~no_touch)
    if touch_not_power_mask.any():
        touch_not_power_mask_flat = touch_not_power_mask.ravel()
        hp_circ = np.column_stack((hp[...,0].ravel()[touch_not_power_mask_flat],hp[...,1].ravel()[touch_not_power_mask_flat]))
        d = numba_cdist(hp_circ,r,L) - radius**2
        false_cross = np.count_nonzero(d <= err, axis=1)>2

        touch_not_power_mask_flat[touch_not_power_mask_flat] = false_cross
        touch_not_power_mask = touch_not_power_mask_flat.reshape(touch_not_power_mask.shape)

        no_touch = trf.replace_val(no_touch,touch_not_power_mask,True)
        no_touch_vec = np.dstack((no_touch, no_touch))
        hp = trf.replace_val(hp, no_touch_vec, 0)
        hm = trf.replace_val(hm, no_touch_vec, 0)
        ttheta = trf.replace_val(ttheta, no_touch, 0)

    return hp,hm,ttheta,no_touch


@jit(nopython=True)
def get_lP(start, end,L):
    return trf.tnorm(per.per3(end - start,L,L))


@jit(nopython=True)
def get_lC(tpsi, radius):
    return tpsi * radius


@jit(nopython=True)
def get_AP(start, end, tx,L):
    return 0.5 * trf.tcross(per.per3(start - tx,L,L), per.per3(end - tx,L,L))


@jit(nopython=True)
def get_AC(tspi, radius):
    return 0.5 * tspi * radius ** 2
