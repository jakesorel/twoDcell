import numpy as np
from numba import jit
from scipy.spatial import ConvexHull
import two_dimensional_cell.tri_functions as trf

N = 25
x = np.random.uniform(0,10,(N,2))
R = np.random.uniform(0.45,0.55,N)

class PowerTriangulation:
    def __init__(self,x=None,R=None,tri=None,neigh=None,k2s=None):
        assert x is not None, "specify x"
        assert R is not None, "specify R"

        if tri is None:
            tri,vs, n_v = get_power_triangulation(x,R)
            neigh = get_neighbours(tri)
            k2s = get_k2(tri,neigh)
        else:
            assert neigh is not None, "specify neigh"
            assert k2s is not None, "specify k2s"
            self.retriangulate()

    def retriangulate(self):


@jit(nopython=True)
def get_neighbours(tri, neigh=None, Range=None):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.

    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.

    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri.shape[0]
    if neigh is None:
        neigh = np.ones_like(tri, dtype=np.int32) * -1
    if Range is None:
        Range = np.arange(n_v)

    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in Range:  # range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip, tri_sample_flip)).reshape(3, 2)
        for k in range(3):
            if neigh[j, k] == -1:
                msk = (tri_compare[:, :, 0] == tri_i[k, 0]) * (tri_compare[:, :, 1] == tri_i[k, 1])
                if msk.sum() != 0:
                    neighb, l = np.nonzero(msk)
                    neighb, l = neighb[0], l[0]
                    neigh[j, k] = neighb
                    neigh[neighb, np.mod(2 - l, 3)] = j
    return neigh


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
    k2s = np.ones((nv, 3), dtype=np.int32)*-1
    for i in range(nv):
        for k in range(3):
            neighbour = neigh[i, k]
            if neighbour != -1:
                k2 = ((neigh[neighbour] == i) * three).sum()
                k2s[i, k] = k2
    return k2s

@jit(nopython=True)
def norm2(X):
    return np.sqrt(np.sum(X ** 2))


@jit(nopython=True)
def normalized(X):
    return X / norm2(X)


@jit(nopython=True)
def lift(x, R):
    x_norm = x[0] ** 2 + x[1] ** 2 - R ** 2
    x_out = np.zeros(3)
    x_out[:2], x_out[2] = x, x_norm
    return x_out


# --- Delaunay triangulation --------------------------------------------------



@jit(nopython=True)
def get_triangle_normal(A, B, C):
    return normalized(np.cross(A, B) + np.cross(B, C) + np.cross(C, A))


@jit(nopython=True)
def get_power_circumcenter(A, B, C):
    N = get_triangle_normal(A, B, C)
    return (-.5 / N[2]) * N[:2]


@jit(nopython=True)
def is_ccw_triangle(A, B, C):
    m = np.stack((A,B,C))
    M = np.column_stack((m,np.ones(3)))
    return np.linalg.det(M) > 0


@jit(nopython=True)
def get_circumcentre(x, R):
    return get_power_circumcenter(*get_x_lifted(x, R))


@jit(nopython=True)
def get_x_lifted(x, R):
    x_norm = np.sum(x ** 2, axis=1) - R ** 2
    x_lifted = np.concatenate((x, x_norm.reshape(-1, 1)), axis=1)
    # S_lifted = np.concatenate((S.ravel(),S_norm)).reshape(3,-1).T
    return x_lifted


@jit(nopython=True)
def get_vertices(x_lifted, tri_list, n_v):
    V = np.zeros((n_v, 2))
    for i, tri in enumerate(tri_list):
        A, B, C = x_lifted[tri]
        V[i] = get_power_circumcenter(A, B, C)
    return V


@jit(nopython=True)
def build_tri_and_norm(x, simplices, equations):
    saved_tris = equations[:, 2] <= 0
    n_v = saved_tris.sum()
    norms = equations[saved_tris]
    tri_list = np.zeros((n_v, 3), dtype=np.int64)
    i = 0
    for (a, b, c), eq in zip(simplices[saved_tris], norms):
        if is_ccw_triangle(x[a], x[b], x[c]):
            tri_list[i] = a, b, c
        else:
            tri_list[i] = a, c, b
        i += 1
    return tri_list, norms, n_v


def get_power_triangulation(x, R):
    # Compute the lifted weighted points
    x_lifted = get_x_lifted(x, R)
    # Compute the convex hull of the lifted weighted points
    hull = ConvexHull(x_lifted)
    #
    # # Extract the Delaunay triangulation from the lower hull
    tri, norms, n_v = build_tri_and_norm(x, hull.simplices, hull.equations)
    #
    # # Compute the Voronoi points
    V = get_vertices(x_lifted, tri, n_v)
    #
    # # Job done
    return tri, V, n_v




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
def get_retriangulation_mask(x, tx,R,tR, tri, neigh, k2s, vs):
    if tx is None:
        tx = trf.tri_call3(x,tri)
    if tR is None:
        tR = trf.tri_call(R,tri)

    take_ids = neigh * 3 + k2s
    d_cell = tri.take(take_ids).reshape(tri.shape)
    # d_cell = trf.replace_val(d_cell,take_ids<0,-1)
    xd = trf.tri_call3(x, d_cell)
    Rd = trf.tri_call(R,d_cell)
    rad2_a = tx - np.expand_dims(vs, 1)
    rad2_a = rad2_a[...,0]**2 + rad2_a[...,1]**2 - tR**2
    rad2_d = xd - np.expand_dims(vs, 1)
    rad2_d = rad2_d[..., 0] ** 2 + rad2_d[..., 1] ** 2 - Rd**2
    mask = rad2_d < rad2_a
    mask = trf.replace_val(mask,take_ids<0,False) ##when the 'd' cell is a boundary cell.
    return mask


@jit(nopython=True)
def re_triangulate(x,_tri,_neigh,_k2s,tx0,L,ntri,vs0,max_runs=10):
    tri, neigh, k2s = _tri.copy(), _neigh.copy(), _k2s.copy()
    mask = get_retriangulation_mask(x, None,R,None, tri, neigh, k2s, vs0)
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
            mask = get_retriangulation_mask(x, None,R,None, tri, neigh, k2s, vs)
            if n_runs > max_runs:
                failed = True
                continue_loop = False
            if not mask.any():
                continue_loop = False
            n_runs += 1
    return tri,neigh,k2s,failed

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

