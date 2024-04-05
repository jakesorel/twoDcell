
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
