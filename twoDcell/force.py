import numpy as np
from numba import jit

import twoDcell.periodic_functions as per
import twoDcell.tri_functions as trf


class Force:
    """
    Force class
    -----------

    This class is used to calculate the passive mechanics under the SPV model.

    Takes in an instance of the Tissue class, and within it, the Mesh class, and uses information about the geometry to calculate the forces on the cell centroids.

    These forces are accessible in self.F.
    """

    def __init__(self, tissue):
        self.t = tissue
        self.Jp, self.Jm = None, None  ##triangular forms of the J matrix, considering J_ij for CW (Jp) and CCW (Jm) neighbours.
        self.F = None
        self.F_soft = 0  # Soft-repulsion between cell centroids, included for stability following Barton et al.
        self.get_J()
        self.get_F_mechanics()
        if self.t.a == 0:
            self.get_F_soft()
        self.F = sum_F(self.F, self.F_soft)

        self.dA = None
        self.dP = None

    def get_J(self):
        ##for now only with ref. to W, but generalizable.
        self.Jp = get_J(self.t.W, self.t.tc_types, self.t.tc_typesp, self.t.nc_types)
        self.Jm = get_J(self.t.W, self.t.tc_types, self.t.tc_typesm, self.t.nc_types)

    def get_F_mechanics(self):
        """
        Calculates the forces on cell centroids given the del of the energy functional.
        tF is the triangulated form of the forces; i.e. the components of the total force on cell i by each of the involved triangles in the triangulation.

        Energy functional is given by:

        E = Sum_i (kappa_A/2) * (Ai - A0)^2 + (kappa_P/2) * (Pi - P0)^2 + Sum_j J_ij*lij
        """
        # tF = get_tF(self.t.mesh.vp1_vm1,
        #             self.t.mesh.v_vm1,
        #             self.t.mesh.v_vp1,
        #             self.t.mesh.v_x,
        #             self.t.mesh.lm1,
        #             self.t.mesh.lp1,
        #             self.Jm,
        #             self.Jp,
        #             self.t.kappa_A,
        #             self.t.kappa_P,
        #             self.t.A0,
        #             self.t.P0,
        #             self.t.mesh.A,
        #             self.t.mesh.P,
        #             self.t.mesh.tri)

        tF = get_tF(self.t.mesh.L,
                   self.t.mesh.tri,
                   self.t.mesh.radius,
                   self.t.mesh.v_x,
                   self.t.mesh.hp_j,
                   self.t.mesh.hm_j,
                   self.t.mesh.tx,
                   self.t.mesh.no_touch_j_vec,
                   self.t.mesh.no_touch_j_mat,
                   self.t.kappa_P,
                   self.t.kappa_A,
                   self.t.kappa_M,
                   self.t.mesh.P,
                   self.t.mesh.A,
                   self.t.P0,
                   self.t.A0,
                   self.t.mesh.V_in_j)
        self.F = trf.assemble_tri3(tF,
                                   self.t.mesh.tri)  ##this assembles the total force on each cell centroid by summing the contributions from each triangle.
        return self.F

    def get_F_soft(self):
        """
        Soft repulsion between cell centroids. Spring-like force under E_soft = Sum_bond k*(|r_i - r_j| - a)^2 where bond = {i,j} for pairs of centroids i,j for which |r_i - r_j| <= a
        :return:
        """
        self.F_soft = trf.assemble_tri3(get_tFsoft(self.t.mesh.tx,
                                                   self.t.a,
                                                   self.t.k,
                                                   self.t.mesh.L),
                                        self.t.mesh.tri)


@jit(nopython=True)
def get_J(W, tc_types, neigh_tctypes, nc_types):
    return W.take(tc_types.ravel() + nc_types * neigh_tctypes.ravel()).reshape(-1, 3)


@jit(nopython=True)
def get_tF(L, tri, radius, v_x, hp_j, hm_j, tx, no_touch_j_vec, no_touch_j_mat, kappa_P, kappa_A, kappa_M, P, A, P0, A0, V_in_j):
    hp_ri = per.per3(hp_j - tx,L,L)
    hm_ri = per.per3(hm_j - tx,L,L)
    hm_ri_z = np.dstack((-hm_ri[:, :, 1], hm_ri[:, :, 0]))
    hp_ri_z = np.dstack((-hp_ri[:, :, 1], hp_ri[:, :, 0]))

    nhp_ri = trf.tnorm(hp_ri)
    nhm_ri = trf.tnorm(hm_ri)

    dtheta_dhp_cell_i = (hp_ri_z.T / nhp_ri.T ** 2).T
    dtheta_dhp_cell_i = trf.replace_val(dtheta_dhp_cell_i,no_touch_j_vec,0)
    # dtheta_dhp_cell_i[no_touch_j_vec] = 0

    rj = trf.roll3(tx, 1)

    hp_rj = per.per3(hp_j - rj,L,L)
    hm_rj = per.per3(hm_j - rj,L,L)
    hp_rj_z = np.dstack((-hp_rj[:, :, 1], hp_rj[:, :, 0]))
    hm_rj_z = np.dstack((-hm_rj[:, :, 1], hm_rj[:, :, 0]))

    nhp_rj = trf.tnorm(hp_rj)
    dtheta_dhp_cell_j = -(hp_rj_z.T / nhp_rj.T ** 2).T
    # dtheta_dhp_cell_j[no_touch_j_vec] = 0
    dtheta_dhp_cell_j = trf.replace_val(dtheta_dhp_cell_j,no_touch_j_vec,0)


    hphm = per.per3(hp_j - hm_j,L,L)
    hphm_z = np.dstack((-hphm[:, :, 1], hphm[:, :, 0]))
    nhphm = trf.tnorm(hphm)
    hphm_unit = (hphm.T / nhphm.T).T

    # hphm_unit[no_touch_j_vec] = 0
    hphm_unit = trf.replace_val(hphm_unit,no_touch_j_vec,0)


    ##Calculate ∂E/∂Pi and ∂E/∂Ai i.e. for the ith cell, and convert to triangulated form
    dE_dPi = trf.tri_call(2 * kappa_P * (P - P0), tri)
    dE_dAi = trf.tri_call(2 * kappa_A * (A - A0), tri)
    dE_dMi = trf.tri_call(kappa_M, tri)

    dlCi_dtheta = -radius
    dACi_dtheta = -(radius ** 2 / 2)

    # Calculate ∂E/dtheta_ij and ∂E/dtheta_ji by summing the contributions
    # Note that roll(theta_ik) = theta_ji
    dEPC_dtheta_ij = dE_dPi * dlCi_dtheta
    dEPC_dtheta_ji = trf.roll(dE_dPi * dlCi_dtheta, 1)
    dEAC_dtheta_ij = dE_dAi * dACi_dtheta
    dEAC_dtheta_ji = trf.roll(dE_dAi * dACi_dtheta, 1)
    dEM_dtheta_ij = dE_dMi * dlCi_dtheta
    dEM_dtheta_ik = trf.roll(dE_dMi * dlCi_dtheta, 1)
    dE_dtheta_ij = dEPC_dtheta_ij + dEAC_dtheta_ij + dEM_dtheta_ij
    dE_dtheta_ji = dEPC_dtheta_ji + dEAC_dtheta_ji + dEM_dtheta_ik

    #####NOTE ^^ and below can be simplified hugely. Do this at some point. rolls can be pulled into a single operation.

    # Calculate ∂E/dhp by chain rule.
    # This statement below is equivalent to: ∂E/∂hpij = ∂Ei/∂hpij + ∂Ej/∂hpij
    # This is because hpij has two contributions: from cell i and from cell j
    dEC_dhp = (dE_dtheta_ij.T * dtheta_dhp_cell_i.T + dE_dtheta_ji.T * dtheta_dhp_cell_j.T).T

    dlPi_dhp = hphm_unit
    dEAP_dhp = 0.5 * ((hm_ri_z.T * dE_dAi.T).T - (trf.roll(dE_dAi,1).T * hm_rj_z.T).T)  ##note the minus sign here, as measuring area CW rather than antiCW, i.e. flipped order in the cross prod.

    dEP_dhp = ((dE_dPi.T + trf.roll(dE_dPi, 1).T) * dlPi_dhp.T).T + dEAP_dhp

    dE_dhp = dEC_dhp + dEP_dhp

    ##Calculate the jacobians.
    # Some degeneracy here, so can be optimized later
    dhCCWj_dri, dhCWj_dri = get_circle_vertex_differentials(tx, rj, radius, no_touch_j_mat,L)
    dhCCWj_drj, dhCWj_drj = get_circle_vertex_differentials(rj, tx, radius, no_touch_j_mat,L)

    # Calculate the forces at each cell (for each triangle contribution) for the terms with ∂theta/∂hp
    n_v = len(tri)
    tF_c_h = np.zeros((n_v, 3, 2))
    for i in range(n_v):
        for j in range(3):
            tF_c_h[i, j] += (dE_dhp[i, j]) @ (dhCCWj_dri[i, j])
            tF_c_h[i, np.mod(j + 1, 3)] += (dE_dhp[i, j]) @ (dhCWj_drj[i, j])

    # Calculate the forces at each cell for terms involving ∂theta/∂ri
    tF_s = 0.5 * (dE_dAi.T * hphm_z.T).T + (dE_dtheta_ij.T * (hm_ri_z.T / nhm_ri.T ** 2 - hp_ri_z.T / nhp_ri.T ** 2)).T
    tF_s = trf.replace_val(tF_s,np.isnan(tF_s),0)
    # tF_s[np.isnan(tF_s)] = 0  ##need as some will be divide by 0

    dhv_dri = get_dvdr(v_x)  # order is wrt cell i



    dE_dhv = dEP_dhp.sum(axis=1)

    ##Forces for each cell if cells meet at a power vertex, for cases ∂theta/dhv (i.e. = dhp)
    tF_v_h = np.zeros((n_v, 3, 2))
    for i in range(n_v):
        for j in range(3):
            tF_v_h[i, j] += (dE_dhv[i]) @ dhv_dri[i, j]

    ##Compile accounting for whether power or circle vertex
    tF_h = tF_c_h.copy()
    # tF_h[V_in_j] = tF_v_h[V_in_j]
    tF_h = trf.replace_vec(tF_h,V_in_j,tF_v_h)

    tF_h = tF_v_h.copy()

    tF = -(tF_h + tF_s)
    return tF

#
# @jit(nopython=True)
# def get_tF(vp1_vm1, v_vm1, v_vp1, v_x, lm1, lp1, Jm, Jp, kappa_A, kappa_P, A0, P0, A, P, tri):
#     dAdv_j = np.dstack((vp1_vm1[:, :, 1], -vp1_vm1[:, :, 0])) * 0.5  ##shoelace theorem: i.e. derivative of cross product.
#
#     dPdv_j_m = v_vm1 / np.expand_dims(lm1, 2)
#     dPdv_j_p = v_vp1 / np.expand_dims(lp1, 2)
#     dPdv_j = dPdv_j_p + dPdv_j_m
#
#     dtEdv_l_v_j = dPdv_j_m * np.expand_dims(Jm, 2) + dPdv_j_p * np.expand_dims(Jp, 2)
#
#     dtEdA = trf.tri_call(2 * kappa_A * (A - A0), tri)
#     dtEdP = trf.tri_call(2 * kappa_P * (P - P0), tri)
#
#     dtE_dv = np.expand_dims(dtEdA, 2) * dAdv_j + np.expand_dims(dtEdP, 2) * dPdv_j + dtEdv_l_v_j
#     dtE_dv = dtE_dv[:, 0] + dtE_dv[:, 1] + dtE_dv[:, 2]  # sum over the three contributions
#
#     dvdr = get_dvdr(v_x)  # order is wrt cell i
#
#     dtE_dv = np.expand_dims(dtE_dv, 2)
#
#     dEdr_x = dtE_dv[:, 0] * dvdr[:, :, 0, 0] + dtE_dv[:, 1] * dvdr[:, :, 0, 1]
#     dEdr_y = dtE_dv[:, 0] * dvdr[:, :, 1, 0] + dtE_dv[:, 1] * dvdr[:, :, 1, 1]
#
#     dEdr = np.dstack((dEdr_x, dEdr_y))
#     F = - dEdr
#     return F



@jit(nopython=True)
def _power_vertex_differentials(ri,rj,rk,radius):
    # ri, rj, rk = geom.tS, geom.rj, geom.rk
    # Ri,Rj,Rk = geom.tR,geom.Rj,geom.Rk
    Ri, Rj, Rk = np.ones(ri.shape[0])*radius,np.ones(ri.shape[0])*radius,np.ones(ri.shape[0])*radius
    (rix, riy), (rjx, rjy),(rkx,rky), Ri, Rj,Rk = ri.T, rj.T,rk.T, Ri.T, Rj.T,Rk.T

    # di0 = _tnorm(tS)**2 - tR**2
    # dj0,dk0 = _roll(di0,1),_roll(di0,-1)
    # rj,rk = _roll3(tS,1),_roll3(tS,-1)
    # rkj = rk - rj
    # rji = rj - tS
    # rkj_z = np.dstack((-rkj[:,:,1],rkj[:,:,0]))
    # first_term = _touter(tS,rkj_z)
    # second_term = -_touter(rkj_z,tV)
    # third_term = np.zeros_like(first_term)
    # dkj2 = 0.5*(dk0 - dj0)
    # third_term[:,:,0,1],third_term[:,:,1,0] = -dkj2,dkj2
    # cross = _tcross(rkj,rji)
    # dhv_dri = ((first_term+second_term+third_term).T/cross.T).T
    # return dhv_dri,dhv_dri[:,:,0]*0
    dhvx_drix = -((rjy - rky) * (-(rix ** 2 * rjy) + rjy * Rk ** 2 + 2 * rix * rjy * rkx - rjy * rkx ** 2 + riy ** 2 * (
                rjy - rky) + rix ** 2 * rky - Rj ** 2 * rky - 2 * rix * rjx * rky + rjx ** 2 * rky + rjy ** 2 * rky - rjy * rky ** 2 + Ri ** 2 * (
                                             -rjy + rky) + riy * (
                                             Rj ** 2 + 2 * rix * rjx - rjx ** 2 - rjy ** 2 - Rk ** 2 - 2 * rix * rkx + rkx ** 2 + rky ** 2))) / (
                            2. * (riy * (rjx - rkx) + rjy * rkx - rjx * rky + rix * (-rjy + rky)) ** 2)
    dhvy_drix = ((rjx - rkx) * (-(rix ** 2 * rjy) + rjy * Rk ** 2 + 2 * rix * rjy * rkx - rjy * rkx ** 2 + riy ** 2 * (
                rjy - rky) + rix ** 2 * rky - Rj ** 2 * rky - 2 * rix * rjx * rky + rjx ** 2 * rky + rjy ** 2 * rky - rjy * rky ** 2 + Ri ** 2 * (
                                            -rjy + rky) + riy * (
                                            Rj ** 2 + 2 * rix * rjx - rjx ** 2 - rjy ** 2 - Rk ** 2 - 2 * rix * rkx + rkx ** 2 + rky ** 2))) / (
                            2. * (riy * (rjx - rkx) + rjy * rkx - rjx * rky + rix * (-rjy + rky)) ** 2)

    dhvx_driy = ((rjy - rky)*(-(riy**2*rjx) + rjx*Rk**2 + rix**2*(rjx - rkx) + riy**2*rkx - Rj**2*rkx + rjx**2*rkx - 2*riy*rjy*rkx + rjy**2*rkx - rjx*rkx**2 + Ri**2*(-rjx + rkx) + 2*riy*rjx*rky - rjx*rky**2 + rix*(Rj**2 - rjx**2 + 2*riy*rjy - rjy**2 - Rk**2 + rkx**2 - 2*riy*rky + rky**2)))/(2.*(riy*(rjx - rkx) + rjy*rkx - rjx*rky + rix*(-rjy + rky))**2)

    dhvy_driy =  -((rjx - rkx)*(-(riy**2*rjx) + rjx*Rk**2 + rix**2*(rjx - rkx) + riy**2*rkx - Rj**2*rkx + rjx**2*rkx - 2*riy*rjy*rkx + rjy**2*rkx - rjx*rkx**2 + Ri**2*(-rjx + rkx) + 2*riy*rjx*rky - rjx*rky**2 + rix*(Rj**2 - rjx**2 + 2*riy*rjy - rjy**2 - Rk**2 + rkx**2 - 2*riy*rky + rky**2)))/(2.*(riy*(rjx - rkx) + rjy*rkx - rjx*rky + rix*(-rjy + rky))**2)
    dhv_dri = np.stack((np.stack((dhvx_drix,dhvy_drix)),np.stack((dhvx_driy,dhvy_driy)))).T

    # dhv_dRi = np.stack(((Ri*(rjy - rky))/(riy*(rjx - rkx) + rjy*rkx - rjx*rky + rix*(-rjy + rky)),(Ri*(rjx - rkx))/(-(rjy*rkx) + riy*(-rjx + rkx) + rix*(rjy - rky) + rjx*rky))).T

    return dhv_dri#,dhv_dRi


@jit(nopython=True)
def get_dvdr(v_x):
    """

    Calculates ∂v_j/dr_i the Jacobian for all cells in each triangulation

    Last two dims: ((dvx/drx,dvx/dry),(dvy/drx,dvy/dry))

    These are lifted from Mathematica

    :param x_v_: (n_v x 3 x 2) np.float32 array of cell centroid positions for each cell in each triangulation (first two dims follow order of triangulation)
    :param vs: (n_v x 2) np.float32 array of vertex positions, corresponding to each triangle in the triangulation
    :param L: Domain size (np.float32)
    :return: Jacobian for each cell of each triangulation (n_v x 3 x 2 x 2) np.float32 array (where the first 2 dims follow the order of the triangulation.
    """
    x_v = -v_x
    dvdr = np.empty(x_v.shape + (2,))
    for i in range(3):
        ax, ay = x_v[:, np.mod(i, 3), 0], x_v[:, np.mod(i, 3), 1]
        bx, by = x_v[:, np.mod(i + 1, 3), 0], x_v[:, np.mod(i + 1, 3), 1]
        cx, cy = x_v[:, np.mod(i + 2, 3), 0], x_v[:, np.mod(i + 2, 3), 1]
        # dhx/drx
        dvdr[:, i, 0, 0] = (ax * (by - cy)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((by - cy) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        # dhy/drx
        dvdr[:, i, 0, 1] = (bx ** 2 + by ** 2 - cx ** 2 + 2 * ax * (-bx + cx) - cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((by - cy) * (
                (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        # dhx/dry
        dvdr[:, i, 1, 0] = (-bx ** 2 - by ** 2 + cx ** 2 + 2 * ay * (by - cy) + cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((-bx + cx) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        # dhy/dry
        dvdr[:, i, 1, 1] = (ay * (-bx + cx)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((-bx + cx) * (
                (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

    return dvdr


@jit(nopython=True)
def get_tFsoft(tx, a, k, L):
    """
    Additional "soft" pair-wise repulsion at short range to prevent unrealistic and sudden changes in triangulation.

    Repulsion is on the imediate neighbours (i.e. derived from the triangulation)

    And is performed respecting periodic boudnary conditions (system size = L)

    Suppose l_{ij} = \| r_i - r_j \
    F_soft = -k(l_{ij} - 2a)(r_i - r_j) if l_{ij} < 2a; and =0 otherwise

    :param Cents: Cell centroids on the triangulation (n_v x 3 x 2) **np.ndarray** of dtype **np.float64**
    :param a: Cut-off distance of spring-like interaction (**np.float64**)
    :param k: Strength of spring-like interaction (**np.float64**)
    :param CV_matrix: Cell-vertex matrix representation of the triangulation (n_c x n_v x 3)
    :param n_c: Number of cells (**np.int64**)
    :param L: Domain size/length (**np.float64**)
    :return: F_soft
    """
    rj = trf.roll3(tx, 1)
    rij = per.per3(tx - rj, L, L)
    lij = trf.tnorm(rij)
    norm_ij = rij / np.expand_dims(lij, 2)
    tFsoft_ij = np.expand_dims(-k * (lij - 2 * a) * (lij < 2 * a), 2) * norm_ij
    tFsoft = tFsoft_ij - trf.roll3(tFsoft_ij, -1)
    return tFsoft


@jit(nopython=True)
def sum_F(F, F_soft):
    return F + F_soft



@jit(nopython=True)
def get_circle_vertex_differentials(ri,rj_,radius,no_touch_mat,L):
    rj = ri + per.per3(rj_-ri,L,L)
    (rix, riy), (rjx, rjy) = ri.T, rj.T

    da_dri = np.array(((0.5,0),(0,0.5)))
    F = np.sqrt(-1 + (4 * radius ** 2) / ((rix - rjx) ** 2 + (riy - rjy) ** 2))
    D2 = (rix - rjx)**2 + (riy - rjy)**2
    db_drix = np.stack(((2.*radius**2*(rix - rjx)*(-riy + rjy))/(D2**2*F),0.5*F - (2.*radius**2*(rix - rjx)**2)/(D2**2*F)))
    db_driy = np.stack((-0.5*F - (2.*radius**2*(riy - rjy)*(-riy + rjy))/(D2**2*F),(-2.*radius**2*(rix - rjx)*(riy - rjy))/(D2**2*F)))
    db_dri = np.stack((db_drix, db_driy)).T
    db_dri = trf.replace_val(db_dri,np.isnan(db_dri),0)
    dhCCW_dri, dhCW_dri = da_dri - db_dri, da_dri + db_dri
    dhCCW_dri = trf.replace_val(dhCCW_dri,no_touch_mat,0)
    dhCW_dri = trf.replace_val(dhCW_dri, no_touch_mat, 0)
    return dhCCW_dri, dhCW_dri
#
