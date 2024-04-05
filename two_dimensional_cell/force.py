import numpy as np
from numba import jit

from two_dimensional_cell.differentials import _power_vertex_differentials, _circle_vertex_differentials
# from two_dimensional_cell.tri_functions import _roll, _roll3, _tnorm, _tri_sum, _CV_matrix, _get_neighbours, \
#     _replace_val, _tdot, _tcross, _repeat_mat, _repeat_vec, \
#     _replace_vec

import two_dimensional_cell.tri_functions as trf

from scipy.spatial.distance import cdist


import numpy as np
from numba import jit

import two_dimensional_cell.periodic_functions as per
import two_dimensional_cell.tri_functions as trf


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
        self.get_mechanics()

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
        tF = get_tF(self.t.mesh.tri_per,
                    self.t.mesh.tx,
                    self.t.mesh.rj,
                    self.t.mesh.rk,
                    self.t.mesh.tR,
                    self.t.mesh.Rj,
                    self.t.mesh.Rk,
                    self.t.mesh.hp_j,
                    self.t.mesh.hm_j,
                    self.t.mesh.no_touch_j_vec,
                    self.t.mesh.no_touch_j_mat,
                    self.t.mesh.V_in_j,
                    self.t.mesh.A,
                    self.t.mesh.P,
                    self.t.A0,
                    self.t.P0,
                    self.t.kappa_A,
                    self.t.kappa_P,
                    self.t.kappa_M,
                    self.Jm,
                    self.Jp,
                    self.t.mesh.LC)

        self.F = trf.assemble_tri3(tF,
                                   self.t.mesh.tri)[:self.t.mesh.n_c]  ##this assembles the total force on each cell centroid by summing the contributions from each triangle.
        return self.F


    def get_mechanics(self):
        """
        Calculates the forces on cell centroids given the del of the energy functional.
        tF is the triangulated form of the forces; i.e. the components of the total force on cell i by each of the involved triangles in the triangulation.

        Energy functional is given by:

        E = Sum_i (kappa_A/2) * (Ai - A0)^2 + (kappa_P/2) * (Pi - P0)^2 + Sum_j J_ij*lij
        """
        tF,tG = get_tF_tG(self.t.mesh.tri_per,
                    self.t.mesh.tx,
                    self.t.mesh.rj,
                    self.t.mesh.rk,
                    self.t.mesh.tR,
                    self.t.mesh.Rj,
                    self.t.mesh.Rk,
                    self.t.mesh.hp_j,
                    self.t.mesh.hm_j,
                    self.t.mesh.no_touch_j_vec,
                    self.t.mesh.no_touch_j_mat,
                    self.t.mesh.V_in_j,
                    self.t.mesh.A,
                    self.t.mesh.P,
                    self.t.A0,
                    self.t.P0,
                    self.t.kappa_A,
                    self.t.kappa_P,
                    self.t.kappa_M,
                    self.Jm,
                    self.Jp,
                    self.t.mesh.LC)

        self.F = trf.assemble_tri3(tF,
                                   self.t.mesh.tri)[:self.t.mesh.n_c]  ##this assembles the total force on each cell centroid by summing the contributions from each triangle.
        self.G = trf.assemble_tri(tG,self.t.mesh.tri)[:self.t.mesh.n_c]

        return self.F,self.G

    # def get_F_soft(self):
    #     """
    #     Soft repulsion between cell centroids. Spring-like force under E_soft = Sum_bond k*(|r_i - r_j| - a)^2 where bond = {i,j} for pairs of centroids i,j for which |r_i - r_j| <= a
    #     :return:
    #     """
    #     self.F_soft = trf.assemble_tri3(get_tFsoft(self.t.mesh.tx,
    #                                                self.t.a,
    #                                                self.t.k,
    #                                                self.t.mesh.L),
    #                                     self.t.mesh.tri)
    #


#
# @jit(nopython=True)
# def get_tF_synmorph(vp1_vm1, v_vm1, v_vp1, v_x, lm1, lp1, Jm, Jp, kappa_A, kappa_P, A0, P0, A, P, tri):
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
def get_relative_geometries(tx,rj,hp_j,hm_j,no_touch_j_vec):
    hp_ri = hp_j - tx
    hm_ri = hm_j - tx
    hm_ri_z = np.dstack((-hm_ri[:, :, 1], hm_ri[:, :, 0]))
    hp_ri_z = np.dstack((-hp_ri[:, :, 1], hp_ri[:, :, 0]))


    nhp_ri = trf.tnorm(hp_ri)
    nhm_ri = trf.tnorm(hm_ri)
    dtheta_dhp_cell_i = (hp_ri_z.T / nhp_ri.T ** 2).T
    dtheta_dhp_cell_i = trf.replace_val(dtheta_dhp_cell_i, no_touch_j_vec, 0)
    # dtheta_dhp_cell_i[no_touch_j_vec] = 0

    hp_rj = hp_j - rj
    hm_rj = hm_j - rj
    hp_rj_z = np.dstack((-hp_rj[:, :, 1], hp_rj[:, :, 0]))
    hm_rj_z = np.dstack((-hm_rj[:, :, 1], hm_rj[:, :, 0]))


    nhp_rj = trf.tnorm(hp_rj)
    # nhm_rj = trf.tnorm(hm_rj)
    dtheta_dhp_cell_j = -(hp_rj_z.T / nhp_rj.T ** 2).T
    dtheta_dhp_cell_j = trf.replace_val(dtheta_dhp_cell_j, no_touch_j_vec, 0)
    # dtheta_dhp_cell_j[no_touch_j_vec] = 0

    hphm = hp_j - hm_j
    hphm_z = np.dstack((-hphm[:, :, 1], hphm[:, :, 0]))
    nhphm = trf.tnorm(hphm)
    hphm_unit = (hphm.T / nhphm.T).T
    hphm_unit = trf.replace_val(hphm_unit, no_touch_j_vec, 0)

    # hphm_unit[no_touch_j_vec] = 0

    relative_geometries = (hp_ri,
                           hm_ri,
                           hm_ri_z,
                           hp_ri_z,
                           nhp_ri,
                           nhm_ri,
                           dtheta_dhp_cell_i,
                           hp_rj,
                           hm_rj,
                           hp_rj_z,
                           hm_rj_z,
                           nhp_rj,
                           dtheta_dhp_cell_j,
                           hphm,
                           hphm_z,
                           nhphm,
                           hphm_unit)

    return relative_geometries

@jit(nopython=True)
def get_dE_dvariables(relative_geometries,tri, tR, A, P, A0, P0, lambda_A,lambda_P, lambda_M, Jm, Jp):
    (hp_ri,
     hm_ri,
     hm_ri_z,
     hp_ri_z,
     nhp_ri,
     nhm_ri,
     dtheta_dhp_cell_i,
     hp_rj,
     hm_rj,
     hp_rj_z,
     hm_rj_z,
     nhp_rj,
     dtheta_dhp_cell_j,
     hphm,
     hphm_z,
     nhphm,
     hphm_unit) = relative_geometries
    ##Calculate ∂E/∂Pi and ∂E/∂Ai i.e. for the ith cell, and convert to triangulated form
    Pb = _add_boundary_naughts(P)
    Ab = _add_boundary_naughts(A)

    dE_dPi = trf.tri_call(lambda_P*(Pb-P0),tri)#_triangulated_form(lambda_P * (P - P0), tri)
    dE_dAi = trf.tri_call(lambda_A*(Ab-A0),tri)#_triangulated_form(lambda_A * (A - A0), tri)
    dE_dMi = trf.tri_call(lambda_M*np.ones_like(Pb),tri)#_triangulated_form(lambda_M * np.ones_like(P), tri)
    dE_dlP = np.expand_dims(Jm+trf.roll(Jp,1),2)

    dlCi_dtheta = -tR
    dACi_dtheta = -(tR ** 2 / 2)

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

    dEPP_dhp = ((dE_dPi.T + trf.roll(dE_dPi, 1).T) * dlPi_dhp.T).T

    dElP_dhp = dlPi_dhp*dE_dlP #differential adhesion


    dEP_dhp = dEPP_dhp+ dEAP_dhp + dElP_dhp

    dE_dhp = dEC_dhp + dEP_dhp

    dE_dhv = dEP_dhp.sum(axis=1)

    dE_dvariables = (Pb,
                     Ab,
                     dE_dPi,
                     dE_dAi,
                     dE_dMi,
                     dE_dlP,
                     dlCi_dtheta,
                     dACi_dtheta,
                     dEPC_dtheta_ij,
                     dEPC_dtheta_ji,
                     dEAC_dtheta_ij,
                     dEAC_dtheta_ji,
                     dEM_dtheta_ij,
                     dEM_dtheta_ik,
                     dE_dtheta_ij,
                     dE_dtheta_ji,
                     dEC_dhp,
                     dlPi_dhp,
                     dEAP_dhp,
                     dEPP_dhp,
                     dElP_dhp,
                     dEP_dhp,
                     dE_dhp,
                     dE_dhv)
    return dE_dvariables

@jit(nopython=True)
def get_jacobians(tx, rj, rk, tR, Rj, Rk,no_touch_j_mat):
    ##Calculate the jacobians.
    dhCCWj_dri, dhCWj_dri, dhCCWj_dRi, dhCWj_dRi = \
        _circle_vertex_differentials(tx, rj, tR, Rj, no_touch_j_mat)
    dhCCWj_drj, dhCWj_drj, dhCCWj_dRj, dhCWj_dRj = \
        _circle_vertex_differentials(rj, tx, Rj, tR, no_touch_j_mat)
    dhv_dri, dhv_dRi = _power_vertex_differentials(tx, rj, rk, tR, Rj, Rk)
    return dhCCWj_dri,dhCWj_dri,dhCCWj_dRi,dhCWj_dRi,dhCCWj_drj,dhCWj_drj,dhCCWj_dRj,dhCWj_dRj,dhv_dri,dhv_dRi

# @jit(nopython=True)
def get_tF(tri, tx, rj, rk, tR, Rj, Rk, hp_j, hm_j, no_touch_j_vec, no_touch_j_mat, V_in_j, A, P, A0, P0, lambda_A,
               lambda_P, lambda_M, Jm, Jp,LC):
    relative_geometries = get_relative_geometries(tx,rj,hp_j,hm_j,no_touch_j_vec)
    dE_dvariables = get_dE_dvariables(relative_geometries, tri, tR, A, P, A0, P0, lambda_A, lambda_P, lambda_M, Jm, Jp)
    differentials = get_jacobians(tx, rj, rk, tR, Rj, Rk, no_touch_j_mat)
    return _get_tF(relative_geometries,dE_dvariables,differentials,tri, tx, rj, rk, tR, Rj, Rk, hp_j, hm_j, no_touch_j_vec, no_touch_j_mat, V_in_j, A, P, A0, P0, lambda_A,
               lambda_P, lambda_M, Jm, Jp,LC)

# @jit(nopython=True)
def _get_tF(relative_geometries,dE_dvariables,differentials,tri, tx, rj, rk, tR, Rj, Rk, hp_j, hm_j, no_touch_j_vec, no_touch_j_mat, V_in_j, A, P, A0, P0, lambda_A,
               lambda_P, lambda_M, Jm, Jp,LC):

    (Pb,
     Ab,
     dE_dPi,
     dE_dAi,
     dE_dMi,
     dE_dlP,
     dlCi_dtheta,
     dACi_dtheta,
     dEPC_dtheta_ij,
     dEPC_dtheta_ji,
     dEAC_dtheta_ij,
     dEAC_dtheta_ji,
     dEM_dtheta_ij,
     dEM_dtheta_ik,
     dE_dtheta_ij,
     dE_dtheta_ji,
     dEC_dhp,
     dlPi_dhp,
     dEAP_dhp,
     dEPP_dhp,
     dElP_dhp,
     dEP_dhp,
     dE_dhp,
     dE_dhv) = dE_dvariables

    (hp_ri,
     hm_ri,
     hm_ri_z,
     hp_ri_z,
     nhp_ri,
     nhm_ri,
     dtheta_dhp_cell_i,
     hp_rj,
     hm_rj,
     hp_rj_z,
     hm_rj_z,
     nhp_rj,
     dtheta_dhp_cell_j,
     hphm,
     hphm_z,
     nhphm,
     hphm_unit) = relative_geometries

    (dhCCWj_dri, dhCWj_dri,
     dhCCWj_dRi, dhCWj_dRi,
     dhCCWj_drj, dhCWj_drj,
     dhCCWj_dRj, dhCWj_dRj,
     dhv_dri, dhv_dRi) = differentials

    # Calculate the forces at each cell for terms involving ∂theta/∂ri
    tF_s = 0.5 * (dE_dAi.T * hphm_z.T).T + \
                (dE_dtheta_ij.T * (hm_ri_z.T / nhm_ri.T ** 2 - hp_ri_z.T / nhp_ri.T ** 2)).T
    tF_s = trf.replace_val(tF_s, np.isnan(tF_s), 0)


    # Calculate the forces at each cell (for each triangle contribution) for the terms with ∂theta/∂hp
    tF_h = np.zeros_like(tx)
    for i in range(tx.shape[0]):
        for j in range(3):
            if V_in_j[i,j]:
                tF_h[i, j] += (dE_dhv[i]) @ dhv_dri[i, j]
            else:
                tF_h[i, j] += (dE_dhp[i, j]) @ (dhCCWj_dri[i, j])
                tF_h[i, np.mod(j + 1, 3)] += (dE_dhp[i, j]) @ (dhCWj_drj[i, j])

    # Old version below:
    # -----------------
    # ##Forces for each cell if cells meet at a power vertex, for cases ∂theta/dhv (i.e. = dhp)
    # tF_v_h = np.zeros_like(tx)
    # for i in range(tx.shape[0]):
    #     for j in range(3):
    #         tF_v_h[i, j] += (dE_dhv[i]) @ dhv_dri[i, j]
    #
    # ##Compile accounting for whether power or circle vertex
    # tF_h = tF_c_h.copy()
    # # tF_h = trf.replace_vec(tF_h,V_in_j,tF_v_h)
    # tF_h[V_in_j] = tF_v_h[V_in_j]

    #
    # # Calculate the forces at each cell (for each triangle contribution) for the terms with ∂theta/∂hp
    # tF_c_h = np.zeros_like(tx)
    # for i in range(tx.shape[0]):
    #     for j in range(3):
    #         tF_c_h[i, j] += (dE_dhp[i, j]) @ (dhCCWj_dri[i, j])
    #         tF_c_h[i, np.mod(j + 1, 3)] += (dE_dhp[i, j]) @ (dhCWj_drj[i, j])
    #
    # # Calculate the forces at each cell for terms involving ∂theta/∂ri
    # tF_s = 0.5 * (dE_dAi.T * hphm_z.T).T + \
    #             (dE_dtheta_ij.T * (hm_ri_z.T / nhm_ri.T ** 2 - hp_ri_z.T / nhp_ri.T ** 2)).T
    # tF_s = trf.replace_val(tF_s, np.isnan(tF_s), 0)
    #
    #
    # # tF_s[np.isnan(tF_s)] = 0  ##need as some will be divide by 0
    #
    #
    # ##Forces for each cell if cells meet at a power vertex, for cases ∂theta/dhv (i.e. = dhp)
    # tF_v_h = np.zeros_like(tx)
    # for i in range(tx.shape[0]):
    #     for j in range(3):
    #         tF_v_h[i, j] += (dE_dhv[i]) @ dhv_dri[i, j]
    #
    # ##Compile accounting for whether power or circle vertex
    # tF_h = tF_c_h.copy()
    # # tF_h = trf.replace_vec(tF_h,V_in_j,tF_v_h)
    # tF_h[V_in_j] = tF_v_h[V_in_j]
    #---
    #END
    #---


    tF = tF_h + tF_s
    return -tF

# @jit(nopython=True)
def get_tG(tri, tx, rj, rk, tR, Rj, Rk, hp_j, hm_j, no_touch_j_vec, no_touch_j_mat, V_in_j, A, P, A0, P0, lambda_A,
               lambda_P, lambda_M, Jm, Jp,LC):
    relative_geometries = get_relative_geometries(tx,rj,hp_j,hm_j,no_touch_j_vec)
    dE_dvariables = get_dE_dvariables(relative_geometries, tri, tR, A, P, A0, P0, lambda_A, lambda_P, lambda_M, Jm, Jp)
    differentials = get_jacobians(tx, rj, rk, tR, Rj, Rk, no_touch_j_mat)
    return _get_tG(relative_geometries,dE_dvariables,differentials,tri, tx, rj, rk, tR, Rj, Rk, hp_j, hm_j, no_touch_j_vec, no_touch_j_mat, V_in_j, A, P, A0, P0, lambda_A,
               lambda_P, lambda_M, Jm, Jp,LC)

# @jit(nopython=True)
def _get_tG(relative_geometries,dE_dvariables,differentials,tri, tx, rj, rk, tR, Rj, Rk, hp_j, hm_j, no_touch_j_vec, no_touch_j_mat, V_in_j, A, P, A0, P0, lambda_A,
               lambda_P, lambda_M, Jm, Jp,LC):

    """
    NEED TO ADD J contribution?

    Does tG have a self tG_s term? i.e. /dri
    """


    (Pb,
     Ab,
     dE_dPi,
     dE_dAi,
     dE_dMi,
     dE_dlP,
     dlCi_dtheta,
     dACi_dtheta,
     dEPC_dtheta_ij,
     dEPC_dtheta_ji,
     dEAC_dtheta_ij,
     dEAC_dtheta_ji,
     dEM_dtheta_ij,
     dEM_dtheta_ik,
     dE_dtheta_ij,
     dE_dtheta_ji,
     dEC_dhp,
     dlPi_dhp,
     dEAP_dhp,
     dEPP_dhp,
     dElP_dhp,
     dEP_dhp,
     dE_dhp,
     dE_dhv) = dE_dvariables
    #
    # (hp_ri,
    #  hm_ri,
    #  hm_ri_z,
    #  hp_ri_z,
    #  nhp_ri,
    #  nhm_ri,
    #  dtheta_dhp_cell_i,
    #  hp_rj,
    #  hm_rj,
    #  hp_rj_z,
    #  hm_rj_z,
    #  nhp_rj,
    #  dtheta_dhp_cell_j,
    #  hphm,
    #  hphm_z,
    #  nhphm,
    #  hphm_unit) = relative_geometries

    (dhCCWj_dri, dhCWj_dri,
     dhCCWj_dRi, dhCWj_dRi,
     dhCCWj_drj, dhCWj_drj,
     dhCCWj_dRj, dhCWj_dRj,
     dhv_dri, dhv_dRi) = differentials

    nv = tri.shape[0]
    tG_h = np.zeros((nv, 3))
    for i in range(nv):
        for j in range(3):
            if V_in_j[i,j]:
                tG_h[i, j] += (dE_dhv[i]) @ dhv_dRi[i, j]
            else:
                tG_h[i, j] += (dE_dhp[i, j]) @ (dhCCWj_dRi[i, j])
                tG_h[i, np.mod(j + 1, 3)] += (dE_dhp[i, j]) @ (dhCWj_dRj[i, j])



    # ##Forces for each cell if cells meet at a power vertex, for cases ∂theta/dhv (i.e. = dhp)
    # tG_v_h = np.zeros((nv, 3))
    # for i in range(nv):
    #     for j in range(3):
    #         tG_v_h[i, j] += (dE_dhv[i]) @ dhv_dRi[i, j]
    #

    # ##Compile accounting for whether power or circle vertex
    # tG_h = tG_c_h.copy()
    # tG_h[V_in_j] = tG_v_h[V_in_j]

    LCb = _add_boundary_naughts(LC)
    tLCb = trf.tri_call(LCb,tri)


    tG_s = dE_dAi*tLCb + dE_dPi*2*np.pi + dE_dMi*tLCb/tR

    # tG_s /= 1.0*trf.assemble_tri(np.ones_like(tri),tri) ##May b

    tG = tG_s + tG_h
    # tG = tG_h
    return -tG

# @jit(nopython=True)
def get_tF_tG(tri, tx, rj, rk, tR, Rj, Rk, hp_j, hm_j, no_touch_j_vec, no_touch_j_mat, V_in_j, A, P, A0, P0, lambda_A,
               lambda_P, lambda_M, Jm, Jp,LC):
    relative_geometries = get_relative_geometries(tx,rj,hp_j,hm_j,no_touch_j_vec)
    dE_dvariables = get_dE_dvariables(relative_geometries, tri, tR, A, P, A0, P0, lambda_A, lambda_P, lambda_M, Jm, Jp)
    differentials = get_jacobians(tx, rj, rk, tR, Rj, Rk, no_touch_j_mat)
    tF = _get_tF(relative_geometries,dE_dvariables,differentials,tri, tx, rj, rk, tR, Rj, Rk, hp_j, hm_j, no_touch_j_vec, no_touch_j_mat, V_in_j, A, P, A0, P0, lambda_A,
               lambda_P, lambda_M, Jm, Jp,LC)
    tG = _get_tG(relative_geometries,dE_dvariables,differentials,tri, tx, rj, rk, tR, Rj, Rk, hp_j, hm_j, no_touch_j_vec, no_touch_j_mat, V_in_j, A, P, A0, P0, lambda_A,
               lambda_P, lambda_M, Jm, Jp,LC)
    return tF,tG


@jit(nopython=True)
def get_J(W, tc_types, neigh_tctypes, nc_types):
    return W.take(tc_types.ravel() + nc_types * neigh_tctypes.ravel()).reshape(-1, 3)


@jit(nopython=True)
def _add_boundary_naughts(val,n_boundary=4):
    out = np.zeros(val.size+n_boundary)
    out[:-n_boundary] = val
    return out

