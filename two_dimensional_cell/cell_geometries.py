import numpy as np
from numba import jit

from two_dimensional_cell.differentials import _power_vertex_differentials, _circle_vertex_differentials
# from two_dimensional_cell.tri_functions import _roll, _roll3, _tnorm, _tri_sum, _CV_matrix, _get_neighbours, \
#     _replace_val, _tdot, _tcross, _repeat_mat, _repeat_vec, \
#     _replace_vec

import two_dimensional_cell.tri_functions as trf

from scipy.spatial.distance import cdist


class geometries:
    def __init__(self, S, R, V, tri_list, n_v, n_b):

        # # 1. Extract triangulation information to restructure the data
        # self.S, self.R, self.V, self.tri_list, self.n_v = S, R, V, tri_list, n_v
        # self.tV = _tV(self.V)
        # self.n_c = self.R.size
        # self.n_b = n_b
        # self.tS, self.tR = _triangulated_form2(S, tri_list), _triangulated_form(R, tri_list)
        # self.CV_matrix = trf.CV_matrix(self.tri_list, self.n_v, self.n_c)
        # self.rj = trf.roll3(self.tS, 1)
        # self.Rj = trf.roll(self.tR, 1)
        # self.rk = trf.roll3(self.tS, -1)
        # self.Rk = trf.roll(self.tR, -1)
        #
        # # 2. Get neighbours, and corresponding vertices
        # self.neighbours, self.ls = trf.get_neighbours(self.tri_list)
        # self.jneighbours = trf.roll(self.neighbours, -1)
        # self.kneighbours = trf.roll(self.neighbours, +1)
        # # self.jls = _roll(self.ls,-1)
        # self.vj_neighbours = self.V[self.jneighbours]  # indexed so that index i is the vertex opposite the tri edge ij
        # self.vk_neighbours = self.V[self.kneighbours]
        #
        # # 3. Measure centroid-vertex, vertex-vertex etc. vectors and distances
        # self.riV = _riV(self.tV, self.tS)
        # self.nriV = trf.tnorm(self.riV)
        #
        # self.rij, self.rik = _rij(self.tS, 1), _rij(self.tS, -1)
        # self.nrij, self.nrik = trf.tnorm(self.rij), trf.tnorm(self.rik)
        #
        # self.riVj_neighbours = _riV(self.vj_neighbours, self.tS)
        # self.nriVj_neighbours = trf.tnorm(self.riVj_neighbours)
        #
        # self.riVk_neighbours = _riV(self.vk_neighbours, self.tS)
        # self.nriVk_neighbours = trf.tnorm(self.riVk_neighbours)
        #
        # self.V_Vj = _V_Vj(self.tV, self.vj_neighbours)
        # self.nV_Vj = trf.tnorm(self.V_Vj)
        #
        # self.Vk_V = _V_Vj(self.vk_neighbours, self.tV)
        # self.nVk_V = trf.tnorm(self.Vk_V)
        #
        # # 4. Calculate the distance of the bisector of a pair of circles (i--j) from the centroid  (i)
        # self.R2ij, self.R2ik = _R2ij(self.tR, 1), _R2ij(self.tR, -1)
        # self.dij, self.dik = _dij(self.nrij, self.R2ij), _dij(self.nrik, self.R2ik)
        #
        # # 5. Classify edges
        # # self.V_in_j,self.V_out_j,self.no_touch_j = _classify_edges(self.riV,self.nriV,self.riVj_neighbours,self.nriVj_neighbours,self.V_Vj,self.tR,self.dij)
        # # self.V_in_k,self.V_out_k,self.no_touch_k = _classify_edges(self.riVk_neighbours,self.nriVk_neighbours,self.riV,self.nriV,self.Vk_V,self.tR,self.dik)
        # self.V_in_j, self.V_out_j, self.no_touch_j = _classify_edges(self.nriV, self.nriVj_neighbours, self.tR, self.Rj,
        #                                                              self.nrij)
        # self.V_in_k, self.V_out_k, self.no_touch_k = _classify_edges(self.nriVk_neighbours, self.nriV, self.tR, self.Rk,
        #                                                              self.nrik)
        #
        #
        # # 5. Calculate the three angles
        #
        # self.ttheta_j, self.hm_j, self.hp_j = _ttheta(self.V_in_j, self.V_out_j, self.no_touch_j, self.tR, self.tS,
        #                                               self.tV,
        #                                               self.nrij, self.vj_neighbours)
        # self.ttheta_k, self.hm_k, self.hp_k = _ttheta(self.V_in_k, self.V_out_k, self.no_touch_k, self.tR, self.tS,
        #                                               self.vk_neighbours,
        #                                               self.nrik, self.tV, dir=-1)
        #
        #
        # self.hp_j,self.hm_j,self.ttheta_j,self.no_touch_j = _classification_correction(self.S,self.R,self.n_b,
        #                                                                                self.ttheta_j, self.hm_j,
        #                                                                                self.hp_j,self.no_touch_j,
        #                                                                                self.V_in_j)
        # self.hp_k,self.hm_k,self.ttheta_k,self.no_touch_k = _classification_correction(self.S,self.R,self.n_b,
        #                                                                                self.ttheta_k, self.hm_k,
        #                                                                                self.hp_k,self.no_touch_k,
        #                                                                                self.V_in_k)
        #
        #
        # self.no_touch_j_mat = trf.repeat_mat(self.no_touch_j)
        # self.no_touch_j_vec = trf.repeat_vec(self.no_touch_j)
        #
        # self.no_touch_k_mat = trf.repeat_mat(self.no_touch_k)
        # self.no_touch_k_vec = trf.repeat_vec(self.no_touch_k)
        #
        #
        # self.tphi_j = _tvecangle(self.riVj_neighbours, self.riV)
        # self.tphi_k = _tvecangle(self.riV, self.riVk_neighbours)
        #
        #
        # self.rihm_j, self.rihp_j = _riV(self.hm_j, self.tS), _riV(self.hp_j, self.tS)
        # self.rihm_k, self.rihp_k = _riV(self.hm_k, self.tS), _riV(self.hp_k, self.tS)
        # self.nrihp_j, self.nrihp_k, self.nrihm_j, self.nrihm_k = trf.tnorm(self.rihp_j), trf.tnorm(self.rihp_k), trf.tnorm(
        #     self.rihm_j), trf.tnorm(self.rihm_k)
        #
        # self.hmj_hpj = _V_Vj(self.hm_j, self.hp_j)
        # self.hmk_hpk = _V_Vj(self.hm_k, self.hp_k)
        # self.nhmj_hpj = trf.tnorm(self.hmj_hpj)
        # self.nhmk_hpk = trf.tnorm(self.hmk_hpk)
        #
        # self.tpsi_j = self.tphi_j - self.ttheta_j
        # self.tpsi_k = self.tphi_k - self.ttheta_k
        #
        # # #6. Calculate distances and areas
        # self.tlP, self.tlC = _lP(self.hp_j, self.hm_j), _lC(self.tpsi_j, self.tR)
        # self.tAP, self.tAC = _AP(self.hm_j, self.hp_j, self.tS), _AC(self.tpsi_j, self.tR)
        # self.lP = trf.tri_sum(self.n_c, self.CV_matrix, self.tlP)
        # self.lC = trf.tri_sum(self.n_c, self.CV_matrix, self.tlC)
        # self.AP = trf.tri_sum(self.n_c, self.CV_matrix, self.tAP)
        # self.AC = trf.tri_sum(self.n_c, self.CV_matrix, self.tAC)
        # self.A = self.AP + self.AC
        # self.P = self.lP + self.lC
        #
        # self.lambda_A, self.lambda_P = 1, 1
        # self.lambda_M = 1
        # self.A0, self.P0 = 1, 1

    def get_F(self):
        ##Consider  non power vertex interactions first
        hp_ri = self.hp_j - self.tS
        hm_ri = self.hm_j - self.tS
        hm_ri_z = np.dstack((-hm_ri[:, :, 1], hm_ri[:, :, 0]))
        hp_ri_z = np.dstack((-hp_ri[:, :, 1], hp_ri[:, :, 0]))

        # ##new  feasibly can remove due to degeneracy with the dCCW_dri matrix.
        # hm_ri_z[self.no_touch_j_vec] = 0
        # hp_ri_z[self.no_touch_j_vec] = 0
        # ##new

        nhp_ri = _tnorm(hp_ri)
        nhm_ri = _tnorm(hm_ri)
        dtheta_dhp_cell_i = (hp_ri_z.T / nhp_ri.T ** 2).T
        dtheta_dhp_cell_i[self.no_touch_j_vec] = 0

        hp_rj = self.hp_j - self.rj
        hm_rj = self.hm_j - self.rj
        hp_rj_z = np.dstack((-hp_rj[:, :, 1], hp_rj[:, :, 0]))
        hm_rj_z = np.dstack((-hm_rj[:, :, 1], hm_rj[:, :, 0]))

        # ##new, feasibly can remove due to degeneracy with the dCCW_dri matrix.
        # hm_rj_z[self.no_touch_j_vec] = 0
        # hp_rj_z[self.no_touch_j_vec] = 0
        # ##new

        nhp_rj = _tnorm(hp_rj)
        nhm_rj = _tnorm(hm_rj)
        dtheta_dhp_cell_j = -(hp_rj_z.T / nhp_rj.T ** 2).T
        dtheta_dhp_cell_j[self.no_touch_j_vec] = 0

        hphm = self.hp_j - self.hm_j
        hphm_z = np.dstack((-hphm[:, :, 1], hphm[:, :, 0]))
        nhphm = _tnorm(hphm)
        hphm_unit = (hphm.T / nhphm.T).T
        hphm_unit[self.no_touch_j_vec] = 0

        ##Calculate ∂E/∂Pi and ∂E/∂Ai i.e. for the ith cell, and convert to triangulated form
        dE_dPi = _triangulated_form(self.lambda_P * (self.P - self.P0), self.tri_list)
        dE_dAi = _triangulated_form(self.lambda_A * (self.A - self.A0), self.tri_list)
        dE_dMi = _triangulated_form(self.lambda_M * np.ones_like(self.P), self.tri_list)

        dlCi_dtheta = -self.tR
        dACi_dtheta = -(self.tR ** 2 / 2)

        # Calculate ∂E/dtheta_ij and ∂E/dtheta_ji by summing the contributions
        # Note that roll(theta_ik) = theta_ji
        dEPC_dtheta_ij = dE_dPi * dlCi_dtheta
        dEPC_dtheta_ji = _roll(dE_dPi * dlCi_dtheta, 1)
        dEAC_dtheta_ij = dE_dAi * dACi_dtheta
        dEAC_dtheta_ji = _roll(dE_dAi * dACi_dtheta, 1)
        dEM_dtheta_ij = dE_dMi * dlCi_dtheta
        dEM_dtheta_ik = _roll(dE_dMi * dlCi_dtheta, 1)
        dE_dtheta_ij = dEPC_dtheta_ij + dEAC_dtheta_ij + dEM_dtheta_ij
        dE_dtheta_ji = dEPC_dtheta_ji + dEAC_dtheta_ji + dEM_dtheta_ik

        #####NOTE ^^ and below can be simplified hugely. Do this at some point. rolls can be pulled into a single operation.

        # Calculate ∂E/dhp by chain rule.
        # This statement below is equivalent to: ∂E/∂hpij = ∂Ei/∂hpij + ∂Ej/∂hpij
        # This is because hpij has two contributions: from cell i and from cell j
        dEC_dhp = (dE_dtheta_ij.T * dtheta_dhp_cell_i.T + dE_dtheta_ji.T * dtheta_dhp_cell_j.T).T

        dlPi_dhp = hphm_unit
        dEAP_dhp = 0.5 * ((hm_ri_z.T * dE_dAi.T).T - (_roll(dE_dAi,
                                                            1).T * hm_rj_z.T).T)  ##note the minus sign here, as measuring area CW rather than antiCW, i.e. flipped order in the cross prod.

        dEP_dhp = ((dE_dPi.T + _roll(dE_dPi, 1).T) * dlPi_dhp.T).T + dEAP_dhp

        dE_dhp = dEC_dhp + dEP_dhp

        ##Calculate the jacobians.
        # Some degeneracy here, so can be optimized later
        self.dhCCWj_dri, self.dhCWj_dri, self.dhCCWj_dRi, self.dhCWj_dRi = \
            _circle_vertex_differentials(self.tS, self.rj, self.tR, self.Rj, self.no_touch_j_mat)
        self.dhCCWj_drj, self.dhCWj_drj, self.dhCCWj_dRj, self.dhCWj_dRj = \
            _circle_vertex_differentials(self.rj, self.tS, self.Rj, self.tR, self.no_touch_j_mat)

        # Calculate the forces at each cell (for each triangle contribution) for the terms with ∂theta/∂hp
        self.tF_c_h = np.zeros((self.n_v, 3, 2))
        for i in range(self.n_v):
            for j in range(3):
                self.tF_c_h[i, j] += (dE_dhp[i, j]) @ (self.dhCCWj_dri[i, j])
                self.tF_c_h[i, np.mod(j + 1, 3)] += (dE_dhp[i, j]) @ (self.dhCWj_drj[i, j])

        # Calculate the forces at each cell for terms involving ∂theta/∂ri
        self.tF_s = 0.5 * (dE_dAi.T * hphm_z.T).T + \
                    (dE_dtheta_ij.T * (hm_ri_z.T / nhm_ri.T ** 2 - hp_ri_z.T / nhp_ri.T ** 2)).T
        self.tF_s[np.isnan(self.tF_s)] = 0  ##need as some will be divide by 0

        ##Now do the same for the power vertices
        self.dhv_dri, self.dhv_dRi = _power_vertex_differentials(self.tS, self.rj, self.rk, self.tR, self.Rj, self.Rk)

        # #I think this is correct...
        # ###CHECK THIS
        # Here I am summing ∂E/∂hp for all hps in a triangle to give ∂E/∂hv, given hps are all equal to hv when cells meet at a power vertex
        # #However, I may be double counting (i.e. may need to divide by 2). Need to double check. May be worth writing from scratch and comparing.

        # dE_dhv = dE_dhp.sum(axis=1)

        dE_dhv = dEP_dhp.sum(axis=1)

        ##Forces for each cell if cells meet at a power vertex, for cases ∂theta/dhv (i.e. = dhp)
        self.tF_v_h = np.zeros((self.n_v, 3, 2))
        for i in range(self.n_v):
            for j in range(3):
                self.tF_v_h[i, j] += (dE_dhv[i]) @ self.dhv_dri[i, j]

        ##Compile accounting for whether power or circle vertex
        tF_h = self.tF_c_h.copy()
        tF_h[self.V_in_j] = self.tF_v_h[self.V_in_j]

        tF = tF_h + self.tF_s

        ##Compile forces to cells by summing components from each triangle
        F = np.stack((_tri_sum(self.n_c, self.CV_matrix, tF[:, :, 0]), _tri_sum(self.n_c, self.CV_matrix, tF[:, :, 1])),
                     axis=1)

        # True force is F=- ∂E/∂ri. May want to change around notation so as not to confuse.
        self.FP = -F

        self.tG_c_h = np.zeros((self.n_v, 3))
        for i in range(self.n_v):
            for j in range(3):
                self.tG_c_h[i, j] += (dE_dhp[i, j]) @ (self.dhCCWj_dRi[i, j])
                self.tG_c_h[i, np.mod(j + 1, 3)] += (dE_dhp[i, j]) @ (self.dhCWj_dRj[i, j])

        ##Forces for each cell if cells meet at a power vertex, for cases ∂theta/dhv (i.e. = dhp)
        self.tG_v_h = np.zeros((self.n_v, 3))
        for i in range(self.n_v):
            for j in range(3):
                self.tG_v_h[i, j] += (dE_dhv[i]) @ self.dhv_dRi[i, j]

        ##Compile accounting for whether power or circle vertex
        tG_h = self.tG_c_h.copy()
        tG_h[self.V_in_j] = self.tG_v_h[self.V_in_j]

        ##Compile forces to cells by summing components from each triangle
        G_h = -_tri_sum(self.n_c, self.CV_matrix, tG_h)

        ##a bit repetitious, can clean up later.
        # Clean notation, but will be a tiny bit slower.
        G_s = -(self.lambda_P * (self.P - self.P0) * self.P / self.R + self.lambda_A * (
                    self.A - self.A0) * self.A / self.R + self.lambda_M * self.lC ** 2 / self.R)

        self.G = G_h + G_s


    def get_F_soft(self):
        tF_soft_ij = -self.k*(self.nrij - 2*self.a)*(self.nrij<2*self.a)
        tF_soft_ik = -self.k*(self.nrik - 2*self.a)*(self.nrik<2*self.a)
        tF_soft_ij = (tF_soft_ij.T*self.rij.T/self.nrij.T).T
        tF_soft_ik = (tF_soft_ik.T*self.rik.T/self.nrik.T).T
        tF_soft = tF_soft_ik + tF_soft_ij
        self.F_soft = np.stack((_tri_sum(self.n_c,self.CV_matrix,tF_soft[:,:,0]),_tri_sum(self.n_c,self.CV_matrix,tF_soft[:,:,1])),axis=1)

@jit(nopython=True)
def _R2ij(tR, direc=1):
    return tR ** 2 - _roll(tR ** 2, direc)


@jit(nopython=True)
def _dij(nrij, R2ij):
    return (nrij ** 2 + R2ij) / (2 * nrij)


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


@jit(nopython=True, cache=True)
def _triangulated_form(x, tri_list):
    return x.take(tri_list)


@jit(nopython=True, cache=True)
def _triangulated_form2(x, tri_list):
    return np.dstack((x[:, 0].take(tri_list), x[:, 1].take(tri_list)))


@jit(nopython=True, cache=True)
def _rij(tS, direc=1):
    """
    dir = 1 --> "ij" dir = -1 "ik"
    :param tS:
    :pSaram tR:
    :param V:
    :param tri_list:
    :param dir:
    :return:
    """
    return np.dstack((tS[:, :, 0] - _roll(tS[:, :, 0], direc), tS[:, :, 1] - _roll(tS[:, :, 1], direc)))


@jit(nopython=True)
def _tV(V):
    return np.stack((V, V, V), axis=1)


@jit(nopython=True)
def _riV(tV, tS):
    """
    Vector from hv to ri
    :param tV:
    :param tS:
    :return:
    """
    return tS - tV


@jit(nopython=True)
def _V_Vj(tV, v_neighbours):
    return tV - v_neighbours


@jit(nopython=True)
def _tvecangle(a, b):
    """
    Signed angle between two (triangle form) sets of vectors
    :param a:
    :param b:
    :return:
    """
    return np.arctan2(_tcross(a, b), _tdot(a, b))


@jit(nopython=True)
def _vecangle(a, b):
    """
    Signed angle between two vectors
    :param a:
    :param b:
    :return:
    """
    cross = a[0] * b[1] - b[0] * a[1]
    dot = a[0] * b[0] + a[1] * b[1]
    return np.arctan2(cross, dot)


@jit(nopython=True)
def _classify_edges(nriV, nriVj_neighbours, tR, Rj, nrij):
    # riV,nriV,riVj_neighbours,nriVj_neighbours,V_Vj,tR,dij = self.riV,self.nriV,self.riVj_neighbours,self.nriVj_neighbours,self.V_Vj,self.tR,self.dij
    V_in = (nriV < tR)
    V_out = (nriVj_neighbours < tR)

    no_touch = (2 * (tR ** 2 + Rj ** 2) / nrij ** 2 - ((tR ** 2 - Rj ** 2) ** 2) / nrij ** 4 - 1) < 0

    return V_in, V_out, no_touch


def _classification_correction(r,R,n_b,ttheta, hm, hp,no_touch,V_in_j,err=1e-7):
    """
    Deals with cases where two circles intersect within a 3rd cell
    """
    #r,R,n_b,ttheta, hm, hp,no_touch,V_in_j = self.S,self.R,self.n_b,self.ttheta_j, self.hm_j,self.hp_j,self.no_touch_j,self.V_in_j
    touch_not_power_mask = (~V_in_j)*(~no_touch)
    hp_circ = hp[touch_not_power_mask]
    d = cdist(hp_circ,r[:-n_b])**2 - R[:-n_b]**2
    false_cross = np.count_nonzero(d <= err, axis=1)>2
    touch_not_power_mask[touch_not_power_mask] = false_cross
    no_touch = _replace_val(no_touch,touch_not_power_mask,True)
    no_touch_vec = np.dstack((no_touch, no_touch))
    hp = _replace_val(hp, no_touch_vec, 0)
    hm = _replace_val(hm, no_touch_vec, 0)
    ttheta = _replace_val(ttheta, no_touch, 0)

    return hp,hm,ttheta,no_touch





@jit(nopython=True)
def _ttheta(V_in, V_out, no_touch, tR, tS, tV, nrij, vj_neighbours, dir=1):
    V_in2, V_out2, no_touch2 = np.dstack((V_in, V_in)), np.dstack((V_out, V_out)), np.dstack((no_touch, no_touch))
    start, end = vj_neighbours.copy(), tV.copy()
    h_CCW, h_CW = _tintersections(tS, _roll3(tS, dir), tR, _roll(tR, dir), nrij)

    end = _replace_vec(end, ~V_in2, h_CCW)
    start = _replace_vec(start, ~V_out2, h_CW)
    end = _replace_val(end, no_touch2, 0)
    start = _replace_val(start, no_touch2, 0)

    # ###new line, strange nans appearing, presumably because circles no longer intersect. Feasibly can remove the above statement of no_touch
    # end = _replace_val(end,np.isnan(h_CCW),0)
    # start = _replace_val(start,np.isnan(h_CW),0)
    # ##end

    ttheta = _tvecangle(start - tS, end - tS)
    return ttheta, start, end


@jit(nopython=True, cache=True)
def _lP(start, end):
    return _tnorm(end - start)


@jit(nopython=True, cache=True)
def _lC(tpsi, tR):
    return tpsi * tR


@jit(nopython=True, cache=True)
def _AP(start, end, tS):
    return 0.5 * _tcross(start - tS, end - tS)


@jit(nopython=True, cache=True)
def _AC(tspi, tR):
    return 0.5 * tspi * tR ** 2



"""
These modules below are currently not used... 
"""


@jit(nopython=True, cache=True)
def _angles(nrij):
    a, b, c = nrij, _roll(nrij, 1), _roll(nrij, -1)
    angles = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    return angles


def _alt_pts(int_pts, neighbours, tri_list, ls):
    """
    l is the col index of the opposite cell (i.e. the one not shared)
    :param int_pts:
    :param neighbours:
    :param tri_list:
    :return:
    """
    alt_pts = np.zeros_like(int_pts)
    for ti, tri in enumerate(tri_list):
        for i in range(3):
            k = np.mod(i - 1, 3)
            neigh_i = neighbours[ti, k]
            if neigh_i != -1:
                col_i = np.mod(-ls[ti, k], 3)
                alt_pts[ti, i] = int_pts[neigh_i, col_i]
            else:
                alt_pts[ti, i] = np.nan, np.nan
    return alt_pts


