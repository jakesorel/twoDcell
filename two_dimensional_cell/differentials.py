import numpy as np
import two_dimensional_cell.tri_functions as trf
# from two_dimensional_cell.tri_functions import _tcross, _tdot,_roll,_roll3,_touter,_tidentity,_tnorm,_tmatmul,_replace_val
from numba import jit
#
# @jit(nopython=True)
# def _power_vertex_differentials(tS,tR, rij,rik,riV):
#     rj,rk = _roll3(tS,1),_roll3(tS,-1)
#     Rj,Rk = _roll(tR,1),_roll(tR,-1)
#     rjk = _roll3(rij,1)
#     cross = _tcross(tS,rj)+_tcross(rj,rk)+_tcross(rk,tS)
#     rjkz = np.dstack((-rjk[:,:,1],rjk[:,:,0]))
#     mult = np.dstack((rjk[:,:,1],rjk[:,:,0]))
#     first_term = 1/(2*(cross)**2)*rj[:,:,1] * (tR**2 - Rk**2 - rij[:,:,0]*rik[:,:,0])
#     second_term = cross * (rij[:,:,0]+rik[:,:,0])
#     third_term = (tR**2 - Rj**2 + rj[:,:,1]**2 - rij[:,:,0]*rik[:,:,0])*rk[:,:,1]**2
#     fourth_term = (np.dstack((rjk[:,:,1],-rjk[:,:,0])).T * (tS[:,:,1].T**2)).T
#     fifth_term = (rj[:,:,1]*rk[:,:,1]**2).reshape(-1,3,1) *np.array((((-1,1),),))
#     sixth_term = (tS[:,:,1]*(Rj**2 - rj[:,:,1]**2 - Rk**2 + rk[:,:,1]**2)).reshape(-1,3,1) *np.array((((-1,1),),))
#     dhv_dri = (mult.T *(first_term.T + second_term.T + third_term.T + fourth_term.T + fifth_term.T + sixth_term.T)).T
#     # numerator = _touter(rjkz,riV) #### rjkz[:,:,:,np.newaxis]*riV[:,:,np.newaxis,:]
#     # dhv_dri =  (numerator.T/cross.T).T
#     dhv_dRi = (tR.T*rjkz.T / cross.T).T
#     return dhv_dri,dhv_dRi


@jit(nopython=True)
def _power_vertex_differentials(ri,rj,rk,Ri,Rj,Rk):
    # ri, rj, rk = geom.tS, geom.rj, geom.rk
    # Ri,Rj,Rk = geom.tR,geom.Rj,geom.Rk
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

    dhv_dRi = np.stack(((Ri*(rjy - rky))/(riy*(rjx - rkx) + rjy*rkx - rjx*rky + rix*(-rjy + rky)),(Ri*(rjx - rkx))/(-(rjy*rkx) + riy*(-rjx + rkx) + rix*(rjy - rky) + rjx*rky))).T

    return dhv_dri,dhv_dRi

@jit(nopython=True)
def dhdr(rijk):
    """
    Calculates ∂h_j/dr_i the Jacobian for all cells in each triangulation
    Last two dims: ((dhx/drx,dhx/dry),(dhy/drx,dhy/dry))
    These are lifted from Mathematica
    :param rijk_: (n_v x 3 x 2) np.float32 array of cell centroid positions for each cell in each triangulation (first two dims follow order of triangulation)
    :param vs: (n_v x 2) np.float32 array of vertex positions, corresponding to each triangle in the triangulation
    :param L: Domain size (np.float32)
    :return: Jacobian for each cell of each triangulation (n_v x 3 x 2 x 2) np.float32 array (where the first 2 dims follow the order of the triangulation.
    """
    DHDR = np.empty(rijk.shape + (2,))
    for i in range(3):
        ax,ay = rijk[:,np.mod(i,3),0],rijk[:,np.mod(i,3),1]
        bx, by = rijk[:, np.mod(i+1,3), 0], rijk[:, np.mod(i+1,3), 1]
        cx, cy = rijk[:, np.mod(i+2,3), 0], rijk[:, np.mod(i+2,3), 1]
        #dhx/drx
        DHDR[:, i, 0, 0] = (ax * (by - cy)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((by - cy) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/drx
        DHDR[:, i, 1,0] = (bx ** 2 + by ** 2 - cx ** 2 + 2 * ax * (-bx + cx) - cy ** 2) / (
                    2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((by - cy) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhx/dry
        DHDR[:, i, 0, 1] = (-bx ** 2 - by ** 2 + cx ** 2 + 2 * ay * (by - cy) + cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((-bx + cx) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/dry
        DHDR[:, i, 1,1] = (ay * (-bx + cx)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((-bx + cx) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)


    return DHDR

@jit(nopython=True)
def _circle_vertex_differentials(ri,rj,Ri,Rj,no_touch_mat):
    ###ORIGINAL
    # ri,rj,Ri,Rj = geom.tS,_roll3(geom.tS,1),geom.tR,_roll(geom.tR,1)
    (rix, riy), (rjx, rjy), Ri, Rj = ri.T, rj.T, Ri.T, Rj.T
    da_drix = np.stack((0.5 - ((Ri ** 2 - Rj ** 2) * (rix - rjx) * (-rix + rjx)) / (
                (rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2 - (Ri ** 2 - Rj ** 2) / (
                                    2. * ((rix - rjx) ** 2 + (riy - rjy) ** 2)),
                        -(((Ri ** 2 - Rj ** 2) * (rix - rjx) * (-riy + rjy)) / (
                                    (rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2)))
    da_driy = np.stack(
        (-(((Ri ** 2 - Rj ** 2) * (-rix + rjx) * (riy - rjy)) / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2),
         0.5 - (Ri ** 2 - Rj ** 2) / (2. * ((rix - rjx) ** 2 + (riy - rjy) ** 2)) - (
                     (Ri ** 2 - Rj ** 2) * (riy - rjy) * (-riy + rjy)) / (
                     (rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2))
    db_drix = np.stack(((0.25 * (
                (4 * (Ri ** 2 - Rj ** 2) ** 2 * (rix - rjx)) / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 3 - (
                    4 * (Ri ** 2 + Rj ** 2) * (rix - rjx)) / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2) * (
                                     -riy + rjy)) / np.sqrt(
        -1 - (Ri ** 2 - Rj ** 2) ** 2 / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2 + (2 * (Ri ** 2 + Rj ** 2)) / (
                    (rix - rjx) ** 2 + (riy - rjy) ** 2)), (0.25 * (rix - rjx) * (
                (4 * (Ri ** 2 - Rj ** 2) ** 2 * (rix - rjx)) / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 3 - (
                    4 * (Ri ** 2 + Rj ** 2) * (rix - rjx)) / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2)) / np.sqrt(
        -1 - (Ri ** 2 - Rj ** 2) ** 2 / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2 + (2 * (Ri ** 2 + Rj ** 2)) / (
                    (rix - rjx) ** 2 + (riy - rjy) ** 2)) + 0.5 * np.sqrt(
        -1 - (Ri ** 2 - Rj ** 2) ** 2 / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2 + (2 * (Ri ** 2 + Rj ** 2)) / (
                    (rix - rjx) ** 2 + (riy - rjy) ** 2))))
    db_driy = np.stack((-0.5 * np.sqrt(
        -1 - (Ri ** 2 - Rj ** 2) ** 2 / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2 + (2 * (Ri ** 2 + Rj ** 2)) / (
                    (rix - rjx) ** 2 + (riy - rjy) ** 2)) + (0.25 * (
                (4 * (Ri ** 2 - Rj ** 2) ** 2 * (riy - rjy)) / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 3 - (
                    4 * (Ri ** 2 + Rj ** 2) * (riy - rjy)) / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2) * (
                                                                         -riy + rjy)) / np.sqrt(
        -1 - (Ri ** 2 - Rj ** 2) ** 2 / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2 + (2 * (Ri ** 2 + Rj ** 2)) / (
                    (rix - rjx) ** 2 + (riy - rjy) ** 2)), (0.25 * (rix - rjx) * (
                (4 * (Ri ** 2 - Rj ** 2) ** 2 * (riy - rjy)) / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 3 - (
                    4 * (Ri ** 2 + Rj ** 2) * (riy - rjy)) / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2)) / np.sqrt(
        -1 - (Ri ** 2 - Rj ** 2) ** 2 / ((rix - rjx) ** 2 + (riy - rjy) ** 2) ** 2 + (2 * (Ri ** 2 + Rj ** 2)) / (
                    (rix - rjx) ** 2 + (riy - rjy) ** 2))))
    da_dri, db_dri = np.stack((da_drix, da_driy)).T, np.stack((db_drix, db_driy)).T
    dhCCW_dri, dhCW_dri = da_dri - db_dri, da_dri + db_dri
    dhCCW_dri = trf.replace_val(dhCCW_dri,no_touch_mat,0)
    dhCW_dri = trf.replace_val(dhCW_dri, no_touch_mat, 0)

    ##new for ∂h/∂R, not checked yet.
    d22 = (rix - rjx)**2 + (riy-rjy)**2
    da_dRi = np.stack(((Ri*(-rix + riy))/d22,(Ri*(-rjx + rjy))/d22)).T
    db_dRi =np.stack(((-1.*Ri*(d22 - 1.*Ri**2 + Rj**2)*(riy - 1.*rjy))/(d22**2*np.sqrt(-(((d22 - Ri**2)**2 - 2*(d22 + Ri**2)*Rj**2 + Rj**4)/d22**2))),(1.*Ri*(d22 - 1.*Ri**2 + Rj**2)*(rix - 1.*rjx))/(d22**2*np.sqrt(-(((d22 - Ri**2)**2 - 2*(d22 + Ri**2)*Rj**2 + Rj**4)/d22**2))))).T

    dhCCW_dRi,dhCW_dRi = da_dRi - db_dRi, da_dRi + db_dRi
    dhCCW_dRi = trf.replace_val(dhCCW_dRi,no_touch_mat[:,:,:,0],0)
    dhCW_dRi = trf.replace_val(dhCW_dRi, no_touch_mat[:,:,:,0], 0)
    return dhCCW_dri, dhCW_dri,dhCCW_dRi,dhCW_dRi
#

#
# #
# # # @jit(nopython=True)
# def _circle_vertex_differentials_alt(ris,rjs,Ris,Rjs,no_touch_mat):
#     # ri,rj,Ri,Rj = geom.tS,_roll3(geom.tS,1),geom.tR,_roll(geom.tR,1)
#
#     # (rix, riy), (rjx, rjy), Ri, Rj = ri.T, rj.T, Ri.T, Rj.T
#     Sqrt = np.sqrt
#
#     dhCCWj_dri = np.zeros((ris.shape[0],3,2,2))
#     dhCCWj_drj = np.zeros((ris.shape[0], 3, 2, 2))
#     for ti,i in np.array(np.nonzero(no_touch_mat)).T:
#         (rix, riy), (rjx, rjy), Ri, Rj = ris[ti,i], rjs[ti,i], Ris[ti,i], Rjs[ti,i]
#         dhxdrix = 1 + ((rix - rjx)*(-rix + rjx))/((rix - rjx)**2 + (riy - rjy)**2) - ((rix - rjx)*(-rix + rjx)*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2))/((rix - rjx)**2 + (riy - rjy)**2)**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)/(2.*((rix - rjx)**2 + (riy - rjy)**2)) + ((rix - rjx)*Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**1.5 - ((rix - rjx)*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(-Ri**2 + Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(riy - rjy))/(2.*((rix - rjx)**2 + (riy - rjy)**2)**2.5*Sqrt(-((Ri**4 + (rix**2 + riy**2 - Rj**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2)**2 - 2*Ri**2*(rix**2 + riy**2 + Rj**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2))/(rix**2 + riy**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2))))
#         dhydrix = ((rix - rjx)**2*Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2))))/((rix - rjx)**2 + (riy - rjy)**2)**1.5 - Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))/Sqrt((rix - rjx)**2 + (riy - rjy)**2) + ((rix - rjx)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2) - ((rix - rjx)*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2 + ((rix - rjx)**2*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(-Ri**2 + Rj**2 + (rix - rjx)**2 + (riy - rjy)**2))/(2.*((rix - rjx)**2 + (riy - rjy)**2)**2.5*Sqrt(-((Ri**4 + (rix**2 + riy**2 - Rj**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2)**2 - 2*Ri**2*(rix**2 + riy**2 + Rj**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2))/(rix**2 + riy**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2))))
#         dhxdriy = Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))/Sqrt((rix - rjx)**2 + (riy - rjy)**2) + ((-rix + rjx)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2) - ((-rix + rjx)*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2 - ((-(((Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)) + ((Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2*(riy - rjy))/(2.*((rix - rjx)**2 + (riy - rjy)**2)**2))*(-riy + rjy))/(2.*Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))*Sqrt((rix - rjx)**2 + (riy - rjy)**2)) + (Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))*(riy - rjy)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**1.5
#         dhydriy =-(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)/(2.*((rix - rjx)**2 + (riy - rjy)**2)) + ((rix - rjx)*Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**1.5 + ((riy - rjy)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2) - ((Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(riy - rjy)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2 + ((rix - rjx)*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(-Ri**2 + Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(riy - rjy))/(2.*((rix - rjx)**2 + (riy - rjy)**2)**2.5*Sqrt(-((Ri**4 + (rix**2 + riy**2 - Rj**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2)**2 - 2*Ri**2*(rix**2 + riy**2 + Rj**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2))/(rix**2 + riy**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2))))
#         dhxdrjx = -(((rix - rjx)*(-rix + rjx))/((rix - rjx)**2 + (riy - rjy)**2)) + ((rix - rjx)*(-rix + rjx)*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2))/((rix - rjx)**2 + (riy - rjy)**2)**2 + (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)/(2.*((rix - rjx)**2 + (riy - rjy)**2)) - ((rix - rjx)*Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**1.5 - ((((rix - rjx)*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2))/((rix - rjx)**2 + (riy - rjy)**2) - ((rix - rjx)*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2)/(2.*((rix - rjx)**2 + (riy - rjy)**2)**2))*(-riy + rjy))/(2.*Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))*Sqrt((rix - rjx)**2 + (riy - rjy)**2))
#         dhydrjx = 1 - ((rix - rjx)**2*Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2))))/((rix - rjx)**2 + (riy - rjy)**2)**1.5 + Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))/Sqrt((rix - rjx)**2 + (riy - rjy)**2) - ((rix - rjx)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2) + ((rix - rjx)*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2 - ((rix - rjx)**2*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(-Ri**2 + Rj**2 + (rix - rjx)**2 + (riy - rjy)**2))/(2.*((rix - rjx)**2 + (riy - rjy)**2)**2.5*Sqrt(-((Ri**4 + (rix**2 + riy**2 - Rj**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2)**2 - 2*Ri**2*(rix**2 + riy**2 + Rj**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2))/(rix**2 + riy**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2))))
#         dhxdrjy = -(Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))/Sqrt((rix - rjx)**2 + (riy - rjy)**2)) - ((-rix + rjx)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2) + ((-rix + rjx)*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2 - ((((Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2) - ((Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2*(riy - rjy))/(2.*((rix - rjx)**2 + (riy - rjy)**2)**2))*(-riy + rjy))/(2.*Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))*Sqrt((rix - rjx)**2 + (riy - rjy)**2)) - (Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))*(riy - rjy)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**1.5
#         dhydrjy = (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)/(2.*((rix - rjx)**2 + (riy - rjy)**2)) + ((-rix + rjx)*Sqrt(Ri**2 - (Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)**2/(4.*((rix - rjx)**2 + (riy - rjy)**2)))*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**1.5 + ((Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(riy - rjy)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2 + (riy - rjy)**2/(rix**2 + riy**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2) - ((rix - rjx)*(Ri**2 - Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(-Ri**2 + Rj**2 + (rix - rjx)**2 + (riy - rjy)**2)*(riy - rjy))/(2.*((rix - rjx)**2 + (riy - rjy)**2)**2.5*Sqrt(-((Ri**4 + (rix**2 + riy**2 - Rj**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2)**2 - 2*Ri**2*(rix**2 + riy**2 + Rj**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2))/(rix**2 + riy**2 - 2*rix*rjx + rjx**2 - 2*riy*rjy + rjy**2))))
#         dhCCWj_dri[ti,i] = np.stack((np.stack((dhxdrix,dhxdriy)),np.stack((dhydrix,dhydriy))))
#         dhCCWj_drj[ti,i] = np.stack((np.stack((dhxdrjx,dhxdrjy)),np.stack((dhydrjx,dhydrjy))))
#     return dhCCWj_dri,dhCCWj_drj
# #
# def _circle_vertex_differentials(R2ij,tR,rij,nrij,n_v,no_touch_mat,no_touch_vec):
#     """
#     Some of this is wasted given many entries are not used in the end. May be more effiicent to calculate on element-by-element basis
#     :param R2ij:
#     :param tS:
#     :param tR:
#     :param rij:
#     :param nrij:
#     :param riV:
#     :param nv:
#     :return:
#     """
#     # outer_rij_rij = _touter(rij,rij).T ##Check this again coz. function has been fixed
#     # rijz = np.dstack((-rij[:,:,1],rij[:,:,0]))
#     # outer_rij_rijz = _touter(rij,rijz).T
#     outer_rij_rij = _touter(rij,rij).transpose(0,1,3,2).T ##Check this again coz. function has been fixed
#     rijz = np.dstack((-rij[:,:,1],rij[:,:,0]))
#     outer_rij_rijz = _touter(rij,rijz).transpose(0,1,3,2).T
#
#
#     nrijT = nrij.T
#     R2ijT = R2ij.T
#     rijT = rij.T
#
#
#
#     I = _tidentity(n_v)
#
#     IT = I.T
#     rotate_mat = np.zeros_like(I)
#     rotate_mat[:,:,1,0] = 1
#     rotate_mat[:,:,0,1] = -1
#
#
#     Rj = _roll(tR,1)
#
#     a = 0.5 - R2ij/(2*nrij**2)
#     b = 0.5*np.sqrt(2*(tR**2 + Rj**2)/(nrij**2) - R2ij**2/nrij**4 - 1)
#
#
#
#     aT,bT = a.T,b.T
#
#
#     a_terms_ri = R2ijT/nrijT**4 * outer_rij_rij + \
#                 aT *IT
#     b_terms_ri = R2ijT/(2*bT) * (1/nrijT**6 - 1/nrijT**4)*outer_rij_rijz + \
#                 bT * rotate_mat.T
#
#     dhCCW_dri = a_terms_ri + b_terms_ri
#
#     dhCW_dri = a_terms_ri - b_terms_ri
#
#     dhCCW_dRi = tR.T/(nrijT**2)* (rijz.T/(2*bT) * (1 - R2ijT/(nrijT**2)) - rijT)
#     dhCW_dRi = tR.T/(nrijT**2)* (-rijz.T/(2*bT) * (1 - R2ijT/(nrijT**2)) - rijT)
#
#
#     #set differentials to 0 if is nan (i.e. is a type -1 edge)
#     dhCCW_dri, dhCW_dri, dhCCW_dRi, dhCW_dRi = dhCCW_dri.T, dhCW_dri.T, dhCCW_dRi.T, dhCW_dRi.T
#     dhCCW_dri = _replace_val(dhCCW_dri,no_touch_mat,0)
#     dhCW_dri = _replace_val(dhCW_dri, no_touch_mat, 0)
#     dhCCW_dRi = _replace_val(dhCCW_dRi, no_touch_vec, 0)
#     dhCW_dRi = _replace_val(dhCW_dRi, no_touch_vec, 0)
#     return dhCCW_dri, dhCW_dri, dhCCW_dRi, dhCW_dRi
#

@jit(nopython=True)
def _dtheta_dh(hm,hp):
    """
    theta = atan2(hm x hp, hm . hp)

    :param hm:
    :param hp:
    :return:
    """
    hmz = np.dstack((-hm[:,:,1],hm[:,:,0]))
    hpz = np.dstack((-hp[:,:,1],hp[:,:,0]))
    denominator = trf.tnorm(hm)**2 + trf.tnorm(hp)**2
    return (hmz.T/denominator.T).T,(hpz.T/denominator.T).T

@jit(nopython=True)
def _compile_chain2(dadx1,dx1db,dadx2,dx2db):
    return trf.tmatmul(dadx1,dx1db) + trf.tmatmul(dadx2,dx2db)

@jit(nopython=True)
def _compile_alt_thetas(mask,true_val,false_val):
    maskT = mask.T
    return (maskT * true_val.T + ~maskT*false_val.T).T



#
# def _dphi_dri(hv,hvm1,tS,tR,rij,riV,neighbours):
#     """
#     May need to play around to make jitted. But fast enough for now
#
#     :param hv:
#     :param hvm1:
#     :param tS:
#     :param tR:
#     :param rij:
#     :param riV:
#     :param neighbours:
#     :return:
#     """
#     dphi_dhvm1,dphi_dhv = _dtheta_dh(hvm1,hv)
#     dhv_dri,dhv_dRi = _power_vertex_differentials(tS, tR, rij, riV)
#     dhvm1_dri = np.dstack((dhv_dri[:,:,0][_roll(neighbours, -1)],dhv_dri[:,:,1][_roll(neighbours, -1)]))
#     dhvm1_dRi = np.dstack((dhv_dRi[:,:,0][_roll(neighbours, -1)],dhv_dRi[:,:,1][_roll(neighbours, -1)]))
#     dphi_dri = _tmatmul(dphi_dhv,dhv_dri) + _tmatmul(dphi_dhvm1,dhvm1_dri)
#     dphi_dRi = _tmatmul(dphi_dhv,dhv_dRi) + _tmatmul(dphi_dhvm1,dhvm1_dRi)
#
