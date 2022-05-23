
@jit(nopython=True)
def _circle_vertex_differentials(ri,rj,radius,no_touch_mat):
    ###OradiusGINAL
    # radius,radius,radius,radius = geom.tS,_roll3(geom.tS,1),geom.tR,_roll(geom.tR,1)
    (r_ix, r_iy), (r_jx, r_jy) = ri.T, rj.T
    da_dr_ix = np.stack((0.5,0))
    da_dr_iy = np.stack((0,0.5))
    db_dr_ix = np.stack(((0.25 * ( - (4 * (radius ** 2 + radius ** 2) * (r_ix - r_jx)) / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 2) * (
                                     -r_iy + r_jy)) / np.sqrt(
        -1 - (radius ** 2 - radius ** 2) ** 2 / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 2 + (2 * (radius ** 2 + radius ** 2)) / (
                    (r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2)), (0.25 * (r_ix - r_jx) * (
                (4 * (radius ** 2 - radius ** 2) ** 2 * (r_ix - r_jx)) / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 3 - (
                    4 * (radius ** 2 + radius ** 2) * (r_ix - r_jx)) / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 2)) / np.sqrt(
        -1 - (radius ** 2 - radius ** 2) ** 2 / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 2 + (2 * (radius ** 2 + radius ** 2)) / (
                    (r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2)) + 0.5 * np.sqrt(
        -1 - (radius ** 2 - radius ** 2) ** 2 / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 2 + (2 * (radius ** 2 + radius ** 2)) / (
                    (r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2))))
    db_dr_iy = np.stack((-0.5 * np.sqrt(
        -1 - (radius ** 2 - radius ** 2) ** 2 / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 2 + (2 * (radius ** 2 + radius ** 2)) / (
                    (r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2)) + (0.25 * (
                (4 * (radius ** 2 - radius ** 2) ** 2 * (r_iy - r_jy)) / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 3 - (
                    4 * (radius ** 2 + radius ** 2) * (r_iy - r_jy)) / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 2) * (
                                                                         -r_iy + r_jy)) / np.sqrt(
        -1 - (radius ** 2 - radius ** 2) ** 2 / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 2 + (2 * (radius ** 2 + radius ** 2)) / (
                    (r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2)), (0.25 * (r_ix - r_jx) * (
                (4 * (radius ** 2 - radius ** 2) ** 2 * (r_iy - r_jy)) / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 3 - (
                    4 * (radius ** 2 + radius ** 2) * (r_iy - r_jy)) / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 2)) / np.sqrt(
        -1 - (radius ** 2 - radius ** 2) ** 2 / ((r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2) ** 2 + (2 * (radius ** 2 + radius ** 2)) / (
                    (r_ix - r_jx) ** 2 + (r_iy - r_jy) ** 2))))
    da_dradius, db_dradius = np.stack((da_dr_ix, da_dr_iy)).T, np.stack((db_dr_ix, db_dr_iy)).T
    dhCCW_dradius, dhCW_dradius = da_dradius - db_dradius, da_dradius + db_dradius
    dhCCW_dradius = _replace_val(dhCCW_dradius,no_touch_mat,0)
    dhCW_dradius = _replace_val(dhCW_dradius, no_touch_mat, 0)

    ##new for ∂h/∂R, not checked yet.
    d22 = (r_ix - r_jx)**2 + (r_iy-r_jy)**2
    da_dradius = np.stack(((radius*(-r_ix + r_iy))/d22,(radius*(-r_jx + r_jy))/d22)).T
    db_dradius =np.stack(((-1.*radius*(d22 - 1.*radius**2 + radius**2)*(r_iy - 1.*r_jy))/(d22**2*np.sqrt(-(((d22 - radius**2)**2 - 2*(d22 + radius**2)*radius**2 + radius**4)/d22**2))),(1.*radius*(d22 - 1.*radius**2 + radius**2)*(r_ix - 1.*r_jx))/(d22**2*np.sqrt(-(((d22 - radius**2)**2 - 2*(d22 + radius**2)*radius**2 + radius**4)/d22**2))))).T

    dhCCW_dradius,dhCW_dradius = da_dradius - db_dradius, da_dradius + db_dradius
    dhCCW_dradius = _replace_val(dhCCW_dradius,no_touch_mat[:,:,:,0],0)
    dhCW_dradius = _replace_val(dhCW_dradius, no_touch_mat[:,:,:,0], 0)
    return dhCCW_dradius, dhCW_dradius,dhCCW_dradius,dhCW_dradius
#
