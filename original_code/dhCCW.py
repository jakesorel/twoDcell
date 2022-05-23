import power_triangulation as pt
from power_triangulation import *
import numpy as np
from cell_geometries import geometries
import time
import matplotlib.pyplot as plt
from tri_functions import hexagonal_lattice,square_boundary
from tri_functions import _roll,_roll3,_tnorm,_get_b_edges,_tri_sum,normalise,_cosine_rule
from cell_geometries import _angles
from scipy.spatial.distance import cdist
import os
from tri_functions import _replace_val,_tdot

xlim,ylim = (-20,20),(-20,20)
S = hexagonal_lattice(6,6,0.1)
Sb = square_boundary(xlim,ylim)
n_b = Sb.shape[0]
S = np.vstack((S,Sb))

# R = np.ones_like(S[:, 0]) *0.1 # 3 * np.random.random(sample_count) + .2
# R = np.random.uniform(0.05,0.1,S.shape[0])+0.6
plot_name = "plots%.3f"%time.time()
os.mkdir(plot_name)
R = np.ones_like(S[:, 0])*1
nt = 100
S_save = np.zeros((nt,S.shape[0],2))
save_skip = 20
P_save = np.zeros((nt,S.shape[0]-n_b))
tri_list, V, n_v = pt.get_power_triangulation(S, R)
geom = geometries(S, R, V, tri_list, n_v,n_b=n_b)
geom.P0 = 3.4
geom.differential_matrices()

from cell_geometries import _tintersections

@jit(nopython=True)
def _tintersectionsnew(tS,rij,nrij,Ri,dij):
    a = tS.T+-(rij.T)/(nrij.T) * dij.T
    b = np.dstack((rij[:,:,1],-rij[:,:,0])).T/(nrij.T) * np.sqrt(Ri.T**2 - dij.T**2)
    return (a+b).T,(a-b).T

def _intersections(ri,rj,Ri,Rj):
    rij = ri - rj

    nrij = np.linalg.norm(rij)

    # dij = (nrij ** 2 + (Ri**2 - Rj**2)) / (2 * nrij)
    a = 0.5 * (ri + rj) + (Ri ** 2 - Rj ** 2) / (2 * nrij ** 2) * (rj - ri)
    b = 0.5 * np.sqrt(
        2 * (Ri ** 2 + Rj ** 2) / nrij ** 2 - ((Ri ** 2 - Rj ** 2) ** 2) / nrij ** 4 - 1) * np.stack((rj[1] - ri[1], ri[0] - rj[0]))
    # a,b = a.T,b.T
    # a = ri.T - rij.T/nrij.T * dij.T
    # b = np.array((rij[1],-rij[0])).T/nrij.T * np.sqrt(Ri.T**2 - dij.T**2)
    return (a - b), (a + b)


h_CCW, h_CW = _tintersectionsnew(geom.tS,geom.rij,geom.nrij,geom.tR,geom.dij)
dx = 0.001
nx = 30
x_space = np.arange(-dx*nx/2,dx*nx/2,dx)

ti = 8
ri,rj,rk = geom.tS[ti]
Ri,Rj,Rk = geom.tR[ti]
X,Y = np.meshgrid(ri[0]+x_space,ri[1]+x_space,indexing="ij")

h_CW_mat = np.zeros((nx,nx,2))
h_CCW_mat = np.zeros((nx,nx,2))

for i in range(nx):
    for j in range(nx):
        ri = np.array((X[i,j],Y[i,j]))
        h_CCW_mat[i,j],h_CW_mat[i,j] = _intersections(ri, rj, Ri, Rj)

dh_CW_mat_dx_num = (np.roll(h_CW_mat,-1,axis=0) - h_CW_mat)/dx
dh_CW_mat_dy_num = (np.roll(h_CW_mat,-1,axis=1) - h_CW_mat)/dx
dh_CCW_mat_dx_num = (np.roll(h_CCW_mat,-1,axis=0) - h_CCW_mat)/dx
dh_CCW_mat_dy_num = (np.roll(h_CCW_mat,-1,axis=1) - h_CCW_mat)/dx

# plt.imshow(dh_CW_mat_dx[1:-2,1:-2,1])
# plt.show()


# def _dhCW_dri(ri,rj,Ri,Rj):
#     rij = ri - rj
#     nrij = np.linalg.norm(rij)
#     dij = (nrij ** 2 + (Ri**2 - Rj**2)) / (2 * nrij)
#     rij_z = np.array((-rij[1],rij[0]))
#     dadri = 1/nrij**3 * np.outer(rij,rij_z)*(np.sqrt(Ri**2 - dij**2) + (nrij - dij)/(2*nrij**3 * np.sqrt(Ri**2 - dij**2))) + np.sqrt(Ri**2 - dij**2)*np.array(((0,1),(-1,0)))
#     dbdri = np.identity(2)*(1 - dij/nrij) + (1/nrij**2)*np.array(((rij[0]**2 * (2*dij/nrij - 1),Ri**2 - Rj**2 + rij[0]*rij[1]),(Ri**2 - Rj**2 + rij[0]*rij[1],rij[1]**2 * (2*dij/nrij - 1))))
#     return dadri + dbdri, dadri - dbdri
#

def dh_dri(ri,rj,Ri,Rj):
    (rix,riy),(rjx,rjy) = ri,rj
    da_drix = np.array((0.5 - ((Ri**2 - Rj**2)*(rix - rjx)*(-rix + rjx))/((rix - rjx)**2 + (riy - rjy)**2)**2 - (Ri**2 - Rj**2)/(2.*((rix - rjx)**2 + (riy - rjy)**2)),
                                -(((Ri**2 - Rj**2)*(rix - rjx)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2)))
    da_driy = np.array((-(((Ri**2 - Rj**2)*(-rix + rjx)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2),
                                0.5 - (Ri**2 - Rj**2)/(2.*((rix - rjx)**2 + (riy - rjy)**2)) - ((Ri**2 - Rj**2)*(riy - rjy)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2))
    db_drix =np.array(((0.25*((4*(Ri**2 - Rj**2)**2*(rix - rjx))/((rix - rjx)**2 + (riy - rjy)**2)**3 - (4*(Ri**2 + Rj**2)*(rix - rjx))/((rix - rjx)**2 + (riy - rjy)**2)**2)*(-riy + rjy))/np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2)),(0.25*(rix - rjx)*((4*(Ri**2 - Rj**2)**2*(rix - rjx))/((rix - rjx)**2 + (riy - rjy)**2)**3 - (4*(Ri**2 + Rj**2)*(rix - rjx))/((rix - rjx)**2 + (riy - rjy)**2)**2))/np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2)) + 0.5*np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2))))
    db_driy =np.array((-0.5*np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2)) + (0.25*((4*(Ri**2 - Rj**2)**2*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**3 - (4*(Ri**2 + Rj**2)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2)*(-riy + rjy))/np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2)),(0.25*(rix - rjx)*((4*(Ri**2 - Rj**2)**2*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**3 - (4*(Ri**2 + Rj**2)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2))/np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2))))
    da_dri,db_dri = np.stack((da_drix,da_driy)).T,np.stack((db_drix,db_driy)).T
    return da_dri-db_dri, da_dri+db_dri

@jit(nopython=True)
def tdh_dri(ri,rj,Ri,Rj):
    (rix,riy),(rjx,rjy),Ri,Rj = ri.T,rj.T,Ri.T,Rj.T
    da_drix = np.stack((0.5 - ((Ri**2 - Rj**2)*(rix - rjx)*(-rix + rjx))/((rix - rjx)**2 + (riy - rjy)**2)**2 - (Ri**2 - Rj**2)/(2.*((rix - rjx)**2 + (riy - rjy)**2)),
                                -(((Ri**2 - Rj**2)*(rix - rjx)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2)))
    da_driy = np.stack((-(((Ri**2 - Rj**2)*(-rix + rjx)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2),
                                0.5 - (Ri**2 - Rj**2)/(2.*((rix - rjx)**2 + (riy - rjy)**2)) - ((Ri**2 - Rj**2)*(riy - rjy)*(-riy + rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2))
    db_drix =np.stack(((0.25*((4*(Ri**2 - Rj**2)**2*(rix - rjx))/((rix - rjx)**2 + (riy - rjy)**2)**3 - (4*(Ri**2 + Rj**2)*(rix - rjx))/((rix - rjx)**2 + (riy - rjy)**2)**2)*(-riy + rjy))/np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2)),(0.25*(rix - rjx)*((4*(Ri**2 - Rj**2)**2*(rix - rjx))/((rix - rjx)**2 + (riy - rjy)**2)**3 - (4*(Ri**2 + Rj**2)*(rix - rjx))/((rix - rjx)**2 + (riy - rjy)**2)**2))/np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2)) + 0.5*np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2))))
    db_driy =np.stack((-0.5*np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2)) + (0.25*((4*(Ri**2 - Rj**2)**2*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**3 - (4*(Ri**2 + Rj**2)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2)*(-riy + rjy))/np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2)),(0.25*(rix - rjx)*((4*(Ri**2 - Rj**2)**2*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**3 - (4*(Ri**2 + Rj**2)*(riy - rjy))/((rix - rjx)**2 + (riy - rjy)**2)**2))/np.sqrt(-1 - (Ri**2 - Rj**2)**2/((rix - rjx)**2 + (riy - rjy)**2)**2 + (2*(Ri**2 + Rj**2))/((rix - rjx)**2 + (riy - rjy)**2))))
    da_dri,db_dri = np.stack((da_drix,da_driy)).T,np.stack((db_drix,db_driy)).T
    dhCCW_dri,dhCW_dri = da_dri-db_dri, da_dri+db_dri
    return dhCCW_dri,dhCW_dri
ti = 70
ri,rj,rk = geom.tS[ti]
Ri,Rj,Rk = geom.tR[ti]
dhCCW_dri,dhCW_dri = tdh_dri(geom.tS,geom.rj,geom.tR,geom.Rj)
ccw, cw = dh_dri(ri,rj,Ri,Rj)




ccw,cw = tdh_dri(geom.tS,_roll3(geom.tS),geom.tR,_roll(geom.tR))

dh_CCW_dri_exact = np.zeros((nx,nx,2,2))
dh_CW_dri_exact = np.zeros((nx,nx,2,2))
for i in range(nx):
    for j in range(nx):
        ri = np.array((X[i,j],Y[i,j]))
        dh_CCW_dri_exact[i,j],dh_CW_dri_exact[i,j] = dh_dri(ri, rj, Ri, Rj)

fig, ax = plt.subplots(1,2)
ax[0].imshow(dh_CCW_mat_dx_num[1:-2,1:-2,0])
ax[1].imshow(dh_CCW_dri_exact[1:-2,1:-2,0,0])
fig.show()

plt.scatter(dh_CW_mat_dx_num[1:-2,1:-2,0],dh_CW_dri_exact[1:-2,1:-2,0,0])
plt.show()

plt.scatter(dh_CW_mat_dx_num[1:-2,1:-2,1],dh_CW_dri_exact[1:-2,1:-2,1,0])
plt.show()

plt.scatter(dh_CCW_mat_dy_num[1:-2,1:-2,0],dh_CCW_dri_exact[1:-2,1:-2,0,1])
plt.show()

plt.scatter(dh_CCW_mat_dy_num[1:-2,1:-2,1],dh_CCW_dri_exact[1:-2,1:-2,1,1])
plt.show()


plt.plot((dh_CCW_mat_dy_num[1:-2,1:-2,1]-dh_CCW_dri_exact[1:-2,1:-2,1,1]-0.5).ravel())
plt.show()


plt.plot((dh_CCW_mat_dy_num[1:-2,1:-2,1]-0.5-dh_CCW_dri_exact[1:-2,1:-2,1,1]).ravel()/dh_CCW_mat_dy_num[1:-2,1:-2,1].ravel()*100)
plt.show()

"""
Build code to calculate explicitly the differential 
"""

