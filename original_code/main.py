import power_triangulation as pt
from power_triangulation import *
import numpy as np
from cell_geometries import geometries
from cell_geometries import _triangulated_form
import time
import matplotlib.pyplot as plt
from tri_functions import hexagonal_lattice,square_boundary
from tri_functions import _roll,_roll3,_tnorm,_get_b_edges,_tri_sum,normalise,_cosine_rule
from cell_geometries import _angles
from scipy.spatial.distance import cdist
import os
from tri_functions import _replace_val,_tdot
from differentials import _dtheta_dh,_power_vertex_differentials,_compile_chain2,_circle_vertex_differentials,_compile_alt_thetas,dhdr,_circle_vertex_differentials_alt


xlim,ylim = (-20,20),(-20,20)
S = hexagonal_lattice(3,3,0.1)
# S = np.array(((0,0),(2,0),(1,np.sqrt(3))))+np.array((3,3)) #+ np.random.normal(0,0.25,(3,2))
# S = np.array(((0,0),(1,np.sqrt(3)))) #+ np.random.normal(0,0.25,(2,2))
Sb = square_boundary(xlim,ylim)
n_b = Sb.shape[0]
S = np.vstack((S,Sb))

plot_name = "plots_april"

R = np.ones_like(S[:, 0])*1.1547005387844387 + 0.1 #+ np.random.normal(0,0.1,S[:,0].size)
tri_list, V, n_v = pt.get_power_triangulation(S, R)
geom = geometries(S, R, V, tri_list, n_v,n_b=n_b)
geom.P0 = (R[0]*np.pi*2)*0.5
geom.A0 = 0.5
geom.lambda_P = 1
geom.lambda_A = 0
geom.lambda_M = 0

# voronoi_cell_map = pt.get_voronoi_cells(S, V, tri_list)
# fig, ax = plt.subplots()
# pt.display(ax, S, R, tri_list, voronoi_cell_map, tri_alpha=0.0, n_b=n_b)
# ax.scatter(S[:,0],S[:,1],zorder=1000)
# fig.show()
# geom.differential_matrices()
geom.get_F()
self = geom

self.lambda_M = np.zeros(self.n_c)
# self.lambda_M[42] = 1
self.lambda_M = 1

self.get_F()

voronoi_cell_map = pt.get_voronoi_cells(S, V, tri_list)
fig, ax = plt.subplots()
pt.display(ax, S, R, tri_list, voronoi_cell_map, tri_alpha=0.0, n_b=n_b)
# ax.quiver(geom.S[:, 0], geom.S[:, 1], geom.FP[:, 0], geom.FP[:, 1], zorder=1e11)#,scale=1)
# ax.scatter(geom.hp_j[:,:,0].ravel(),geom.hp_j[:,:,1].ravel(),zorder=1000,s=10,alpha=0.6,color="red")#,dEP_dhp[:,:,0].ravel(),dEP_dhp[:,:,1].ravel(),zorder=1000)
ax.scatter(geom.hm_j[:,:,0].ravel(),geom.hm_j[:,:,1].ravel(),zorder=1000,s=10,alpha=0.6,color="red")#,dEP_dhp[:,:,0].ravel(),dEP_dhp[:,:,1].ravel(),zorder=1000)

# ax.scatter(geom.hm_j[:,:,0].ravel(),geom.hm_j[:,:,1].ravel(),zorder=1000)#,dEP_dhp[:,:,0].ravel(),dEP_dhp[:,:,1].ravel(),zorder=1000)

# ax.scatter(geom.hp_k[:,:,0].ravel(),geom.hp_k[:,:,1].ravel(),zorder=1000)#,dEP_dhp[:,:,0].ravel(),dEP_dhp[:,:,1].ravel(),zorder=1000)
# ax.scatter(geom.hm_k[:,:,0].ravel(),geom.hm_k[:,:,1].ravel(),zorder=1000)#,dEP_dhp[:,:,0].ravel(),dEP_dhp[:,:,1].ravel(),zorder=1000)

# ax.scatter(geom.hm_j[:,:,0].ravel(),geom.hm_j[:,:,1].ravel(),zorder=1000)#,dEP_dhp[:,:,0].ravel(),dEP_dhp[:,:,1].ravel(),zorder=1000)
# ax.scatter(geom.tV[:,:,0].ravel(),geom.tV[:,:,1].ravel(),zorder=1000)#,dEP_dhp[:,:,0].ravel(),dEP_dhp[:,:,1].ravel(),zorder=1000)

# ax.quiver(geom.tS[:,:,0].ravel(),geom.tS[:,:,1].ravel(),-geom.tFC_s[:,:,0].ravel(),-geom.tFC_s[:,:,1].ravel(),zorder=1e3,scale=0.2)
#
# ax.scatter(geom.hp_j[30,:,0].ravel(),geom.hp_j[30,:,1].ravel(),zorder=1000)#,dEP_dhp[:,:,0].ravel(),dEP_dhp[:,:,1].ravel(),zorder=1000)

ax.axis("on")
for i in range(geom.n_c):
    ax.text(geom.S[i,0],geom.S[i,1],i,zorder=1e11,fontsize=20)
print(self.FP.max())
fig.show()


plt.scatter(self.lC[:-self.n_b],np.linalg.norm(self.FP[:-self.n_b],axis=1))
plt.show()


np.linalg.norm(self.FP[:-self.n_b],axis=1)[self.lC[:-self.n_b]==0]


#
# voronoi_cell_map = pt.get_voronoi_cells(S, V, tri_list)
# fig, ax = plt.subplots()
# pt.display(ax, S, R, tri_list, voronoi_cell_map, tri_alpha=0.0, n_b=n_b)
# # geom.FP = F_s+F_h
# ax.quiver(geom.S[:, 0], geom.S[:, 1], geom.FP[:, 0], geom.FP[:, 1], zorder=1e11)
# # ax.quiver(geom.S[:, 0], geom.S[:, 1], F_s[:, 0], F_s[:, 1], zorder=1e11,color="green")
# # ax.quiver(geom.S[:, 0], geom.S[:, 1], F_h[:, 0], F_h[:, 1], zorder=1e11,color="red")
# # ax.quiver(geom.tS[:,:,0].ravel(),geom.tS[:,:,1].ravel(),geom.tF_c_h[:,:,0].ravel(),geom.tF_c_h[:,:,1].ravel(),zorder=1e3)
# ax.scatter(geom.V[:,0],geom.V[:,1],zorder=1e13)
# # ax.quiver(geom.tV[:,:,0].ravel(),geom.tV[:,:,1].ravel(),self.dPi_dhv[:,:,0].ravel(),self.dPi_dhv[:,:,1].ravel(),zorder=1e13,scale=0.01)
#
# ax.scatter(geom.hp_j[:,:,0].ravel(),geom.hp_j[:,:,1].ravel(),zorder=1000)
# # ax.quiver(geom.hp_j[:,:,0].ravel(),geom.hp_j[:,:,1].ravel(),dEP_dhp[:,:,0].ravel(),dEP_dhp[:,:,1].ravel(),zorder=1000)
# # ax.quiver(geom.hp_j[:,:,0].ravel(),geom.hp_j[:,:,1].ravel(),dtheta_dhp_cell_i[:,:,0].ravel(),dtheta_dhp_cell_i[:,:,1].ravel(),zorder=1000)
# # ax.quiver(geom.hp_j[:,:,0].ravel(),geom.hp_j[:,:,1].ravel(),dtheta_dhp_cell_j[:,:,0].ravel(),dtheta_dhp_cell_j[:,:,1].ravel(),zorder=1000,color="Red")
#
# # ax.quiver(geom.V[:,0],geom.V[:,1],dEP_dhv[:,0],dEP_dhv[:,1],zorder=1e12)
# for i in range(geom.n_c):
#     ax.text(geom.S[i,0],geom.S[i,1],i,zorder=1e11,fontsize=20)
# print(self.FP.max())
# fig.show()

theta_space = np.linspace(-np.pi,np.pi,100)
dtheta = theta_space[1]-theta_space[0]
f = np.sqrt(2*(1-np.cos(theta_space)))
df = np.cos(np.mod(theta_space,np.pi*2)/2)
ndf = (f[1:]-f[:-1])/dtheta


print(geom.FP[:3])



nt = 1000
# R = np.ones_like(S[:, 0])*0.9
S_save = np.zeros((nt,S.shape[0],2))
R_save = np.zeros((nt,S.shape[0]))

save_skip = 20
P_save = np.zeros((nt,S.shape[0]-n_b))
A_save = np.zeros((nt,S.shape[0]-n_b))

min_corner = np.amin(S[:-n_b], axis=0) - np.max(R)*3
max_corner = np.amax(S[:-n_b], axis=0) + np.max(R)*3
xlim = (min_corner[0], max_corner[0])
ylim = (min_corner[1], max_corner[1])

P0 = R[0]*(np.pi*1.5)
A0 = (R[0]**2 * np.pi)*0.7

dt = 0.005
v0 = 0
for t in range(nt):
    tri_list, V, n_v = pt.get_power_triangulation(S, R)
    geom = geometries(S, R, V, tri_list, n_v,n_b=n_b)
    geom.P0 = P0
    geom.A0 = A0
    geom.lambda_P = 1
    geom.lambda_A = 0
    geom.lambda_M = 0
    # geom.differential_matrices()
    geom.get_F()
    S += geom.FP*dt
    theta_noise = np.random.uniform(0,np.pi*2,geom.n_c)
    noise = np.array((v0*np.cos(theta_noise),v0*np.sin(theta_noise))).T
    S += noise*dt
    dR = geom.G*dt
    dR[-geom.n_b:] = 0

    # R += geom.G*dt
    S_save[t] = S
    R_save[t] = R
    if np.mod(t,save_skip)==0:
        tri_list, V, n_v = pt.get_power_triangulation(S, R)
        voronoi_cell_map = pt.get_voronoi_cells(S, V, tri_list)
        fig, ax = plt.subplots()
        pt.display(ax, S, R, tri_list, voronoi_cell_map, tri_alpha=0.0, n_b=n_b,xlim=xlim,ylim=ylim,line_col="white")
        ax.scatter(geom.S[:,0],geom.S[:,1],zorder=1e10,color="black")
        # ax.quiver(geom.S[:, 0], geom.S[:, 1], geom.FP[:, 0], geom.FP[:, 1], zorder=1e10)
        # ax.scatter(geom.hm_k[i, :, 0], geom.hm_k[i, :, 1], zorder=1e10)
        # ax.scatter(geom.V[:, 0], geom.V[:, 1], color="red", zorder=1e10, alpha=0.5)
        fig.savefig("%s/%d.pdf"%(plot_name,t))
        plt.close("all")
    P_save[t] = geom.P[:-geom.n_b]
    A_save[t] = geom.A[:-geom.n_b]

    print(t)

fig, ax = plt.subplots(1,2,figsize=(5,2))
ax[0].plot(P_save.mean(axis=1))
ax[0].plot(np.arange(P_save.mean(axis=1).size),np.repeat(P0,P_save.mean(axis=1).size))
ax[1].violinplot(P_save[:,::int(P_save.shape[0]/10)])
fig.savefig("%s/p_save.pdf"%(plot_name))


fig, ax = plt.subplots(1,2,figsize=(5,2))
ax[0].plot(A_save.mean(axis=1))
ax[1].violinplot(A_save[:,::int(P_save.shape[0]/10)])
fig.savefig("%s/a_save.pdf"%(plot_name))


S = S_save[1]
tri_list, V, n_v = pt.get_power_triangulation(S, R)
geom = geometries(S, R, V, tri_list, n_v,n_b=n_b)
geom.differential_matrices()
geom.get_F()
voronoi_cell_map = pt.get_voronoi_cells(S, V, tri_list)

i = 57
fig, ax = plt.subplots()
pt.display(ax,S, R, tri_list, voronoi_cell_map,tri_alpha=0.0,n_b = n_b)
ax.quiver(geom.S[:,0],geom.S[:,1],geom.FP[:,0],geom.FP[:,1],zorder=1e10)
ax.scatter(geom.hm_k[i,:,0],geom.hm_k[i,:,1],zorder=1e10)
ax.scatter(geom.V[i,0],geom.V[i,1],color="red",zorder=1e10,alpha=0.5)
fig.show()
print(np.abs(geom.FP).max(),np.abs(geom.FP).min())


from cell_geometries import _tintersections

# tphi = geom.tphi_j
tV = geom.tV
self = geom
tS, tR, nrij = self.tS, self.tR, self.nrij
vj_neighbours = geom.vj_neighbours
V_in = geom.V_in_j
V_out = geom.V_out_j
no_touch = geom.no_touch_j
dir = 1
h_CCW, h_CW = _tintersections(tS, _roll3(tS, dir), tR, _roll(tR, dir), nrij)

def _tintersectionsnew(tS,rij,nrij,Ri,dij):
    a = tS.T+-(rij.T)/(nrij.T) * dij.T
    b = np.dstack((rij[:,:,1],-rij[:,:,0])).T/(nrij.T) * np.sqrt(Ri.T**2 - dij.T**2)
    return (a+b).T,(a-b).T
h_CCWn, h_CWn = _tintersectionsnew(tS,geom.rij,geom.nrij,geom.tR,geom.dij)
plt.scatter(h_CCW,h_CCWn)
plt.show()
def d0(ri,Ri):
    return np.sum(ri**2)-Ri**2

def vertex(ri,rj,rk,Ri,Rj,Rk):
    num = (rk-rj)*d0(ri,Ri)+(rj-ri)*d0(rk,Rk)+(ri-rk)*d0(rj,Rj)
    num_z = np.array((-num[1],num[0]))
    cross = np.cross((ri-rj),(rj-rk))
    return num_z/(2*cross)

def dhv_dri(ri,rj,rk,Ri,Rj,Rk):
    """
    of the Jacobian, the first dimension is hvx,hvy and the second is rix,riy
    :param ri:
    :param rj:
    :param rk:
    :param Ri:
    :param Rj:
    :param Rk:
    :return:
    """
    rkj = rk - rj
    outer = np.outer(ri,np.array((-rkj[1],rkj[0])))
    second_term = 0.5*(d0(rk,Rk) - d0(rj,Rj))*np.array(((0,-1),(1,0)))
    cross = np.cross((ri-rj),(rj-rk))
    hv = vertex(ri,rj,rk,Ri,Rj,Rk)
    return (outer+second_term- np.outer(np.array((-rkj[1],rkj[0])),hv))/(cross)

ri,rj,rk = geom.tS[42]
Ri,Rj,Rk = geom.tR[42]

print(dhv_dri(ri,rj,rk,Ri,Rj,Rk))

dx = 0.001
nx = 100
x_space = np.arange(-dx*nx/2,dx*nx/2,dx)
X,Y = np.meshgrid(ri[0]+x_space,ri[1]+x_space,indexing="ij")

Hv = np.empty((nx,nx,2))
for i in range(nx):
    for j in range(nx):
        ri = np.array((X[i,j],Y[i,j]))
        Hv[i,j] = vertex(ri,rj,rk,Ri,Rj,Rk)
dHv_drix_num = (np.roll(Hv,-1,axis=0) - Hv)/dx
dHv_driy_num = (np.roll(Hv,-1,axis=1) - Hv)/dx
dHv_dri_num = np.zeros((nx,nx,2,2))
dHv_dri_num[:,:,0],dHv_dri_num[:,:,1] = dHv_drix_num,dHv_driy_num
# dHv_dri_num = np.dstack((dHv_drix_num,dHv_driy_num))

dHv_dri_exact = np.empty((nx,nx,2,2))
for i in range(nx):
    for j in range(nx):
        ri = np.array((X[i,j],Y[i,j]))
        dHv_dri_exact[i,j] = dhv_dri(ri,rj,rk,Ri,Rj,Rk)

plt.scatter(dHv_dri_exact[1:-2,1:-2].ravel(),dHv_dri_num[1:-2,1:-2].ravel())
plt.show()


plt.plot((dHv_dri_exact[1:-2,1:-2].transpose(0,1,3,2).ravel()-dHv_dri_num[1:-2,1:-2].ravel())/dHv_dri_exact[1:-2,1:-2].transpose(0,1,3,2).ravel())
plt.show()

fig, ax = plt.subplots(2)
ax[0].imshow(dHv_dri_exact[:,:,0,1])
ax[1].imshow(dHv_drix_num[1:-2,:,1])
fig.show()

plt.scatter(dHv_dri_exact[1:-2,:,0,1].ravel(),dHv_drix_num[1:-2,:,1].ravel())
plt.show()

plt.plot(100*(dHv_dri_exact[1:-2,:,0,1].ravel()-dHv_drix_num[1:-2,:,1].ravel())/dHv_dri_exact[1:-2,:,0,1].ravel())
plt.show()

from cell_geometries import _ttheta
def fn():
    # pt.get_power_triangulation(S, R)
    tdh_dri(geom.tS,_roll3(geom.tS),geom.tR,_roll(geom.tR))


def timeit(fn,n=int(1e3)):
    t0 = time.time()
    for i in range(n):
        fn()
    t1 = time.time()
    print((t1 - t0)/n * 1000, "ms")


phi = _tri_sum(geom.n_c,geom.CV_matrix,geom.tphi_j)
theta = _tri_sum(geom.n_c,geom.CV_matrix,geom.ttheta_k)
psi = _tri_sum(geom.n_c,geom.CV_matrix,geom.tpsi_j)

lP = _tri_sum(geom.n_c,geom.CV_matrix,geom.lP)
lC = _tri_sum(geom.n_c,geom.CV_matrix,geom.lC)
AP = _tri_sum(geom.n_c,geom.CV_matrix,geom.AP)
AC = _tri_sum(geom.n_c,geom.CV_matrix,geom.AC)

cols = plt.cm.plasma(normalise(theta))



