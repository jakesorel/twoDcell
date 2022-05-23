import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def get_area(pts):
    x,y = pts.T
    rx,ry = np.roll(x,-1),np.roll(y,-1)
    a = (x*ry - y*rx).sum()/2
    return a

@jit(nopython=True)
def get_centroid(pts):
    """location of centroid of a lamina from CCW points"""
    x, y = pts.T
    rx, ry = np.roll(x, -1), np.roll(y, -1)
    a = (x * ry - y * rx).sum() / 2
    cross = (x*ry-rx*y)
    cx = ((x+rx)*cross).sum()/(6*a)
    cy = ((y+ry)*cross).sum()/(6*a)
    return cx,cy

@jit(nopython=True)
def get_inertia(pts):
    'Moments and product of inertia about centroid.'
    x, y = pts.T
    rx, ry = np.roll(x, -1), np.roll(y, -1)
    a = (x * ry - y * rx).sum() / 2
    cross = (x*ry-rx*y)
    cx = ((x+rx)*cross).sum()/(6*a)
    cy = ((y+ry)*cross).sum()/(6*a)
    sxx = ((y**2 + y*ry + ry**2)*cross).sum()/12 - a*cy**2
    syy = ((x**2 + x*rx + rx**2)*cross).sum()/12 - a*cx**2
    sxy = ((x*ry + 2*x*y + 2*rx*ry + rx*y)*cross).sum()/24 - a*cx*cy
    return sxx,syy,sxy

@jit(nopython=True)
def sort_coords(coords,centre,start=None):
    if start is None:
        start_angle = 0
    else:
        start_vec = start-centre
        start_angle = np.arctan2(start_vec[1],start_vec[0])
    ncoords = coords-centre
    return coords[np.mod(np.arctan2(ncoords[:,1],ncoords[:,0])-start_angle,np.pi*2).argsort()]

@jit(nopython=True)
def approximate_arc(x1, x2, r, R, alpha_small=1):
    nx1, nx2 = x1 - r, x2 - r
    alpha1, alpha2 = np.arctan2(nx1[1], nx1[0]), np.arctan2(nx2[1], nx2[0])
    dalpha = np.mod(alpha2 - alpha1, np.pi * 2)
    nalpha = int(np.round(dalpha / alpha_small))
    if nalpha != 0:
        alpha_space = np.linspace(alpha1, alpha1 + dalpha, nalpha)
        x = np.row_stack((r[0] + R * np.cos(alpha_space), r[1] + R * np.sin(alpha_space))).T
    else:
        x = np.row_stack((x1,x2))
    return x

@jit(nopython=True)
def get_minor_major_axes(cll_i,S,R,hp,hm,CV_matrix,no_touch,alpha_small=0.05):
    cll_mask = CV_matrix[cll_i]==1
    cll_mask_flat = cll_mask.ravel()
    hpi = np.column_stack((hp[:,:,0].ravel()[cll_mask_flat],hp[:,:,1].ravel()[cll_mask_flat]))
    hmi = np.column_stack((hm[:,:,0].ravel()[cll_mask_flat],hm[:,:,1].ravel()[cll_mask_flat]))
    touch_i = (~no_touch).ravel()[cll_mask_flat]
    if touch_i.sum()!=0: ##account for fact that isolated cell will not have any contact, and also hence no major/minor axes
        hpi = hpi[touch_i]
        hmi = hmi[touch_i]
        vertex_centroid = np.array((((hpi[:,0]).mean()+hmi[:,0].mean())/2, ##OK, as hpi and hmi will have same size
                                          ((hpi[:,1]).mean()+hmi[:,1].mean())/2))

        hpi = sort_coords(hpi,vertex_centroid,hpi[0]) ##sort the coordinates CCW, starting from hpi[0]
        hmi = sort_coords(hmi,vertex_centroid,hmi[0]) ##as above, starting with next CCW coordinate from hpi[0]
        npts = hpi.shape[0]

        ##concatenate the discretized coordinates. Given this is done clockwise, sufficient to do only the 'curved' regions
        # (which may just be single points)
        coords = np.empty((0,2))
        for k in range(npts):
            coords = np.row_stack((coords,approximate_arc(hpi[k],hmi[k],S[cll_i],R[cll_i],alpha_small = alpha_small)))
        coords = np.row_stack((coords,hpi[0].reshape(-1,2)))

        ##Calculate intertial tensor
        Cxx,Cyy,Cxy = get_inertia(coords)
        C = np.array(((Cxx,Cxy),(Cxy,Cyy)))

        ##Compute the eigenvalues and eigenvectors
        eigval,eigvec = np.linalg.eig(C)

        ##Ascribe long and short axes
        major_ax = eigvec[np.argmin(np.abs(eigval))]
        minor_ax = eigvec[np.argmax(np.abs(eigval))]
    else:
        major_ax = np.array((np.nan,np.nan))
        minor_ax = np.array((np.nan,np.nan))

    return major_ax,minor_ax

#
# t0 = time.time()
# for i in range(1000):
#     get_minor_major_axes(cll_i, S, R, hp, hm, CV_matrix, no_touch, alpha_small=0.05)
# t1 = time.time()
# print(t1-t0)
#
#
# xlim,ylim = (-20,20),(-20,20)
# S = hexagonal_lattice(3,3,0.1)
# # S = np.array(((0,0),(2,0),(1,np.sqrt(3))))+np.array((3,3)) #+ np.random.normal(0,0.25,(3,2))
# # S = np.array(((0,0),(1,np.sqrt(3)))) #+ np.random.normal(0,0.25,(2,2))
# Sb = square_boundary(xlim,ylim)
# n_b = Sb.shape[0]
# S = np.vstack((S,Sb))
#
# plot_name = "plots_april"
#
# R = np.ones_like(S[:, 0])*1.1547005387844387 + 0.1 #+ np.random.normal(0,0.1,S[:,0].size)
# tri_list, V, n_v = pt.get_power_triangulation(S, R)
# geom = geometries(S, R, V, tri_list, n_v,n_b=n_b)
#
# voronoi_cell_map = pt.get_voronoi_cells(S, V, tri_list)
# fig, ax = plt.subplots()
# pt.display(ax, geom.S, geom.R, geom.tri_list, voronoi_cell_map, tri_alpha=0.0, n_b=n_b)
# # longaxes = np.array([get_orr2(i) for i in range(geom.n_c - geom.n_b)])
# # for i in range(geom.n_c-geom.n_b):
# for i in range(geom.n_c - geom.n_b):
#     long_axis,short_axis = get_orr2(i)
#
#     ax.quiver(geom.S[i,0],geom.S[i,1],long_axis[0],long_axis[1],zorder=1e9)
#     # ax.quiver(geom.S[i,0],geom.S[i,1],short_axis[0],short_axis[1],zorder=1e9,color="red")
#
# #     ax.text(geom.S[i,0],geom.S[i,1],i,zorder=1e11,fontsize=20)
#
# # for ii in range(geom.n_c):
#     # ax.text(geom.S[ii,0],geom.S[ii,1],ii,zorder=1e11,fontsize=20)
#
# fig.show()
#
#
# ##straight-lines will be hpj_i --> hmj_i; curved-lines wil be hmj_i --> hpj_{i+1}
#
#
#
# fig, ax = plt.subplots()
# pt.display(ax, geom.S, geom.R, geom.tri_list, voronoi_cell_map, tri_alpha=0.0, n_b=n_b)
#
# ax.scatter(hmj[:,0],hmj[:,1],color="red",zorder=10000)
# ax.scatter(hpj[:,0],hpj[:,1],color="blue",zorder=10000)
# for k in range(hmj.shape[0]):
#     ax.text(hmj[k,0],hmj[k,1],k,color="red",zorder=10000)
#     ax.text(hpj[k,0],hpj[k,1],k,color="blue",zorder=10000)
#
# c = np.zeros((hmj.shape[0]*2+1,2))
# c[:-1:2] = hpj
# c[1::2] = hmj
# c[-1] = hpj[0]
# # ax.plot(c[:,0],c[:,1],zorder=1000)
# for k in range(hmj.shape[0]):
#     c = np.array((hmj[k],hpj[np.mod(k+1,hmj.shape[0])]))
#     c2 = np.array((hpj[k],hmj[k]))
#
#     # c = np.array((hpj[k],hmj[np.mod(k+1,hmj.shape[0])]))
#
#     ax.plot(c[:,0],c[:,1],color="k",zorder=10000)
#     ax.plot(c2[:,0],c2[:,1],color="r",zorder=10000)
#
# fig.show()
#
#
#
#
# coords = np.unique(np.row_stack((hpj,hmj)),axis=0).mean(axis=0)
# coords = coords[np.arctan2(coords[:,1],coords[:,0]).argsort()]
#
#
# V = geom.tV[geom.CV_matrix[i]==1]
# ax.scatter(hpj[:,0],hpj[:,1],zorder=1000)
# ax.scatter(hmj[:,0],hmj[:,1],zorder=1000)
#
# print(self.FP.max())
# fig.show()
#
#
#
# ##Could approximate circle with projection of power vertices on circle.
# ##Alternatively, could approximate circle with many points on circle.
#
# (geom.CV_matrix[i]*geom.hp_j[:,:,0]).sum(axis=-1)
#
# ##2. sort in CCW direction about the centre (calculate this explicitly)
#
# ##3. perform PCA and solve for long-axis.
#
#

