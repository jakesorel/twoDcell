import two_dimensional_cell.tri_functions as trf
from two_dimensional_cell.mesh import Mesh
from two_dimensional_cell.tissue import Tissue
from two_dimensional_cell.tissue import *
import numpy as np
import matplotlib.pyplot as plt
import two_dimensional_cell.periodic_functions as per

radius= 1/np.sqrt(3)

P0=(radius*np.pi*2)*1.2
A0=(radius**2 * np.pi)*0.9

tissue_params = {"L": 9,
                 "radius":radius,
                 "A0": A0,
                 "P0": P0,
                 "kappa_A": 0.5,
                 "kappa_P": 0.1,
                 "kappa_M":0.5,
                 "W": np.array(((0, 0.00762), (0.00762, 0))),
                 "a": 0,
                 "k": 0}
active_params = {"v0": 1e-1,
                 "Dr": 1e-1}
init_params = {"init_noise": 0.2,
               "c_type_proportions": (1.0,0.0),
               "n_take":60}
run_options = {"equiangulate": True,
               "equi_nkill": 10}
simulation_params = {"dt": 0.02,
                     "tfin": 20,
                     "tskip": 1,
                     "grn_sim": None,
                     "random_seed":10}
save_options = {"save": "skeleton",
                "result_dir": "results",
                "name": "ctype_example2",
                "compressed": True}

t= Tissue(tissue_params=tissue_params, active_params=active_params, init_params=init_params,run_options=run_options,calc_force=True)
# t.initialize_mesh(n_take=20,run_options=run_options)
# t.complete_initialization()
t.tissue_params["P0"]*=0.8
#
#
#
# # @jit(nopython=True)
# def _vectorify_boundary(x, n,n_boundary,boundary_val=0):
#     out = x * np.ones(n+n_boundary)
#     out[-n_boundary:] = boundary_val
#     return out
# mid = np.array((4.5,4.5))
# x = np.array(((0,0),
#               (0,1),
#               (np.sqrt(3)/2,0.5)
#               # ,
#               # (np.sqrt(3)/2,-0.5)
#               ))*radius + mid
#
# t.mesh = Mesh(x, radius, 9, run_options=run_options)
#
# self = t
# tp = {"L": 9,
#                  "radius":radius,
#                  "A0": A0,
#                  "P0": P0,
#                  "kappa_A": 0.5,
#                  "kappa_P": 0.0,
#                  "kappa_M":0.1,
#                  "W": np.array(((0, 0.00762), (0.00762, 0))),
#                  "a": 0,
#                  "k": 0}
# if self.mesh is not None:
#     for par in ["A0", "P0", "kappa_A", "kappa_P", "kappa_M"]:
#         self.tissue_params[par] = _vectorify_boundary(tp[par], self.mesh.n_c, 4, 0)
#
#     self.active = ActiveForce(self, active_params)
#
#     self.get_forces()
# t.tissue_params["P0"]*=0.65


#
dt = 0.025
#
n_rep = int(1000)
E_save = np.zeros(n_rep)
x_save = [None]*n_rep
# tri_save = [None]*n_rep
# mesh_save = [None]*n_rep
for i in range(n_rep):
    t.update(dt)
    F = t.get_forces()  # calculate the forces.
    # F = Force(t).F
    t.mesh.x += F*dt
    t.mesh.x = per.mod2(t.mesh.x,t.mesh.L,t.mesh.L)
    E_save[i] = np.sum(t.kappa_A[:t.mesh.n_c]*(t.mesh.A-t.A0[:t.mesh.n_c])**2 + t.kappa_P[:t.mesh.n_c]*(t.mesh.P-t.P0[:t.mesh.n_c])**2)
    x_save[i] = t.mesh.x
#

plt.close("all")
plt.plot(E_save)
plt.show()

animate(x_save,tissue_params["L"],tissue_params["radius"],n_frames=20)



import matplotlib.pyplot as plt

def plot_circle(centre, radius,ax,color="black",alpha=1):
    theta = np.linspace(0,np.pi*2,100)
    ax.plot(centre[0]+ radius*np.cos(theta),centre[1]+radius*np.sin(theta),color=color,alpha=alpha)

F = Force(t).F

fig, ax = plt.subplots()
# radius = 0.5
for i, xx in enumerate(t.mesh.y[:t.mesh.n_c]):
    plot_circle(xx,radius,ax,color=plt.cm.plasma((t.mesh.A[i]-t.mesh.A.min())/(t.mesh.A.max()-t.mesh.A.min())))
    ax.text(xx[0],xx[1],i)
# z = t.mesh.y[np.concatenate((t.mesh.idxs,(-1,-1,-1,-1))) == 93]
# for i, xx in enumerate(t.mesh.y[t.mesh.n_c:]):
#     plot_circle(xx, radius, ax,color="red")
# for i, xx in enumerate(z):
#     plot_circle(xx, radius, ax,color="red",alpha=0.3)
ax.quiver(t.mesh.x[:,0],t.mesh.x[:,1],F[:,0],F[:,1],scale=0.1)
# ax.quiver(t.mesh.x[:,0],t.mesh.x[:,1],t.F[:,0],t.F[:,1],scale=2,color="red")

ax.set(aspect=1)
# plot_circle(self.y[4], radius, ax,color="green")
fig.show()
# fig.savefig("triangulation.pdf",dpi=300)

#
# L = 9.5
# A0 = 1
# init_noise = 1e-1
# x = trf.hexagonal_lattice(int(L), int(np.ceil(L)), noise=init_noise, A=A0)
# x += 1e-3
#
# x = x[np.argsort(x.max(axis=1))[:int(L ** 2 / A0)]]
#
# x += 0.25
# np.argsort(x.max(axis=1))
# radius = 0.6
# buffer = 1.2
#
# self = Mesh(x,radius,L,buffer)
# self.update()


import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_hex
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi

from matplotlib.patches import Polygon
from shapely.geometry import Polygon, Point
from descartes import PolygonPatch

"""
Plotting funcitons
------------------

Sets of plotting functions for the voronoi model, allowing to generate static images and animations. Colours and other kwargs can be parsed. 
"""


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def hex_to_rgb(value):
    """
    Convert a hex to an rgb value.
    :param value:
    :return:
    """
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) / 255 for i in range(0, lv, lv // 3)) + (1,)



def plot_vor(ax, x, L, radius,cols=None, cbar=None, **kwargs):
    """
    Plot the Voronoi.

    Takes in a set of cell locs (x), tiles these 9-fold, plots the full voronoi, then crops to the field-of-view

    :param L: Domain size
    :param x: Cell locations (nc x 2)
    :param ax: matplotlib axis
    :param cols: array of strings (e.g. hex) defining the colour of each cell, in the order of x
    :param cbar: dictionary defining the options of the colorbar. cbar["cmap"] is the colormap. cbar["vmax],cbar["vmin] are the max and min vals in the cmap. cbar["label"] is the colorbar label.
    """

    if cols is None:
        cols = np.repeat("grey", x.shape[0])
    mesh = Mesh(x,tissue_params["radius"],tissue_params["L"],run_options=run_options)
    # y = np.vstack([x + np.array([i * L, j * L]) for i, j in np.array([grid_x.ravel(), grid_y.ravel()]).T])

    # cols_print = np.tile(cols, 9)
    # bleed = 0.1
    # cols_print = cols_print[(y < L * (1 + bleed)).all(axis=1) + (y > -L * bleed).all(axis=1)]
    # y = y[(y < L * (1 + bleed)).all(axis=1) + (y > -L * bleed).all(axis=1)]
    y = mesh.y
    regions, vertices = voronoi_finite_polygons_2d(Voronoi(y))
    # patches = []
    for i, region in enumerate(regions):
        # patches.append(Polygon(vertices[region], True, facecolor=cols_print[i], ec=(1, 1, 1, 1), **kwargs))
        poly = Polygon(vertices[region])
        circle = Point(y[i]).buffer(radius)
        cell_poly = circle.intersection(poly)
        if cell_poly.area != 0:
            ax.add_patch(PolygonPatch(cell_poly, ec="white", fc="green"))

    # p = PatchCollection(patches, match_original=True)
    # p.set_array(c_types_print)
    # ax.add_collection(p)
    ax.set(xlim=(0, L), ylim=(0, L), aspect=1)
    ax.axis("off")
    if cbar is not None:
        sm = plt.cm.ScalarMappable(cmap=cbar["cmap"], norm=plt.Normalize(vmax=cbar["vmax"], vmin=cbar["vmin"]))
        sm._A = []
        cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=10, orientation="vertical")
        cl.set_label(cbar["label"])

# fig, ax = plt.subplots()
# plot_vor(ax,x,tissue_params["L"],tissue_params["radius"])
# fig.show()


def animate(x_save, L,radius, n_frames=100, file_name=None, dir_name="plots", cbar=None, **kwargs):
    """
    Animate the simulation

    :param x_save: nt x nc x 2 array of positions
    :param L: domain size
    :param cols: either a nc array of strings defining fixed colors, or a nt x nc array if colors are to vary.
    :param n_frames: number of frames ot plot
    :param file_name: file name
    :param dir_name: directory name into which the animation is saved
    :param cbar: see above.
    :param kwargs: other arguments parsed into plot_vor
    :return:
    """

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    skip = int((len(x_save)) / n_frames)

    def animate(i):
        ax1.cla()
        plot_vor(ax1, x_save[skip * i], L,radius, **kwargs)
        ax1.set(aspect=1, xlim=(0, L), ylim=(0, L))

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    if file_name is None:
        file_name = "animation %d" % time.time()
    an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
    an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)

