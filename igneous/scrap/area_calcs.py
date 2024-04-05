from scipy.spatial.distance import cdist
from matplotlib.patches import Circle
import seaborn as sns


def hexagonal_lattice(rows=3, cols=3, noise=0.0005, A=None):
    """
    Assemble a hexagonal lattice
    :param rows: Number of rows in lattice
    :param cols: Number of columns in lattice
    :param noise: Noise added to cell locs (Gaussian SD)
    :return: points (nc x 2) cell coordinates.
    """
    points = []
    for row in range(rows * 2):
        for col in range(cols):
            x = (col + (0.5 * (row % 2))) * np.sqrt(3)
            y = row * 0.5
            x += np.random.normal(0, noise)
            y += np.random.normal(0, noise)
            points.append((x, y))
    points = np.asarray(points)
    if A is not None:
        points = points * np.sqrt(2 * np.sqrt(3) / 3) * np.sqrt(A)
    return points


msh = Mesh(mesh_params={"L": 1, "R_mult": 1e4})
x = hexagonal_lattice(4, 4, 1e-1)
x -= x.min()
x /= x.max() / 2
x += np.array([0.3, 0.3])
x = np.mod(x, 1)
R = np.ones(len(x)) * 0.05
msh.load_X(x, R)

x_hat, R_hat, dictionary = generate_triangulation_mask(x, R, 1, 1)

_x = np.mgrid[0:1:0.001, 0:1:0.001].transpose(1, 2, 0)
ims = np.zeros((len(x_hat), _x.shape[0], _x.shape[1]))
for i in range(len(x_hat)):
    ims[i] = np.sum((_x - x_hat[i]) ** 2, axis=-1) - R_hat[i] ** 2

assignment = np.argmin(ims, axis=0)
assignment = dictionary[assignment].astype(np.float)
assignment[ims.min(axis=0) > 0] = np.nan
mesh_props = msh.mesh_props

empirical_areas = np.zeros((len(x)))
for i in range(len(x)):
    empirical_areas[i] = (assignment == i).sum()
plt.scatter(empirical_areas * (0.001 ** 2), msh.mesh_props["A"])
plt.show()

mesh_props["t_zeta"] = tvecangle(mesh_props["v_x"], mesh_props["v_p1_x"])
mesh_props["zeta"] = assemble_scalar(mesh_props["t_zeta"], mesh_props["tri"], n_c)

np.argsort(np.abs(empirical_areas * (0.001 ** 2) - msh.mesh_props["A"]))
mesh_props["h_m_x"][mesh_props["tri"] == 29] - mesh_props["h_p_x"][mesh_props["tri"] == 29]

hm_hp = jnp.array([mesh_props["h_m"][mesh_props["tri"] == 1], mesh_props["h_p"][mesh_props["tri"] == 1]])
hp_v = jnp.array([mesh_props["h_m"][mesh_props["tri"] == 1], mesh_props["v"][mesh_props["tri"] == 1]])

fig, ax = plt.subplots()
ax.scatter(*x_hat.T)
ax.plot(hm_hp[..., 0], hm_hp[..., 1], zorder=10000, color="darkred")
ax.plot(hp_v[..., 0], hp_v[..., 1], zorder=10000, color="darkblue")

ax.scatter(*mesh_props["h_p"][mesh_props["tri"] == 1].T, zorder=1000, color="darkred", s=50)
ax.scatter(*mesh_props["h_m"][mesh_props["tri"] == 1].T, zorder=1000, color="darkred", s=50)

ax.scatter(*mesh_props["h_p"].T)

# for center, radius in zip(x_hat,R_hat):
#     ax.add_patch(Circle(center, radius, edgecolor='b', facecolor='none'))
center, radius = x_hat[1], R_hat[1]
ax.add_patch(Circle(center, radius, edgecolor='red', facecolor='none'))
for i in range(len(x)):
    ax.annotate(i, (x[i, 0], x[i, 1]))
for i in range(len(mesh_props["tri"])):
    for j in range(3):
        start, end = mesh_props["v"][i, j], mesh_props["v_p1"][i, j]
        # start = np.mod(start,1)
        # end = start + periodic_displacement(end,start,1)
        # ax.scatter(*start,color="blue")
        ax.plot((start[0], end[0]), (start[1], end[1]), color="black")

ax.imshow(np.flip(assignment.T, axis=0), zorder=-1, extent=[0, 1, 0, 1], cmap="rainbow")
ax.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
fig.show()

i = 1

fig, ax = plt.subplots()
hm_hp = jnp.array([mesh_props["h_m"][mesh_props["tri"] == i], mesh_props["h_p"][mesh_props["tri"] == i]])
hp_v = jnp.array([mesh_props["h_m"][mesh_props["tri"] == i], mesh_props["v"][mesh_props["tri"] == i]])

ax.scatter(*x_hat.T)
ax.plot(hm_hp[..., 0], hm_hp[..., 1], zorder=10000, color="darkred")
ax.plot(hp_v[..., 0], hp_v[..., 1], zorder=10000, color="darkblue")

ax.scatter(*mesh_props["h_p"][mesh_props["tri"] == i].T, zorder=1000, color="darkred", s=50)
ax.scatter(*mesh_props["h_m"][mesh_props["tri"] == i].T, zorder=1000, color="darkred", s=50)

# ax.scatter(*mesh_props["h_p"].T)

# for center, radius in zip(x_hat,R_hat):
#     ax.add_patch(Circle(center, radius, edgecolor='b', facecolor='none'))

for _i in [30, 29, 26]:
    center, radius = x[_i], R[_i]
    ax.add_patch(Circle(center, radius, edgecolor='red', facecolor='none'))
# center, radius = x_hat[30], R_hat[30]
# ax.add_patch(Circle(center, radius, edgecolor='red', facecolor='none'))

for _i in range(len(x)):
    ax.annotate(_i, (x[_i, 0], x[_i, 1]))
for _i in range(len(mesh_props["tri"])):
    for j in range(3):
        if (mesh_props["tri"][_i] == i).any():
            start, end = mesh_props["v"][_i, j], mesh_props["v_p1"][_i, j]
            # start = np.mod(start,1)
            # end = start + periodic_displacement(end,start,1)
            # ax.scatter(*start,color="blue")
            ax.plot((start[0], end[0]), (start[1], end[1]), color="black")
# image = np.array([(assignment==i)*i for i in list(tri[mask].ravel())]).sum(axis=0)
# ax.imshow(np.flip(image.T,axis=0),zorder=-1,extent=[0,1,0,1],cmap="Reds")

ax.imshow(np.flip((assignment == i).T, axis=0), zorder=-1, extent=[0, 1, 0, 1], cmap="Reds")
# ax.set(xlim=(0,0.5), ylim=(0,0.5))
fig.show()

empirical_areas = np.zeros((20))
for i in range(20):
    empirical_areas[i] = (assignment == i).sum()
plt.scatter(empirical_areas * (0.001 ** 2), msh.mesh_props["A"])
plt.show()

msh._triangulate()
mesh_props = get_geometry(msh.mesh_props)
# mesh_props =  get_tintersections(mesh_props)
hp = mesh_props["h_CCW_p"]
hm = mesh_props["h_CW_p"]

touch_not_power_mask = (~mesh_props["V_in_p1"]) * (~mesh_props["no_touch_p1"])

hp_circ = hp[touch_not_power_mask]
d = cdist(hp_circ, mesh_props["x"]) ** 2 - mesh_props["R"] ** 2

false_cross = np.count_nonzero(d <= 1e-7, axis=1) > 2
print(any(false_cross))

fig, ax = plt.subplots()
ax.scatter(*x.T)
# for center, radius in zip(x,R):
#     ax.add_patch(Circle(center, radius, edgecolor='b', facecolor='none'))
# for center, radius in zip(x[[5,11,6]], R[[5,11,6]]):
#     ax.add_patch(Circle(center, radius, edgecolor='r', facecolor='none'))

# ax.scatter(*hp.T)
# for i in range(len(x)):
#     ax.annotate(i, (x[i,0], x[i,1]))
# ax.scatter(*mesh_props["v"].T)
# ax.scatter(*mesh_props["v"][8])
ax.set(xlim=(0, 1), ylim=(0, 1))

for i in range(len(x_hat) - 4):
    ax.annotate(dictionary[i], (x_hat[i, 0], x_hat[i, 1]), color="red")

tx_hat = x_hat[tri_hat]
for i in range(3):
    for j in range(len(tx_hat)):
        start, end = tx_hat[j, i], tx_hat[j, (i + 1) % 3]
        if ((start // 1) == 0).all():
            ax.plot((start[0], end[0]), (start[1], end[1]), color="black")
        else:
            ax.plot((start[0], end[0]), (start[1], end[1]), color="grey", alpha=0.2)

ax.set(xlim=(0, 1), ylim=(0, 1), aspect=1)
fig.show()

#
i = 14  # (array([12, 14, 16, 17, 18, 19, 24, 25, 26, 36, 46, 52, 53]),

j = 2  # array([2, 2, 2, 1, 1, 0, 1, 2, 2, 2, 1, 0, 0]))

fig, ax = plt.subplots()
ax.scatter(*mesh_props["tx"][i, j])
ax.scatter(*mesh_props["tx_m1"][i, j], color="grey")
ax.scatter(*mesh_props["tx_p1"][i, j])
ax.scatter(*mesh_props["v"][i, j])

for _i, (center, radius) in enumerate(zip(mesh_props["tx"][i], mesh_props["tR"][i])):
    ax.add_patch(Circle(center, radius, edgecolor=plt.cm.plasma(_i / 3), facecolor='none'))

for _i, (center, radius) in enumerate(zip(mesh_props["x"], mesh_props["R"])):
    ax.add_patch(Circle(center, radius, edgecolor="grey", facecolor='none', alpha=0.2))

for _i, (center, radius) in enumerate(zip(mesh_props["x"][np.where(mesh_props["theta"] < 0)[0]],
                                          mesh_props["R"][np.where(mesh_props["theta"] < 0)[0]])):
    ax.add_patch(Circle(center, radius, edgecolor="blue", facecolor='none', alpha=0.4))

#
# for _i, (center, radius) in enumerate(zip(mesh_props["tx"][i],mesh_props["tR"][i])):
#     ax.add_patch(Circle(center+np.array([0,-1]), radius, edgecolor=plt.cm.plasma(_i/3), facecolor='none'))
#
# for _i, (center, radius) in enumerate(zip(mesh_props["tx"][i],mesh_props["tR"][i])):
#     ax.add_patch(Circle(center+np.array([0,1]), radius, edgecolor=plt.cm.plasma(_i/3), facecolor='none'))

# ax.scatter(*mesh_props["h_mid"][i,j],color="green")
# ax.scatter(*tx_p1_registered[i,j],color="green")

ax.scatter(*mesh_props["h_CCW"][i, j], color="blue")
# ax.scatter(*mesh_props["h_p"][i,j],color="purple")
ax.scatter(*mesh_props["h_m"][i, j].T, color="black", alpha=0.2, s=100)

ax.set(aspect=1)
fig.show()

import numpy as np
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import Point, Polygon, GeometryCollection
from shapely.ops import unary_union


def order_vertices(vertices, center):
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return vertices[sorted_indices]


def intersection_circle_polygon(circle_center, circle_radius, polygon_vertices):
    ordered_vertices = order_vertices(polygon_vertices, circle_center)
    polygon = Polygon(ordered_vertices)
    circle = Point(circle_center).buffer(circle_radius)
    intersection = polygon.intersection(circle)
    return intersection


def plot_intersection(intersection, ax):
    if intersection.is_empty:
        print("No intersection")
    else:
        if intersection.geom_type == 'Polygon':
            intersection_patch = PolygonPatch(intersection, fc='red', alpha=0.5)
            ax.add_patch(intersection_patch)
        elif intersection.geom_type == 'MultiPolygon':
            for poly in intersection:
                intersection_patch = PolygonPatch(poly, fc='red', alpha=0.5)
                ax.add_patch(intersection_patch)


from scipy.spatial import Voronoi

vor = Voronoi(x_hat)

intersections = []
for i in range(len(x)):
    intersections += [
        intersection_circle_polygon(mesh_props["x"][i], mesh_props["R"][i], mesh_props["v"][mesh_props["tri"] == i])]

fig, ax = plt.subplots()
for intersection in intersections:
    plot_intersection(intersection, ax)
fig.show()

"""
To do: 

1. Fix the vertex allocation such that they are all facing in the correct direction. 
2. Check whether this solves 
3. If not, partition the areas by the sub-triangles from vertex to vertex to cell centre and repeat. 

The errors are reference frame invariant

"""
