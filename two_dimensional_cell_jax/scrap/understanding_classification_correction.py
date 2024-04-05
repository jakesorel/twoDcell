import two_dimensional_cell.power_triangulation as pt
import numpy as np
import two_dimensional_cell.tri_functions as trf
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from descartes import PolygonPatch  # Used to convert Shapely geometry to Matplotlib patches
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from two_dimensional_cell.mesh import Mesh
from two_dimensional_cell.sim_plotting import *



def estimate_areas(x,R,N=500):
    lowest = x.min()-R.max()
    highest = x.max()+R.max()
    X,Y = np.mgrid[lowest:highest:(highest-lowest)/N,lowest:highest:(highest-lowest)/N]
    dx = X[1,0]-X[0,0]
    XY = np.column_stack((X.ravel(),Y.ravel()))
    d2 = (((np.expand_dims(x, 0) - np.expand_dims(XY, 1))**2).sum(axis=-1)) - np.expand_dims(R,0)**2
    d2 = np.column_stack([((d2<0).any(axis=1)*d2.max()),d2])
    themap = np.argmin(d2, axis=1).reshape(X.shape)
    areas = (np.bincount(themap.ravel())*dx**2)[1:]
    return areas
    # plt.imshow(np.argmin(d2,axis=1).reshape(X.shape))
    # plt.show()



for j in range(500):

    np.random.seed(j)
    N = 50
    x = trf.hexagonal_lattice(10,10,0) + 20
    R = np.random.uniform(1,1.5,len(x))

    msh= Mesh(x,R,60,run_options={})
    vertices = get_by_cell_vertices_from_mesh(msh)

    perimeters = np.zeros(len(vertices))
    areas = np.zeros((len(vertices)))
    for i, vtx in enumerate(vertices):
        poly = Polygon(vtx)
        circle = Point(msh.y[i]).buffer(msh.R_extended[i])
        cell_poly = circle.intersection(poly)
        perimeters[i] = cell_poly.length
        areas[i] = cell_poly.area
    if (np.abs(perimeters - msh.P)>0.1).any():
        print(j)

idxs = np.nonzero(np.abs(perimeters - msh.P)>0.1)[0]
# idxs = np.unique(msh.tri[(msh.tri == idx).any(axis=1)])
idx = idxs[0]


connected_to_idx = np.unique(msh.tri[(msh.tri==idx).any(axis=1)])
vs_connected = msh.vs[(msh.tri==idx).any(axis=1)]


circles = [Point(centroid).buffer(radius) for centroid, radius in zip(msh.y[connected_to_idx], msh.R_extended[connected_to_idx])]

thecircle = Point(msh.y[idx]).buffer(msh.R_extended[idx])


#
#
# # Create a figure and axis
# fig, ax = plt.subplots()
# ax.scatter(*vs_connected.T)
#
# # Plot circles using Matplotlib patches
# for circle in circles:
#     ax.add_patch(PolygonPatch(circle, fc='blue', ec='black', alpha=0.5))
#
# ax.add_patch(PolygonPatch(thecircle, fc='red', ec='red', alpha=0.5))
#
# i, vtx = idx,vertices[idx]
# poly = Polygon(vtx)
# circle = Point(msh.y[i]).buffer(msh.R_extended[i])
# cell_poly = circle.intersection(poly)
# perimeters[i] = cell_poly.length
# ax.add_patch(PolygonPatch(cell_poly, ec="white", fc="green"))
#
# # Set limits and labels
# ax.set_xlim(vs_connected.min()-5,vs_connected.max()+5)
# ax.set_ylim(vs_connected.min()-5,vs_connected.max()+5)
# ax.set_aspect('equal', adjustable='datalim')  # Equal aspect ratio
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Circles from Centroids and Radii')
#
# fig.show()
#
#



triangulation_edges = np.row_stack([np.column_stack([msh.tri[:,i],msh.tri[:,(i+1)%3 ]]) for i in range(3) ])

for edge in triangulation_edges:
    if not (np.flip(edge) == triangulation_edges).all(axis=1).any():
        print(edge)


# Create Shapely circles
circles = [Point(centroid).buffer(radius) for centroid, radius in zip(msh.y[idxs], msh.R_extended[idxs])]
all_circles = [Point(centroid).buffer(radius) for centroid, radius in zip(msh.y, msh.R_extended)]



# Create a figure and axis
fig, ax = plt.subplots()
ax.scatter(*vs_connected.T,zorder=100)
for edge in triangulation_edges:
    ax.plot(*msh.y[edge].T,color="k")

for circle in all_circles:
    ax.add_patch(PolygonPatch(circle, fc='grey', ec='black', alpha=0.1))

# Plot circles using Matplotlib patches
for circle in circles:
    ax.add_patch(PolygonPatch(circle, fc='blue', ec='black', alpha=0.5))

ax.add_patch(PolygonPatch(thecircle, fc='red', ec='red', alpha=0.5))

i, vtx = idx,vertices[idx]
poly = Polygon(vtx)
circle = Point(msh.y[i]).buffer(msh.R_extended[i])
cell_poly = circle.intersection(poly)
perimeters[i] = cell_poly.length
ax.add_patch(PolygonPatch(cell_poly, ec="white", fc="green"))

# Set limits and labels
ax.set_xlim(msh.y[idxs].min()-5,msh.y[idxs].max()+5)
ax.set_ylim(msh.y[idxs].min()-5,msh.y[idxs].max()+5)
ax.set_aspect('equal', adjustable='datalim')  # Equal aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Circles from Centroids and Radii')

fig.show()

# Show the plot
# fig.savefig("plots/out.pdf",dpi=300)



"""
Multiple issues

For some reason, the perimeters of (some) edge cells are under estimated 

Then also the classification correction mystery. 


Things tried: 

- It is not a poor allocation of CW/CCW of triangles. The only edges for which a flip is not found are those connecting the boundary 
- 
-  
"""
