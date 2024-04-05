import itertools

import numpy as np
import two_dimensional_cell.tri_functions as trf
from matplotlib import pyplot as plot
from matplotlib.collections import LineCollection
from numba import jit
from scipy.spatial import ConvexHull

# --- Misc. geometry code -----------------------------------------------------

'''
Pick N points uniformly from the unit disc
This sampling algorithm does not use rejection sampling.
'''


def disc_uniform_pick(N):
    angle = (2 * np.pi) * np.random.random(N)
    out = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    out *= np.sqrt(np.random.random(N))[:, None]
    return out


@jit(nopython=True, cache=True)
def norm2(X):
    return np.sqrt(np.sum(X ** 2))


@jit(nopython=True, cache=True)
def normalized(X):
    return X / norm2(X)


@jit(nopython=True, cache=True)
def lift(x, R):
    x_norm = x[0] ** 2 + x[1] ** 2 - R ** 2
    x_out = np.zeros(3)
    x_out[:2], x_out[2] = x, x_norm
    return x_out


# --- Delaunay triangulation --------------------------------------------------



@jit(nopython=True, cache=True)
def get_triangle_normal(A, B, C):
    return normalized(np.cross(A, B) + np.cross(B, C) + np.cross(C, A))


@jit(nopython=True, cache=True)
def get_power_circumcenter(A, B, C):
    N = get_triangle_normal(A, B, C)
    return (-.5 / N[2]) * N[:2]


@jit(nopython=True, cache=True)
def is_ccw_triangle(A, B, C):
    m = np.stack((A,B,C))
    M = np.column_stack((m,np.ones(3)))
    return np.linalg.det(M) > 0


@jit(nopython=True, cache=True)
def get_circumcentre(S, R):
    return get_power_circumcenter(*get_S_lifted(S, R))


@jit(nopython=True, cache=True)
def get_S_lifted(S, R):
    S_norm = np.sum(S ** 2, axis=1) - R ** 2
    S_lifted = np.concatenate((S, S_norm.reshape(-1, 1)), axis=1)
    # S_lifted = np.concatenate((S.ravel(),S_norm)).reshape(3,-1).T
    return S_lifted


@jit(nopython=True, cache=True)
def get_vertices(S_lifted, tri_list, n_v):
    V = np.zeros((n_v, 2))
    for i, tri in enumerate(tri_list):
        A, B, C = S_lifted[tri]
        V[i] = get_power_circumcenter(A, B, C)
    return V


@jit(nopython=True, cache=True)
def build_tri_and_norm(S, simplices, equations):
    saved_tris = equations[:, 2] <= 0
    n_v = saved_tris.sum()
    norms = equations[saved_tris]
    tri_list = np.zeros((n_v, 3), dtype=np.int64)
    i = 0
    for (a, b, c), eq in zip(simplices[saved_tris], norms):
        if is_ccw_triangle(S[a], S[b], S[c]):
            tri_list[i] = a, b, c
        else:
            tri_list[i] = a, c, b
        i += 1
    return tri_list, norms, n_v


def get_power_triangulation(S, R):
    # Compute the lifted weighted points
    S_lifted = get_S_lifted(S, R)
    # Compute the convex hull of the lifted weighted points
    hull = ConvexHull(S_lifted)
    #
    # # Extract the Delaunay triangulation from the lower hull
    tri_list, norms, n_v = build_tri_and_norm(S, hull.simplices, hull.equations)
    #
    # # Compute the Voronoi points
    V = get_vertices(S_lifted, tri_list, n_v)
    #
    # # Job done
    return tri_list, V, n_v




# --- Compute Voronoi cells ---------------------------------------------------

'''
Compute the segments and half-lines that delimits each Voronoi cell
  * The segments are oriented so that they are in CCW order
  * Each cell is a list of (i, j), (A, U, tmin, tmax) where
     * i, j are the indices of two ends of the segment. Segments end points are
       the circumcenters. If i or j is set to None, then it's an infinite end
     * A is the origin of the segment
     * U is the direction of the segment, as a unit vector
     * tmin is the parameter for the left end of the segment. Can be -1, for minus infinity
     * tmax is the parameter for the right end of the segment. Can be -1, for infinity
     * Therefore, the endpoints are [A + tmin * U, A + tmax * U]
'''


def get_voronoi_cells(S, V, tri_list):
    # Keep track of which circles are included in the triangulation
    vertices_set = frozenset(itertools.chain(*tri_list))

    # Keep track of which edge separate which triangles
    edge_map = {}
    for i, tri in enumerate(tri_list):
        for edge in itertools.combinations(tri, 2):
            edge = tuple(sorted(edge))
            if edge in edge_map:
                edge_map[edge].append(i)
            else:
                edge_map[edge] = [i]

    # For each triangle
    voronoi_cell_map = {i: [] for i in vertices_set}

    for i, (a, b, c) in enumerate(tri_list):
        # For each edge of the triangle
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
            # Finite Voronoi edge
            edge = tuple(sorted((u, v)))
            if len(edge_map[edge]) == 2:
                j, k = edge_map[edge]
                if k == i:
                    j, k = k, j

                # Compute the segment parameters
                U = V[k] - V[j]
                U_norm = norm2(U)

                # Add the segment
                voronoi_cell_map[u].append(((j, k), (V[j], U / U_norm, 0, U_norm)))
            else:
                # Infinite Voronoi edge
                # Compute the segment parameters
                A, B, C, D = S[u], S[v], S[w], V[i]
                U = normalized(B - A)
                I = A + np.dot(D - A, U) * U
                W = normalized(I - D)
                if np.dot(W, I - C) < 0:
                    W = -W

                # Add the segment
                voronoi_cell_map[u].append(((edge_map[edge][0], -1), (D, W, 0, None)))
                voronoi_cell_map[v].append(((-1, edge_map[edge][0]), (D, -W, None, 0)))

    # Order the segments
    def order_segment_list(segment_list):
        # Pick the first element
        first = min((seg[0][0], i) for i, seg in enumerate(segment_list))[1]

        # In-place ordering
        segment_list[0], segment_list[first] = segment_list[first], segment_list[0]
        for i in range(len(segment_list) - 1):
            for j in range(i + 1, len(segment_list)):
                if segment_list[i][0][1] == segment_list[j][0][0]:
                    segment_list[i + 1], segment_list[j] = segment_list[j], segment_list[i + 1]
                    break

        # Job done
        return segment_list

    # Job done
    return {i: order_segment_list(segment_list) for i, segment_list in voronoi_cell_map.items()}


# --- Plot all the things -----------------------------------------------------

def display(ax, S, R, tri_list, voronoi_cell_map,tri_alpha=0,n_b=4,xlim=None,ylim=None,line_col="white"):
    # Setup
    plot.axis('equal')
    plot.axis('off')

    # Set min/max display size, as Matplotlib does it wrong
    min_corner = np.amin(S[:-n_b], axis=0) - np.max(R)
    max_corner = np.amax(S[:-n_b], axis=0) + np.max(R)
    if xlim is None:
        xlim = (min_corner[0], max_corner[0])
    if ylim is None:
        ylim = (min_corner[1], max_corner[1])
    plot.xlim(xlim)
    plot.ylim(ylim)

    # Plot the samples
    for Si, Ri in zip(S[:-n_b], R[:-n_b]):
        ax.add_artist(plot.Circle(Si, Ri, fill=True, alpha=1, lw=0., color='#8080f0', zorder=1))

    # Plot the power triangulation
    edge_set = frozenset(tuple(sorted(edge)) for tri in tri_list for edge in itertools.combinations(tri, 2))
    line_list = LineCollection([(S[i], S[j]) for i, j in edge_set], lw=1., colors="k",alpha=tri_alpha, zorder=1000)
    ax.add_collection(line_list)

    # Plot the Voronoi cells
    edge_map = {}
    for segment_list in voronoi_cell_map.values():
        for edge, (A, U, tmin, tmax) in segment_list:
            edge = tuple(sorted(edge))
            if edge not in edge_map:
                if tmax is None:
                    tmax = 10
                if tmin is None:
                    tmin = -10

                edge_map[edge] = (A + tmin * U, A + tmax * U)

    line_list = LineCollection(edge_map.values(), lw=1., colors=line_col, zorder=900)
    # line_list.set_zorder(0)
    ax.add_collection(line_list)

    # Job done



"""
one_in = (tri < n_c).any(axis=1)
new_tri = tri[one_in]

# 4. Remove repeats in new_tri
#   new_tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
#   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
#   cell ids, which are transformed by the mod function to account for periodicity. See function for more details
n_tri = self.remove_repeats(new_tri, n_c)








"""


def remove_repeats(tri, n_c):
    tri = order_tris(np.mod(tri, n_c))
    sorted_tri = tri[np.lexsort(tri.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_tri, axis=0), 1))
    return sorted_tri[row_mask]

@jit(nopython=True)
def order_tris(tri):
    """
    For each triangle (i.e. row in **tri**), order cell ids in ascending order
    :param tri: Triangulation (n_v x 3) np.int32 array
    :return: the ordered triangulation
    """
    nv = tri.shape[0]
    for i in range(nv):
        Min = np.argmin(tri[i])
        tri[i] = tri[i,Min],tri[i,np.mod(Min+1,3)],tri[i,np.mod(Min+2,3)]
    return tri