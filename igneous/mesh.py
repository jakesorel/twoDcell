import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import jax
from scipy import sparse
import triangle as tr
from scipy.spatial import ConvexHull
from igneous.power_triangulation import triangulate
from matplotlib.patches import Polygon
from shapely.geometry import Polygon, Point ##ensure it is v.1.7.1
from descartes import PolygonPatch
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import pandas as pd
import matplotlib.pyplot as plt
class Mesh:
    def __init__(self,mesh_params=None):
        assert mesh_params is not None, "specify mesh_params"
        self.mesh_params = mesh_params
        self.mesh_props = {}
        self.initialize()


    def initialize(self):
        self.provide_default_params()
        self.load_L()

    def provide_default_params(self):
        if "equiangulate" not in self.mesh_params:
            self.mesh_params["equiangulate"] = False
        if "n_equi_kill" not in self.mesh_params:
            self.mesh_params["n_equi_kill"] = 10
        if "periodic_expansion_mode" not in self.mesh_params:
            self.mesh_params["periodic_expansion_mode"] = "R_mult"
        if "R_mult" not in self.mesh_params:
            self.mesh_params["R_mult"] = 3.

    def load_L(self):
        """
        Load L, the periodic square box size, into mesh_props, specified from mesh_params
        """
        assert "L" in self.mesh_params, "mesh_params must contain an entry for L (float)"
        self.mesh_props["L"] = self.mesh_params["L"]

    def load_X(self,x,R,triangulate=True):
        """
        load x (cell centres) and R (cell radii) into the mesh_props dictionary
        """
        self.mesh_props["x"] = x
        self.mesh_props["R"] = R
        self.mesh_props["n_c"] = int(self.mesh_props["x"].shape[0])
        if triangulate:
            self.triangulate()

    def triangulate(self):
        """
        Triangulate using periodic boundary x,R to generate a power triangulation.

        This has the option to equiangulate (not implemented)
        Or to triangulate using the Convex Hull method.

        """
        if self.mesh_params["equiangulate"]:
            ##Perform equiangulation
            assert not self.mesh_params["equiangulate"], "equiangulation not implemented"
        else:
            self._triangulate()

        self.mesh_props = get_geometry(self.mesh_props,int(self.mesh_props["n_c"]))

    def _triangulate(self):
        """
        1. Define the periodic expansion mode. This copies cells around the outside of the box to facilitate triangulation.
        Either this can be "R_mult"

        2. Generate the triangulation mask. This is a transformation to x,R copying cell centres around the box to make
        it compatible for triangulation.

        NB: we will get problematic results in very sparse scenarios, where the consequent triangulation may not fully
        span the 360 degrees of each cell. I have previously used 'ghost cells' to deal with this, but this wasn't in a
        periodic context. Let's watch out for this.

        """

        if self.mesh_params["periodic_expansion_mode"] == "R_mult":
            max_d = self.mesh_props["R"].max()*self.mesh_params["R_mult"]
            if max_d > self.mesh_props["L"]:
                max_d = self.mesh_props["L"]
        else:
            max_d = self.mesh_props["L"]

        x_hat,R_hat,dictionary = generate_triangulation_mask(self.mesh_props["x"],self.mesh_props["R"],self.mesh_props["L"],max_d)
        self.mesh_props["x_hat"],self.mesh_props["R_hat"],self.mesh_props["dictionary"] = x_hat,R_hat,dictionary
        # _x_hat,_R_hat,dictionary = generate_triangulation_mask(self.mesh_props["x"],self.mesh_props["R"],self.mesh_props["L"],max_d)

        # L = self.mesh_props["L"]
        # x_hat = jnp.row_stack([_x_hat,jnp.array([[-max_d*2,-max_d*2],
        #                        [-max_d*2,L+max_d*2],
        #                        [L+max_d*2,-max_d*2],
        #                        [L+max_d*2,L+max_d*2]])])
        # R_hat = jnp.concatenate([_R_hat,jnp.array([1e-3,1e-3,1e-3,1e-3])])


        # 2. Perform the power triangulation on x_hat, R_hat, using the **power_triangulation** module.
        # tri_hat = triangulate(x_hat,R_hat)
        ##For now use the voronoi triangulation. This assumes that radii are identical.
        tri_hat = tr.triangulate({"vertices":np.array(x_hat)})["triangles"]

        # 3. Find triangles with **at least one** cell within the "true" frame (i.e. with **at least one** "normal cell")
        #   (Ignore entries with -1, a quirk of the **triangle** package, which denotes boundary triangles
        #   Generate a mask -- one_in -- that considers such triangles
        #   Save the new triangulation by applying the mask -- new_tri
        tri_hat = tri_hat[(tri_hat != -1).all(axis=1)]
        one_in = (tri_hat < self.mesh_props["n_c"]).any(axis=1)
        tri = tri_hat[one_in]

        # 4. Remove repeats in new_tri
        #   new_tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
        #   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
        #   cell ids, which are transformed by the mod function to account for periodicity. See function for more details

        tri = dictionary[tri]

        self.mesh_props["tri"] = remove_repeats(tri, self.mesh_props["n_c"])

        self.mesh_props["n_v"] = self.mesh_props["tri"].shape[0]

        self.mesh_props["neigh"] = get_neighbours(self.mesh_props["tri"])
        self.mesh_props["k2s"] = get_k2(self.mesh_props["tri"],self.mesh_props["neigh"])




def generate_triangulation_mask(x,R,L,max_d):
    """
    Copy cells within the region of max_d from the boundary to the relevant places outside of the box.

    This makes a larger array x_hat where various x values have been duplicated and transposed by (±L,±L)

    dictionary keeps a record of the transformation.

    Rs are then also duplicated using dictionary.

    This should be JAXed in due course for efficiency.
    """
    x_hat = np.zeros((0,2))
    dictionary = np.zeros((0))
    for i in [0,-1,1]:
        for j in [0,-1,1]:
            y = (x + np.array((i, j)) * L)
            if j == 0:
                if i == 0:
                    mask = np.ones_like(x[:,0],dtype=np.bool_)
                else:
                    val = L*(1-i)/2
                    mask = np.abs(x[:,0]-val)<max_d
            elif i == 0:
                val = L * (1 - j) / 2
                mask = np.abs(x[:, 1] - val) < max_d
            else:
                val_x = L * (1 - i) / 2
                val_y = L * (1 - j) / 2
                mask = np.sqrt((x[:,0]-val_x)**2 + (x[:,1]-val_y)**2) < max_d
            x_hat = np.row_stack((x_hat,y[mask]))
            dictionary = np.concatenate((dictionary,np.nonzero(mask)[0])).astype(int)
    R_hat = R[dictionary]
    return x_hat,R_hat,dictionary



def order_tris(tri):
    """
    For each triangle (i.e. row in **tri**), order cell ids in ascending order
    :param tri: Triangulation (n_v x 3) np.int32 array
    :return: the ordered triangulation
    """
    nv = tri.shape[0]
    for i in range(nv):
        Min = np.argmin(tri[i])
        tri[i] = tri[i, Min], tri[i, np.mod(Min + 1, 3)], tri[i, np.mod(Min + 2, 3)]
    return tri


def remove_repeats(tri, n_c):
    """
    For a given triangulation (nv x 3), remove repeated entries (i.e. rows)
    The triangulation is first re-ordered, such that the first cell id referenced is the smallest. Achieved via
    the function order_tris. (This preserves the internal order -- i.e. CCW)
    Then remove repeated rows via lexsort.
    NB: order of vertices changes via the conventions of lexsort
    Inspired by...
    https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array
    :param tri: (nv x 3) matrix, the triangulation
    :return: triangulation minus the repeated entries (nv* x 3) (where nv* is the new # vertices).
    """
    tri = order_tris(np.mod(tri, n_c))
    sorted_tri = tri[np.lexsort(tri.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_tri, axis=0), 1))
    return sorted_tri[row_mask]



def get_neighbours(tri, neigh=None, Range=None):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.

    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.

    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri.shape[0]
    if neigh is None:
        neigh = np.ones_like(tri, dtype=np.int32) * -1
    if Range is None:
        Range = np.arange(n_v)

    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in Range:  # range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip, tri_sample_flip)).reshape(3, 2)
        for k in range(3):
            if neigh[j, k] == -1:
                msk = (tri_compare[:, :, 0] == tri_i[k, 0]) * (tri_compare[:, :, 1] == tri_i[k, 1])
                if msk.sum() > 0:
                    neighb, l = np.nonzero(msk)
                    neighb, l = neighb[0], l[0]
                    neigh[j, k] = neighb
                    neigh[neighb, np.mod(2 - l, 3)] = j
    return neigh



def get_k2(tri, neigh):
    """
    To determine whether a given neighbouring pair of triangles needs to be re-triangulated, one considers the sum of
    the pair angles of the triangles associated with the cell centroids that are **not** themselves associated with the
    adjoining edge. I.e. these are the **opposite** angles.

    Given one cell centroid/angle in a given triangulation, k2 defines the column index of the cell centroid/angle in the **opposite** triangle

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: Neighbourhood matrix (n_v x 3) np.int32 array
    :return:
    """
    three = np.array([0, 1, 2])
    nv = tri.shape[0]
    k2s = np.empty((nv, 3), dtype=np.int32)
    for i in range(nv):
        for k in range(3):
            neighbour = neigh[i, k]
            k2 = ((neigh[neighbour] == i) * three).sum()
            k2s[i, k] = k2
    return k2s


@jit
def periodic_displacement(x1, x2, L):
    """
    Periodic displacement of "x1-x2" in a square periodic box of shape L x L
    """
    return jnp.mod(x1 - x2 + L / 2, L) - L / 2


@partial(jit, static_argnums=(1,))
def get_geometry(mesh_props,n_c):
    """
    A wrapper for the various calculations to be performed on the triangulation.

    """
    mesh_props = tri_format(mesh_props)
    mesh_props = classify_edges(mesh_props)
    mesh_props = get_circle_intersections(mesh_props)
    mesh_props = update_classification(mesh_props)
    mesh_props = get_h(mesh_props)
    mesh_props = calculate_angles(mesh_props, n_c)
    mesh_props = calculate_perimeters(mesh_props, n_c)
    mesh_props = calculate_areas(mesh_props, n_c)
    return mesh_props

@jit
def tri_format(mesh_props):
    """
    tx is x in triangulated form. if x -> (nc, 2) and tri -> (nv, 3) then tx -> (nv,3,2)

    tR is likewise for R.

    tx_p1 is tx shifted one place counter clockwise over the triplet. Read "p1" as plus 1

    tx_m1 is tx shifted one place clockwise over the triplet.

    likewise for tR_p1, tR_m1

    neigh_p1 is the neighbouring cell index to the one in tri[i,j], rolled CCW with respect to the centre of the cell tri[i,j].

    neigh_m1 is the same but CW

    v_p1 is the CCW vertex with respect to the cell tri[i,j].

    v_m1 is the same but CW

    v_x is the periodic displacement from the cell centre in tx to the corresponding vertex.

    lv_x is the corresponding distance to v_x.

    tx_tx_p1 is the periodic displacement to tx from tx_p1

    lx_p1 is the corresponding distance

    tx_tx_m1 is the periodic displacement to tx from tx_m1

    lx_m1 is the corresponding distance

    Certain metrics calculated may not be necessary for later calculations so let's check for efficiency.

    """
    mesh_props["tx"] = mesh_props["x"][mesh_props["tri"]]
    mesh_props["tR"] = mesh_props["R"][mesh_props["tri"]]

    mesh_props["tx_p1"] = jnp.roll(mesh_props["tx"],-1,axis=1)
    mesh_props["tx_m1"] = jnp.roll(mesh_props["tx"],1,axis=1)
    mesh_props["tR_p1"] = jnp.roll(mesh_props["tR"],-1,axis=1)
    mesh_props["tR_m1"] = jnp.roll(mesh_props["tR"],1,axis=1)



    mesh_props = get_power_circumcenter(mesh_props)
    mesh_props["neigh_p1"] = jnp.roll(mesh_props["neigh"], -1, axis=1)
    mesh_props["neigh_m1"] = jnp.roll(mesh_props["neigh"], 1, axis=1)
    mesh_props["k2s_p1"] = jnp.roll(mesh_props["k2s"], -1, axis=1)
    mesh_props["k2s_m1"] = jnp.roll(mesh_props["k2s"], 1, axis=1)


    mesh_props["v_p1"] = mesh_props["v"][mesh_props["neigh_p1"],(mesh_props["k2s_p1"]+1)%3]
    mesh_props["v_m1"] = mesh_props["v"][mesh_props["neigh_m1"],(mesh_props["k2s_m1"]-1)%3]

    mesh_props["v_x"] = mesh_props["v"]-mesh_props["tx"]
    mesh_props["v_p1_x"] = mesh_props["v_p1"]-mesh_props["tx"]
    mesh_props["v_m1_x"] = mesh_props["v_m1"]-mesh_props["tx"]

    mesh_props["lv_x"] = jnp.linalg.norm(mesh_props["v_x"],axis=-1)
    mesh_props["lv_p1_x"] = jnp.linalg.norm(mesh_props["v_p1_x"],axis=-1)
    mesh_props["lv_m1_x"] = jnp.linalg.norm(mesh_props["v_m1_x"],axis=-1)

    mesh_props["tx_tx_p1"] = periodic_displacement(mesh_props["tx"],mesh_props["tx_p1"],mesh_props["L"])
    mesh_props["tx_tx_m1"] = periodic_displacement(mesh_props["tx"],mesh_props["tx_m1"],mesh_props["L"])
    mesh_props["lx_p1"] = jnp.linalg.norm(mesh_props["tx_tx_p1"],axis=-1)
    mesh_props["lx_m1"] = jnp.linalg.norm(mesh_props["tx_tx_m1"],axis=-1)





    return mesh_props


@jit
def _get_circumcenter_i(txi,tRi,L):
    """
    Circumcentre calculation, taken from Mathematica. Perhaps this could be tidied up somewhat...

    Strategy to handle periodic boundary conditions:
    - For each triple of x positions, take the first and find the periodic displacement to the second and third.
    - Then add the first back once the vertex is found (and mod it)
    """
    txi_0 = txi[0]
    txi_zeroed = periodic_displacement(txi,jnp.expand_dims(txi_0,axis=0),L)
    (xi,yi),(xj,yj),(xk,yk) = txi_zeroed
    Ri,Rj,Rk = tRi
    denom = (2. * (xk * (-yi + yj) + xj * (yi - yk) + xi * (-yj + yk)))
    vx = (xj ** 2 * yi - xk ** 2 * yi + Rk ** 2 * (yi - yj) + xk ** 2 * yj - yi ** 2 * yj + yi * yj ** 2 + (
                    Ri - xi) * (Ri + xi) * (yj - yk) - (xj ** 2 - yi ** 2 + yj ** 2) * yk + (
                            -yi + yj) * yk ** 2 + Rj ** 2 * (-yi + yk))
    vy = (xi ** 2 * xj - xi * xj ** 2 + Rk ** 2 * (-xi + xj) + Rj ** 2 * (
                    xi - xk) - xi ** 2 * xk + xj ** 2 * xk + xi * xk ** 2 - xj * xk ** 2 + Ri ** 2 * (
                            -xj + xk) + xj * yi ** 2 - xk * yi ** 2 - xi * yj ** 2 + xk * yj ** 2 + (
                            xi - xj) * yk ** 2)
    v = jnp.array((vx,vy))/denom
    return v



@jit
def _get_circumcenter_non_periodic(txi,tRi):
    """
    Circumcentre calculation, taken from Mathematica. Perhaps this could be tidied up somewhat...

    Strategy to handle periodic boundary conditions:
    - For each triple of x positions, take the first and find the periodic displacement to the second and third.
    - Then add the first back once the vertex is found (and mod it)
    """
    (xi,yi),(xj,yj),(xk,yk) = txi
    Ri,Rj,Rk = tRi
    denom = (2. * (xk * (-yi + yj) + xj * (yi - yk) + xi * (-yj + yk)))
    vx = (xj ** 2 * yi - xk ** 2 * yi + Rk ** 2 * (yi - yj) + xk ** 2 * yj - yi ** 2 * yj + yi * yj ** 2 + (
                    Ri - xi) * (Ri + xi) * (yj - yk) - (xj ** 2 - yi ** 2 + yj ** 2) * yk + (
                            -yi + yj) * yk ** 2 + Rj ** 2 * (-yi + yk))
    vy = (xi ** 2 * xj - xi * xj ** 2 + Rk ** 2 * (-xi + xj) + Rj ** 2 * (
                    xi - xk) - xi ** 2 * xk + xj ** 2 * xk + xi * xk ** 2 - xj * xk ** 2 + Ri ** 2 * (
                            -xj + xk) + xj * yi ** 2 - xk * yi ** 2 - xi * yj ** 2 + xk * yj ** 2 + (
                            xi - xj) * yk ** 2)
    v = jnp.array((vx,vy))/denom
    return v


@jit
def get_power_circumcenter(mesh_props):
    """
    Calculates the power circumcentre, respecting periodic boundary conditions for all of the triples in the triangulation.

    Please beware that the periodic implementation is in 'beta' and may need 'refinement'.

    Inserts the value into the mesh_props dictionary.
    The vertices need not lie within the bounds of the box. This is useful for ensuring all cells are surrounded completely by other vertices.

    _v is the displacement with respect to cell 1 in the triplet, whereas v is the actual position with respect to all 3.
    Note that mod(v,L) will be identical for all three cells. This is just a strategy to ensure cells are surrounded by
    vertices in some meaningful orientation.

    """
    mesh_props["_v"] = vmap(_get_circumcenter_i,in_axes=(0,0,None))(mesh_props["tx"],mesh_props["tR"],mesh_props["L"])
    mesh_props["v"] = periodic_displacement(jnp.expand_dims(mesh_props["tx"][:, 0], 1),mesh_props["tx"], mesh_props["L"]) + jnp.expand_dims(mesh_props["_v"],axis=1) + mesh_props["tx"]

    return mesh_props



@jit
def classify_edges(mesh_props):
    """
    Classify edges based on whether the power circumcentre lies inside or outside the radius of the cell's circle.

    V_in (n_v x 3) determines whether the **indexed** vertex lies inside the cell radius corresponding to tri[i,j].

    no_touch determines if the pair of circles intersect at all (i.e. true if they don't).
    """

    mesh_props["V_in"] = mesh_props["lv_x"] < mesh_props["tR"]
    # mesh_props["V_in_p1"] = mesh_props["lv_p1_x"] < mesh_props["tR"]
    # mesh_props["V_in_m1"] = mesh_props["lv_m1_x"] < mesh_props["tR"]

    return mesh_props



@jit
def get_circle_intersections(mesh_props):
    """
    Intersect the pair of circles corresponding to each cell in the triangulation and its plus 1 neighbour (i.e. CCW neighbour).

    h_mid is the mid point of the two circles (i.e. equidistant from both of the perimeters).

    NB: in principle it is degenerate to calculate equivalently the minus 1 variant of this, given all edges appear twice in the triangulation
    However, it may be necessary to inspect both instances of an edge to allocate the class of that edge.

    NB: h_CCW is the intersection between cell i and the CCW cell in the triangulation, that is counter clockwise (i.e. facing towards the power vertex of the triple)

    NB: in cases where the discriminant is less than zero, h_CCW will lie outside of the cell. This can be fixed by projection, but I don't think it'll matter for later calculations. However do pay attention.
    """
    tx_p1_registered = mesh_props["tx"] -mesh_props["tx_tx_p1"]

    parallel_component = 0.5 * (mesh_props["tx"] + tx_p1_registered) + \
        jnp.expand_dims((mesh_props["tR"] ** 2 - mesh_props["tR_p1"] ** 2) / (2 * mesh_props["lx_p1"] ** 2),2) * (-mesh_props["tx_tx_p1"])


    _discriminant = 2 * (mesh_props["tR"] ** 2 + mesh_props["tR_p1"] ** 2) / mesh_props["lx_p1"] ** 2 - ((mesh_props["tR"] ** 2 - mesh_props["tR_p1"] ** 2) ** 2) / mesh_props["lx_p1"] ** 4 - 1
    mesh_props["no_touch"] = _discriminant<0
    discriminant = jnp.clip(_discriminant,0,jnp.inf) ##When the discriminant is zero, put


    perpendicular_component = 0.5 * jnp.expand_dims(jnp.sqrt(discriminant),2) * jnp.dstack((tx_p1_registered[...,1] - mesh_props["tx"][...,1], mesh_props["tx"][...,0] - tx_p1_registered[...,0]))
    #
    # _discriminant2 = 4*mesh_props["lx_p1"]**2 * mesh_props["tR_p1"]**2 - (mesh_props["lx_p1"]**2 - mesh_props["tR"]**2 + mesh_props["tR_p1"]**2)**2
    # discriminant2 = jnp.clip(_discriminant2,0,jnp.inf) ##When the discriminant is zero, put
    #
    #
    # a = (1/mesh_props["lx_p1"])*jnp.sqrt(discriminant2)/2
    # # perpendicular_direction = mesh_props["v"]-parallel_component
    # # perpendicular_direction /= jnp.expand_dims(jnp.linalg.norm(perpendicular_direction,axis=-1),-1)
    #
    # perpendicular_direction = jnp.flip(mesh_props["tx_tx_p1"],axis=-1)*jnp.array([-1,1])
    # perpendicular_direction /= jnp.expand_dims(jnp.linalg.norm(perpendicular_direction,axis=-1),-1)
    #
    # perpendicular_component2 = perpendicular_direction*jnp.expand_dims(a,axis=-1)

    mesh_props["h_mid"] = parallel_component
    mesh_props["h_CCW"] = parallel_component - perpendicular_component
    # mesh_props["h_CW"] = parallel_component + perpendicular_component
    return mesh_props

@jit
def update_classification(mesh_props,eps=1e-7):
    """
    There are annoying edge cases where, in a triple, all three circles do not overlap with the power circumcentre,
    yet one of the circle-circle intersects overlaps with the third circle.

    This identifies the scenarios where this occurs, then fixes these scenarios (for both references of the vertex in question).
    Vertices are moved to the "mid" point, such that the measured distances from hp to hm become zero (and the angles)

    """
    ##beware, a periodic displacement here
    h_CCW_tx_m1 = periodic_displacement(mesh_props["h_CCW"],mesh_props["tx_m1"],mesh_props["L"])
    l2_h_x_m1 = ((h_CCW_tx_m1)**2).sum(axis=-1)
    m1_correction = (l2_h_x_m1-mesh_props["tR_m1"]**2) < eps
    mesh_props["no_touch"] += (~mesh_props["V_in"])*(m1_correction)
    mesh_props["no_touch"] += mesh_props["no_touch"][mesh_props["neigh_m1"], ((mesh_props["k2s_m1"] + 1) % 3)]
    mesh_props["no_touch"] = jnp.expand_dims(mesh_props["no_touch"],axis=2)
    mesh_props["h_CCW"] = mesh_props["h_CCW"]*(~mesh_props["no_touch"]) + mesh_props["h_mid"]*mesh_props["no_touch"]
    return mesh_props

@jit
def get_h(mesh_props):
    """
    h_p is the effective vertex, which is either the power circumcentre or the circle circle intersection pointing CCW
    h_m is the equivalent pointing  clockwise. This is just a re-indexing of h_p
    h_p_x is the distance from tx to h_p
    and likewise for h_m_x

    """
    mesh_props["h_p"] = jnp.expand_dims(mesh_props["V_in"],2)*mesh_props["v"] + (~jnp.expand_dims(mesh_props["V_in"],2))*mesh_props["h_CCW"]
    h_m = mesh_props["h_p"][mesh_props["neigh_m1"],(mesh_props["k2s_m1"]+1)%3]
    mesh_props["h_m"] = periodic_displacement(h_m, mesh_props["h_p"],mesh_props["L"])+mesh_props["h_p"]
    mesh_props["h_p_x"] = mesh_props["h_p"] - mesh_props["tx"]
    mesh_props["h_m_x"] = mesh_props["h_m"] - mesh_props["tx"]

    return mesh_props

@jit
def tvecangle(a, b):
    """
    Signed angle between two (triangle form) sets of vectors
    """
    return jnp.arctan2(jnp.cross(a, b,axis=-1), jnp.sum(a*b,axis=-1))


@partial(jit, static_argnums=(2,))
def assemble_scalar(tval, tri, n_c):
    """
    Given a set of scalar values of shape (nv x 3) and a triangulation (nv x 3)
    Sum over all of the components containing a given index to give the total by index.
    """
    val = jnp.bincount(tri.ravel() + 1, weights=tval.ravel(), length=n_c + 1)
    val = val[1:]
    return val

@partial(jit, static_argnums=(1,))
def calculate_angles(mesh_props,n_c):
    """
    t_theta is the angle between the 'effective vertex h' between cell tri[i,j] and its CCW neighbour,
    with respect to the centre of that cell.

    t_theta is the triangle-by-triangle components and
    theta is the sum over all triangles.

    phi is the amonunt of angle left over from 2pi, comprising the curved component of the cell.

    """

    mesh_props["t_theta"] = tvecangle(mesh_props["h_m_x"],mesh_props["h_p_x"])
    mesh_props["theta"] = assemble_scalar(mesh_props["t_theta"],mesh_props["tri"],n_c)
    mesh_props["phi"] = jnp.pi*2 - mesh_props["theta"]
    return mesh_props

@partial(jit, static_argnums=(1,))
def calculate_areas(mesh_props,n_c):
    """
    A_C is the curved area comprised of the sum of circular sectors (the angles made between h_p and v)
    A_S is the "straight" area comprised of triangles {x,h_m, h_p} over all triangles present for a cell.

    """
    mesh_props["A_C"] = 0.5*mesh_props["phi"]*mesh_props["R"]**2
    mesh_props["tA_S"] = 0.5*jnp.cross(mesh_props["h_m_x"],mesh_props["h_p_x"])
    mesh_props["A_S"] = assemble_scalar(mesh_props["tA_S"],mesh_props["tri"],n_c)
    mesh_props["A"] = mesh_props["A_C"] + mesh_props["A_S"]
    return mesh_props

@partial(jit, static_argnums=(1,))
def calculate_perimeters(mesh_props,n_c):
    """
    P_C is the curved perimeter comprised of the sum of circular sectors (the angles made between h_p and v)
    P_S is the "straight" perimeter comprised of triangles {x,h_m, h_p} over all triangles present for a cell.

    NB: in calculating tP_S, in cases where theta < 0, then this should be a signed distance. I have included a scaling using jnp.sign.
    Equally one could do the trigonometry using signed angles as an exercise. It should be identical.
    """
    mesh_props["P_C"] = mesh_props["phi"]*mesh_props["R"]
    mesh_props["tP_S"] = jnp.linalg.norm(mesh_props["h_m"]-mesh_props["h_p"],axis=-1)*jnp.sign(mesh_props["t_theta"])
    mesh_props["P_S"] = assemble_scalar(mesh_props["tP_S"],mesh_props["tri"],n_c)
    mesh_props["P"] = mesh_props["P_C"] + mesh_props["P_S"]

    return mesh_props

def get_by_cell_vertices_from_mesh(x_hat,R_hat):
    tri = tr.triangulate({"vertices":np.array(x_hat)})["triangles"] ##NOTE THAT THIS IS FOR VORONOI. ADAPT FOR POWER
    v_extended = vmap(_get_circumcenter_non_periodic)(x_hat[tri],R_hat[tri])

    vertex_list = []
    for i in range(int(np.max(tri)+1)):
        vertex_list += [np.array(v_extended[(tri==i).any(axis=1)])]
    vertex_list = list(map(sort_points_in_polygon,vertex_list))

    return vertex_list


def sort_points_in_polygon(vtx):
    if len(vtx)>0:
        disp = vtx.copy()
        disp[:,0] = disp[:,0] - vtx[:,0].mean()
        disp[:,1] = disp[:,1] - vtx[:,1].mean()
        order = np.argsort(np.arctan2(disp[:,1],disp[:,0]))

        return np.column_stack((vtx[:,0].take(order),vtx[:,1].take(order)))
    else:
        return np.array(((0,0),(0,0),(0,0)))

def plot_mesh(ax, x, R,L,cols=None, cbar=None,max_d=None, **kwargs):
    if max_d is None:
        max_d = L
    x_hat, R_hat, dictionary = generate_triangulation_mask(x,R,L, max_d)

    if cols is None:
        cols = np.repeat("grey", x.shape[0])
    if (type(cols) is not list) and (type(cols) is not np.ndarray):
        cols = np.repeat(cols, x.shape[0])

    cols_print = cols.take(dictionary)
    vertices = get_by_cell_vertices_from_mesh(x_hat,R_hat)
    for i,vtx in enumerate(vertices):
        # patches.append(Polygon(vertices[region], True, facecolor=cols_print[i], ec=(1, 1, 1, 1), **kwargs))
        poly = Polygon(vtx)
        circle = Point(x_hat[i]).buffer(R_hat[i])
        cell_poly = circle.intersection(poly)
        if cell_poly.area != 0:
            ax.add_patch(PolygonPatch(cell_poly, ec="white", fc=cols_print[i]))

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



if __name__ == "__main__":

    ##For now, only works for R equal (but only re the triangulation, the rest is fowards compatible)
    ##We should try pyvoro

    ##Make a set of points
    x = hexagonal_lattice(4, 4, 1e-1)
    x -= x.min()
    x /= x.max()
    x = np.mod(x, 1)
    R = np.ones(len(x)) * 0.1

    ##Make a mesh
    mesh_params = {"L": 1, "R_mult": 1e4}
    msh = Mesh(mesh_params)
    msh.load_X(x, R)


    ##Print some summaries to show how it works
    keys = ["A","A_S","A_C","P","P_S","P_C"]
    df_cell_props = pd.DataFrame(dict(zip(keys, [msh.mesh_props[key] for key in keys])))
    print(df_cell_props)

    ##Plot a mesh

    fig, ax = plt.subplots()
    plot_mesh(ax, x, R, mesh_params["L"], cols=None, cbar=None, max_d=mesh_params["L"])
    fig.show()


    ##It should be feasible now to write an energy functional with an input from mesh_props and differentiate it.

