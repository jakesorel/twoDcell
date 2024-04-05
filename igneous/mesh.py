import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import jax
from scipy import sparse
import triangle as tr
from scipy.spatial import ConvexHull
from igneous.power_triangulation import triangulate

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
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
    discriminant = jnp.clip(_discriminant,0,jnp.inf) ##When the discriminant is zero, put
    perpendicular_component = 0.5 * jnp.expand_dims(jnp.sqrt(discriminant),2) * jnp.dstack((tx_p1_registered[...,1] - mesh_props["tx"][...,1], mesh_props["tx"][...,0] - tx_p1_registered[...,0]))

    # mesh_props["h_mid"] = parallel_component
    mesh_props["h_CCW"] = parallel_component - perpendicular_component
    # mesh_props["h_CW"] = parallel_component + perpendicular_component
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



"""
Occasionally, vertices are mis-placed. 
"""


if __name__ == "__main__":

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


    msh = Mesh(mesh_params={"L":1,"R_mult":1e4})
    x = hexagonal_lattice(4,4,1e-1)
    x -= x.min()
    # x/= x.max(axis=0)
    x/= x.max()

    x += 0.3

    """
    There seems to be a mis asignment of the circle circle intercept in rare cases 
    
    Certain values of t_theta are wrongly set to nonzero 
    It's hm rather than hp that is the issue. 
    Seems like hm should be set to zero.. 
    
    The V_in value probably needs to be checked on both ends on the tri. 
    I.e. try: 
    Calculate V_in and V_out, then cross compare, and take the product. 
    """
    x = np.mod(x,1)

    # x = np.random.uniform(0,1,(20,2))
    R = np.ones(len(x))*0.2
    msh.load_X(x,R)

    x_hat, R_hat, dictionary = generate_triangulation_mask(x,R,1, 1)

    _x = np.mgrid[0:1:0.001,0:1:0.001].transpose(1,2,0)
    ims = np.zeros((len(x_hat),_x.shape[0],_x.shape[1]))
    for i in range(len(x_hat)):
        ims[i] = np.sum((_x-x_hat[i])**2,axis=-1) - R_hat[i]**2

    assignment = np.argmin(ims,axis=0)
    assignment = dictionary[assignment].astype(np.float)
    assignment[ims.min(axis=0)>0] = np.nan
    mesh_props = msh.mesh_props

    empirical_areas = np.zeros((len(x)))
    for i in range(len(x)):
        empirical_areas[i] = (assignment==i).sum()
    plt.scatter(empirical_areas*(0.001**2),msh.mesh_props["A"])
    plt.show()

    mesh_props["t_zeta"] = tvecangle(mesh_props["v_x"], mesh_props["v_p1_x"])
    mesh_props["zeta"] = assemble_scalar(mesh_props["t_zeta"], mesh_props["tri"], n_c)

    np.argsort(np.abs(empirical_areas*(0.001**2)-msh.mesh_props["A"]))
    mesh_props["h_m_x"][mesh_props["tri"] == 29] - mesh_props["h_p_x"][mesh_props["tri"] == 29]

    hm_hp = jnp.array([mesh_props["h_m"][mesh_props["tri"]==1],mesh_props["h_p"][mesh_props["tri"]==1]])
    hp_v = jnp.array([mesh_props["h_m"][mesh_props["tri"]==1],mesh_props["v"][mesh_props["tri"]==1]])

    fig, ax = plt.subplots()
    ax.scatter(*x_hat.T)
    ax.plot(hm_hp[..., 0], hm_hp[..., 1],zorder=10000,color="darkred")
    ax.plot(hp_v[..., 0], hp_v[..., 1],zorder=10000,color="darkblue")

    ax.scatter(*mesh_props["h_p"][mesh_props["tri"]==1].T,zorder=1000,color="darkred",s=50)
    ax.scatter(*mesh_props["h_m"][mesh_props["tri"]==1].T,zorder=1000,color="darkred",s=50)

    ax.scatter(*mesh_props["h_p"].T)

    # for center, radius in zip(x_hat,R_hat):
    #     ax.add_patch(Circle(center, radius, edgecolor='b', facecolor='none'))
    center, radius  = x_hat[1], R_hat[1]
    ax.add_patch(Circle(center, radius, edgecolor='red', facecolor='none'))
    for i in range(len(x)):
        ax.annotate(i, (x[i,0], x[i,1]))
    for i in range(len(mesh_props["tri"])):
        for j in range(3):
            start,end = mesh_props["v"][i,j],mesh_props["v_p1"][i,j]
            # start = np.mod(start,1)
            # end = start + periodic_displacement(end,start,1)
            # ax.scatter(*start,color="blue")
            ax.plot((start[0],end[0]),(start[1],end[1]),color="black")

    ax.imshow(np.flip(assignment.T,axis=0),zorder=-1,extent=[0,1,0,1],cmap="rainbow")
    ax.set(xlim=(-0.5,1.5), ylim=(-0.5,1.5))
    fig.show()



    empirical_areas = np.zeros((20))
    for i in range(20):
        empirical_areas[i] = (assignment==i).sum()
    plt.scatter(empirical_areas*(0.001**2),msh.mesh_props["A"])
    plt.show()


    msh._triangulate()
    mesh_props = get_geometry(msh.mesh_props)
    # mesh_props =  get_tintersections(mesh_props)
    hp = mesh_props["h_CCW_p"]
    hm = mesh_props["h_CW_p"]

    touch_not_power_mask = (~mesh_props["V_in_p1"]) * (~mesh_props["no_touch_p1"])


    hp_circ = hp[touch_not_power_mask]
    d = cdist(hp_circ, mesh_props["x"])**2 - mesh_props["R"] ** 2

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
    ax.set(xlim=(0,1),ylim=(0,1))

    for i in range(len(x_hat)-4):
        ax.annotate(dictionary[i], (x_hat[i,0], x_hat[i,1]),color="red")

    tx_hat = x_hat[tri_hat]
    for i in range(3):
        for j in range(len(tx_hat)):
            start,end = tx_hat[j,i],tx_hat[j,(i+1)%3]
            if ((start // 1) == 0).all():
                ax.plot((start[0],end[0]),(start[1],end[1]),color="black")
            else:
                ax.plot((start[0],end[0]),(start[1],end[1]),color="grey",alpha=0.2)

    ax.set(xlim=(-1,2),ylim=(-1,2),aspect=1)
    fig.show()




    #
    i = 14 #(array([12, 14, 16, 17, 18, 19, 24, 25, 26, 36, 46, 52, 53]),

    j = 2 # array([2, 2, 2, 1, 1, 0, 1, 2, 2, 2, 1, 0, 0]))

    fig, ax = plt.subplots()
    ax.scatter(*mesh_props["tx"][i,j])
    ax.scatter(*mesh_props["tx_m1"][i,j],color="grey")
    ax.scatter(*mesh_props["tx_p1"][i,j])
    ax.scatter(*mesh_props["v"][i,j])

    for _i, (center, radius) in enumerate(zip(mesh_props["tx"][i],mesh_props["tR"][i])):
        ax.add_patch(Circle(center, radius, edgecolor=plt.cm.plasma(_i/3), facecolor='none'))


    for _i, (center, radius) in enumerate(zip(mesh_props["x"],mesh_props["R"])):
        ax.add_patch(Circle(center, radius, edgecolor="grey", facecolor='none',alpha=0.2))

    for _i, (center, radius) in enumerate(zip(mesh_props["x"][np.where(mesh_props["theta"]<0)[0]],mesh_props["R"][np.where(mesh_props["theta"]<0)[0]])):
        ax.add_patch(Circle(center, radius, edgecolor="blue", facecolor='none',alpha=0.4))


    #
    # for _i, (center, radius) in enumerate(zip(mesh_props["tx"][i],mesh_props["tR"][i])):
    #     ax.add_patch(Circle(center+np.array([0,-1]), radius, edgecolor=plt.cm.plasma(_i/3), facecolor='none'))
    #
    # for _i, (center, radius) in enumerate(zip(mesh_props["tx"][i],mesh_props["tR"][i])):
    #     ax.add_patch(Circle(center+np.array([0,1]), radius, edgecolor=plt.cm.plasma(_i/3), facecolor='none'))

    # ax.scatter(*mesh_props["h_mid"][i,j],color="green")
    # ax.scatter(*tx_p1_registered[i,j],color="green")

    ax.scatter(*mesh_props["h_CCW"][i,j],color="blue")
    # ax.scatter(*mesh_props["h_p"][i,j],color="purple")
    ax.scatter(*mesh_props["h_m"][i,j].T,color="black",alpha=0.2,s=100)

    ax.set(aspect=1)
    fig.show()

    import numpy as np
    import matplotlib.pyplot as plt
    from descartes import PolygonPatch
    from shapely.geometry import Point, Polygon,GeometryCollection
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

    def plot_intersection(intersection,ax):
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

        intersections += [intersection_circle_polygon(mesh_props["x"][i],mesh_props["R"][i],mesh_props["v"][mesh_props["tri"]==i])]

    fig, ax = plt.subplots()
    for intersection in intersections:
        plot_intersection(intersection,ax)
    fig.show()

    """
    To do: 
    
    1. Fix the vertex allocation such that they are all facing in the correct direction. 
    2. Check whether this solves 
    3. If not, partition the areas by the sub-triangles from vertex to vertex to cell centre and repeat. 
    
    The errors are reference frame invariant
    
    """
