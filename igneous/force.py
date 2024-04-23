import numpy as np
import jax.numpy as jnp
from jax import jit, vmap,jacrev
from functools import partial
import jax
from scipy import sparse
from igneous.mesh import get_geometry
import triangle as tr
from scipy.spatial import ConvexHull
from igneous.power_triangulation import triangulate
from matplotlib.patches import Polygon
from shapely.geometry import Polygon, Point ##ensure it is v.1.7.1
from descartes import PolygonPatch
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_debug_nans", True)
import pandas as pd
import matplotlib.pyplot as plt

class Force:
    """
    Force class
    -----------

    This class is used to calculate the passive mechanics under the SPV model.

    Takes in an instance of the Tissue class, and within it, the Mesh class, and uses information about the geometry to calculate the forces on the cell centroids.

    These forces are accessible in self.F.
    """

    def __init__(self, tissue):
        ##say that mesh is a property of the tissue object.
        self.t = tissue

        self.F = None
        self.get_mechanics()


    def get_mechanics(self):
        FF = get_force(jnp.column_stack([self.t.mesh.mesh_props["x"],self.t.mesh.mesh_props["R"]]),self.t.mesh.mesh_props,int(self.t.mesh.mesh_props["n_c"]),self.t.tissue_params)
        F,G = FF[:,:2], FF[:,2]
        return F,G

@partial(jit, static_argnums=(2,))
def get_E(Y, mesh_props, n_c,tissue_params):
    x, R = Y[:, :2], Y[:, 2]
    mesh_props["x"] = x
    mesh_props["R"] = R
    mesh_props["n_c"] = n_c

    mesh_props = get_geometry(mesh_props, n_c)
    E = tissue_params["kappa_A"]*(mesh_props["A"] - tissue_params["A0"]) ** 2 + tissue_params["kappa_P"]*(mesh_props["P"] - tissue_params["P0"]) ** 2
    return E.sum()

@partial(jit, static_argnums=(2,))
def get_force(Y, mesh_props, n_c):
    return - jacrev(get_E)(Y, mesh_props, n_c)
