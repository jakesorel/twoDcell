import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import grad as jgrad
from jax import grad, vmap

from jax import jit
import jax
import jax

from scipy import optimize

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)
import time
import numba as nb


@jit
def f(x, theta=1):
    return jnp.sum((x - theta) ** 2)


f_grad = jit(jgrad(f))




@nb.jit(nopython=True)
def circumcenter_nb(C):
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    ri, rj, rk = C.transpose(1, 2, 0)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size, 2), dtype=np.float64)
    vs[:, 0], vs[:, 1] = ux, uy
    return vs

@jit
def circumcenter(C: np.ndarray) -> np.ndarray:
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    # ri, rj, rk = C.transpose(1, 2, 0)
    (ax, ay),(bx, by),(cx, cy) = C[:,0].T,C[:,1].T,C[:,2].T
    # ax, ay = ri.T
    # bx, by = rj
    # cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    return jnp.column_stack((ux, uy))


@jit
def circumcenter_i(Ci: np.ndarray) -> np.ndarray:
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    # ri, rj, rk = C.transpose(1, 2, 0)
    (ax, ay),(bx, by),(cx, cy) = Ci
    # ax, ay = ri.T
    # bx, by = rj
    # cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    return jnp.array((ux,uy)),d

@nb.jit(nopython=True)
def dhdr(rijk):
    """
    Calculates ∂h_j/dr_i the Jacobian for all cells in each triangulation
    Last two dims: ((dhx/drx,dhx/dry),(dhy/drx,dhy/dry))
    These are lifted from Mathematica
    :param rijk_: (n_v x 3 x 2) np.float32 array of cell centroid positions for each cell in each triangulation (first two dims follow order of triangulation)
    :param vs: (n_v x 2) np.float32 array of vertex positions, corresponding to each triangle in the triangulation
    :param L: Domain size (np.float32)
    :return: Jacobian for each cell of each triangulation (n_v x 3 x 2 x 2) np.float32 array (where the first 2 dims follow the order of the triangulation.
    """
    DHDR = np.empty(rijk.shape + (2,))
    for i in range(3):
        ax,ay = rijk[:,np.mod(i,3),0],rijk[:,np.mod(i,3),1]
        bx, by = rijk[:, np.mod(i+1,3), 0], rijk[:, np.mod(i+1,3), 1]
        cx, cy = rijk[:, np.mod(i+2,3), 0], rijk[:, np.mod(i+2,3), 1]
        #dhx/drx
        DHDR[:, i, 0, 0] = (ax * (by - cy)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((by - cy) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/drx
        DHDR[:, i, 1,0] = (bx ** 2 + by ** 2 - cx ** 2 + 2 * ax * (-bx + cx) - cy ** 2) / (
                    2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((by - cy) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhx/dry
        DHDR[:, i, 0, 1] = (-bx ** 2 - by ** 2 + cx ** 2 + 2 * ay * (by - cy) + cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((-bx + cx) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/dry
        DHDR[:, i, 1,1] = (ay * (-bx + cx)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((-bx + cx) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)


    return DHDR



circumcenter_grad = vmap(jgrad(circumcenter))
C = np.random.uniform(0,10,(180,3,2))

circumcenter_nb(C)
circumcenter(C)
t0 = time.time()

for i in range(int(1e5)):
    circumcenter(C)
t1 = time.time()
print(t1-t0)

@jit
def dhdr_jax(C):
    return jax.vmap(jax.jacrev(circumcenter_i,has_aux=True))(C).transpose(0,2,1,3)


t0 = time.time()
for i in range(int(1e4)):
    dhdr_jax(C)
t1 = time.time()
print(t1-t0)


