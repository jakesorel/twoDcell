"""
Much of this code has been adapted from a GitHub repo. I'll find the source.

"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from jax import numpy as jnp
from functools import partial
import jax
from scipy import sparse
import triangle as tr
from scipy.spatial import ConvexHull


@jit
def is_ccw_triangle(A, B, C):
    m = jnp.stack((A,B,C))
    M = jnp.column_stack((m,jnp.ones(3)))
    return jnp.linalg.det(M) > 0




def build_tri_and_norm(S, simplices, equations):
    """
    Build triangulation, ensuring triangles are CCW orientated.

    This has scope to be optimised.
    """
    saved_tris = equations[:, 2] <= 0
    n_v = saved_tris.sum()
    norms = equations[saved_tris]
    tri_list = jnp.zeros((n_v, 3), dtype=int)
    i = 0
    for (a, b, c), eq in zip(simplices[saved_tris], norms):
        if is_ccw_triangle(S[a], S[b], S[c]):
            tri_list = tri_list.at[i].set(jnp.array((a, b, c)))
        else:
            tri_list = tri_list.at[i].set(jnp.array((a, c, b)))
        i += 1
    return tri_list, norms, n_v

@jit
def get_x_lifted(x, R):
    """
    'Lift' x into the upper parabola
    """
    x_norm = jnp.sum(x**2,axis=1)**2 - R ** 2
    x_lifted = jnp.column_stack((x, x_norm))
    return x_lifted


def triangulate(x, R):
    # Compute the lifted weighted points
    x_lifted = get_x_lifted(x, R)
    # Compute the convex hull of the lifted weighted points
    hull = ConvexHull(x_lifted)
    #
    # # Extract the Delaunay triangulation from the lower hull
    tri_list, norms, n_v = build_tri_and_norm(x, hull.simplices, hull.equations)

    return tri_list




