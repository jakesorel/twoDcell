import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import grad as jgrad
from jax import grad, vmap

from jax import jit
import jax
import jax
from functools import partial
from scipy import optimize

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)


@partial(jax.jit, static_argnums=1)
def roll(x, direc=1):
    """
    Jitted equivalent to np.roll(x,-direc,axis=1)
    direc = 1 --> counter-clockwise
    direc = -1 --> clockwise
    :param x:
    :return:
    """
    if direc == -1:  # old "roll_forward"
        return jnp.column_stack((x[:, 2], x[:, :2]))
    elif direc == 1:  # old "roll_reverse"
        return jnp.column_stack((x[:, 1:3], x[:, 0]))


@jit(nopython=True)
def roll3(x, direc=1):
    """
    Like roll, but when x has shape (nv x 3 x 2) ie is a vector, rather than scalar, quantity.
    :param x:
    :param direc:
    :return:
    """
    x_out = np.empty_like(x)
    x_out[:, :, 0], x_out[:, :, 1] = roll(x[:, :, 0], direc=direc), roll(x[:, :, 1], direc=direc)
    return x_out

