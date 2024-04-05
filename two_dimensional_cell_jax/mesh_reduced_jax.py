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
import time
import numba as nb
@jit
def tnorm(x):
    """
    Calculate the L1 norm of a set of vectors that are given in triangulated form:

    (nv x 3 x 2) ->> (nv x 3)
    :param x:
    :return:
    """
    return jnp.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)

@jit
def get_lP(start, end):
    return tnorm(end - start)

@jit
def get_lC(tpsi, tR):
    return tpsi * tR

@partial(jax.jit, static_argnums=2)
def assemble_tri(tval, tri,n_C):
    """
    Sum all components of a given cell property.
    I.e. (nv x 3) --> (nc x 1)
    :param tval:
    :param tri:
    :return:
    """

    out = jnp.zeros((n_C + 1))
    return out.at[tri].add(tval)

@partial(jax.jit, static_argnums=(5,6))
def get_P(hp_j,hm_j,tpsi_j,tR,tri,n_c,n_C):
    tlP, tlC = get_lP(hp_j, hm_j), get_lC(tpsi_j, tR)
    # self.tP = self.tlP + self.tlC
    _LP = assemble_tri(tlP, tri,n_C)
    _LC = assemble_tri(tlC, tri,n_C)
    LP = _LP[:n_c]
    LC = _LC[:n_c]
    # self._P = self._LP + self._LC
    P = LP + LC
    return P

hp_j,hm_j,tpsi_j,tR,tri, n_c,n_C = sim.t.mesh.hp_j,sim.t.mesh.hm_j,sim.t.mesh.tpsi_j,sim.t.mesh.tR,sim.t.mesh.tri, sim.t.mesh.n_c,sim.t.mesh.n_C

get_P(hp_j,hm_j,tpsi_j,tR,tri,n_c,n_C)

###########

x,R = sim.t.mesh.x,sim.t.mesh.R
tx = sim.t.mesh.tx
tR = sim.t.mesh.tR

def get_

def get_tP()

@jit
def get_tE_i(tx,tR,tP0):
    get_tP(tx,tR)
    tEi = (tP-tP0)**2
    return tEi






tx, trf.roll3(tx, direc)
tR, trf.roll(tR,direc)
lrij

def _tintersections(ri, rj, Ri, Rj, nrij):
    # ri,rj,Ri,Rj,nrij = geom.tS,_roll3(geom.tS,1),geom.tR,_roll(geom.tR),geom.nrij
    ri, rj, Ri, Rj, nrij = ri.T, rj.T, Ri.T, Rj.T, nrij.T
    a = 0.5 * (ri + rj) + (Ri ** 2 - Rj ** 2) / (2 * nrij ** 2) * (rj - ri)
    b = 0.5 * np.sqrt(
        2 * (Ri ** 2 + Rj ** 2) / nrij ** 2 - ((Ri ** 2 - Rj ** 2) ** 2) / nrij ** 4 - 1) * np.stack(
        (rj[1] - ri[1], ri[0] - rj[0]))
    a, b = a.T, b.T
    pos1 = a - b
    pos2 = a + b
    return pos1, pos2


def _ttheta(V_in, V_out, no_touch, tR, tS, tV, nrij, vj_neighbours, dir=1):
    """
    V_in, V_out, no_touch, tR, tS, tV, nrij, vj_neighbours = self.V_in_j, self.V_out_j, self.no_touch_j, self.radius,self.tx, self.vs3,self.lrij, self.vm1
    """
    V_in2, V_out2, no_touch2 = np.dstack((V_in, V_in)), np.dstack((V_out, V_out)), np.dstack((no_touch, no_touch))
    start, end = vj_neighbours.copy(), tV.copy()
    h_CCW, h_CW = _tintersections(tS, trf.roll3(tS, dir), tR, trf.roll(tR, dir), nrij)

    end = trf.replace_vec(end, ~V_in2, h_CCW)
    start = trf.replace_vec(start, ~V_out2, h_CW)
    end = trf.replace_val(end, no_touch2, 0)
    start = trf.replace_val(start, no_touch2, 0)

    ttheta = _tvecangle(start - tS, end - tS)
    return ttheta, start, end

@jit(nopython=True)
def _tvecangle(a, b):
    """
    Signed angle between two (triangle form) sets of vectors
    :param a:
    :param b:
    :return:
    """
    return np.arctan2(trf.tcross(a, b), trf.tdot(a, b))



self.ttheta_j, self.hm_j, self.hp_j = _ttheta(self.V_in_j, self.V_out_j, self.no_touch_j, self.tR,
                                              self.tx,
                                              self.vs3,
                                              self.lrij, self.vm1, dir=1)

self.ttheta_k, self.hm_k, self.hp_k = _ttheta(self.V_in_k, self.V_out_k, self.no_touch_k, self.tR,
                                              self.tx,
                                              self.vp1,
                                              self.lrik, self.vs3, dir=-1)


