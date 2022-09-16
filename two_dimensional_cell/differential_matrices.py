import numpy as np
from tri_functions import _tcross, _tdot,_roll,_roll3,_touter,_tidentity,_tnorm,_tmatmul,_replace_val
from numba import jit
from differentials import _dtheta_dh,_power_vertex_differentials,_compile_chain2,_circle_vertex_differentials,_compile_alt_thetas

dhv_dri,dhv_dRi = _power_vertex_differentials(tS,tR, rij,riV)