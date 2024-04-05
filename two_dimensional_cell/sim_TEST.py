import numpy as np
import two_dimensional_cell as td
import time
from two_dimensional_cell.simulation import Simulation



# radius= 1/np.sqrt(3)


p0 = 3.81
A0 = 1.8
P0 = np.sqrt(A0)*p0
# P0=(radius*np.pi*1.65)
# A0=(radius**2 * np.pi)*0.4
radius_init = 0.5


tissue_params = {"L": 9,
                 "radius":radius_init,
                 "A0": A0,
                 "P0": P0,
                 "kappa_A": 0.15,
                 "kappa_P": 0.1,
                 "kappa_M":0.1,
                 "W": np.array(((0, 0.00762), (0.00762, 0)))*10,
                 "a": 0,
                 "k": 0}
active_params = {"v0": 0.2,
                 "Dr": 1e-1}
init_params = {"init_noise": 0.1,
               "c_type_proportions": (1.0,0.0),
               "n_take":None,
               "init_sf":0.8}
run_options = {"equiangulate": True,
               "equi_nkill": 10}#,
               #"x_init":np.array(((4.25,4.5),
                                  # (4.75,4.5),
                                  # (4.5,4.5+np.sqrt(3)/4)))}
simulation_params = {"dt": 0.02,
                     "tfin": 50,
                     "tskip": 1,
                     "grn_sim": None,
                     "random_seed":10,
                     "radius_damping_coefficient":0.1,
                     "centroid_damping_coefficient":0.1}
save_options = {"save": None,
                "result_dir": "results",
                "name": "ctype_example2",
                "compressed": True}
sim = Simulation(tissue_params=tissue_params,
                    active_params=active_params,
                    init_params=init_params,
                    simulation_params=simulation_params,
                    run_options=run_options,
                    save_options=save_options)

t0 = time.time()
sim.simulate(progress_bar=True)
t1= time.time()
print(t1-t0)


import matplotlib.pyplot as plt
plt.close("all")
plt.plot(sim.R_save)
plt.show()

F,G = sim.t.get_forces()
print(np.linalg.norm(F,axis=1))

"""
F looks fine, 

but both components of G are incorrect. 

For the "self" component, need to prevent the number of vertices factoring into the calc. Easy to resolve

For the "non-self" component (i.e. dtheta), may be easiest to derive again the expressions for dhp/dRi, these may be wrong. 

^^ Check again the code for this to ensure there aren't typos 


"""

sim.animate(n_frames=20,
                    color="#FFBA40",
                    file_name=None)

###There must be some typo in the code for the forces on the radius somewhere, given that in a 3 cell system, it's not symmetric

sim.animate_c_types(n_frames=20,
                    c_type_col_map=["#FFBA40", "#67F5B5"],
                    file_name=None)

#
#
# A = np.random.random((180,3,2))
# A_ = A.copy()
# B = np.random.random((180,3,3,2,2))
# B_ = B.copy()
#
#
# @jit(nopython=True)
# def get_C_(A_,B_):
#     C_ = np.zeros((180,3,3,2))
#     for i in range(180):
#         for j in range(3):
#             for k in range(3):
#                 C_[i,j,k] = A_[i,j]@B_[i,j,k]
#     return C_.sum(axis=2)
#
# # @jit(nopython=True)
# def get_C(A_,B):
#     A = np.expand_dims(A_, axis=2)
#     d = np.einsum("...l,...lm",A,B)
#     C = np.einsum("...kl->...l",d)
#     return C
#
#
# get_C_(A_,B_)
# get_C(A_,B)
# N = int(1e3)
# t0 = time.time()
# for i in range(N):
#     get_C_(A_, B_)
#
# t1= time.time()
# print(t1-t0)
#
# t0 = time.time()
# for i in range(int(1e4)):
#     get_C(A_, B)
#
# t1=time.time()
# print(t1-t0)
#
#
# C_ = C_.sum(axis=2)
