import numpy as np
import two_dimensional_cell as td
import time
from two_dimensional_cell.simulation import Simulation



radius= 1/np.sqrt(3)

P0=(radius*np.pi*1.65)
A0=(radius**2 * np.pi)*0.7

tissue_params = {"L": 9,
                 "radius":radius,
                 "A0": A0,
                 "P0": P0,
                 "kappa_A": 0.5,
                 "kappa_P": 0.5,
                 "kappa_M":0.1,
                 "W": np.array(((0, 0.00762), (0.00762, 0)))*10,
                 "a": 0,
                 "k": 0}
active_params = {"v0": 0.1,
                 "Dr": 1e-1}
init_params = {"init_noise": 0.1,
               "c_type_proportions": (.5,.5),
               "n_take":None}
run_options = {"equiangulate": True,
               "equi_nkill": 10}
simulation_params = {"dt": 0.02,
                     "tfin": 20,
                     "tskip": 1,
                     "grn_sim": None,
                     "random_seed":10}
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


sim.animate(n_frames=20,
                    color="#FFBA40",
                    file_name=None)

sim.animate_c_types(n_frames=20,
                    c_type_col_map=["#FFBA40", "#67F5B5"],
                    file_name=None)