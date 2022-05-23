import numpy as np
import twoDcell as td
import time



radius= 0.6

P0=(radius*np.pi*1.65)
A0=(radius**2 * np.pi)*0.8

tissue_params = {"L": 9,
                 "radius":radius,
                 "A0": A0,
                 "P0": P0,
                 "kappa_A": 0.5,
                 "kappa_P": 0.5,
                 "kappa_M":0.1,
                 "W": np.array(((0, 0.00762), (0.00762, 0))),
                 "a": 0,
                 "k": 0}
active_params = {"v0": 4e-3,
                 "Dr": 1e-1}
init_params = {"init_noise": 0.00005,
               "c_type_proportions": (1.0,0.0)}
run_options = {"equiangulate": True,
               "equi_nkill": 10}
simulation_params = {"dt": 0.02,
                     "tfin": 20,
                     "tskip": 1,
                     "grn_sim": None,
                     "random_seed":10}
save_options = {"save": "skeleton",
                "result_dir": "results",
                "name": "ctype_example2",
                "compressed": True}
sim = td.simulation(tissue_params=tissue_params,
                    active_params=active_params,
                    init_params=init_params,
                    simulation_params=simulation_params,
                    run_options=run_options,
                    save_options=save_options)

t0 = time.time()
sim.simulate(progress_bar=True)
t1= time.time()
print(t1-t0)


sim.animate_c_types(n_frames=20,
                    c_type_col_map=["#FFBA40", "#67F5B5"],
                    file_name=None)
import matplotlib.pyplot as plt
import twoDcell.sim_plotting as plot
import twoDcell.tri_functions as trf
import twoDcell.periodic_functions as per
fig, ax = plt.subplots()
plot.plot_vor(ax,sim.t.mesh.x,sim.t.mesh.L,sim.t.mesh.radius)
ax.quiver(sim.t.mesh.x[:,0],sim.t.mesh.x[:,1],sim.t.F[:,0],sim.t.F[:,1])

t = sim.t
tx_nan = np.zeros((t.mesh.tx.shape[0],9,2))
tx_nan[:,::3] = t.mesh.tx
tx_nan[:,1::3] = t.mesh.tx + per.per3(trf.roll3(t.mesh.tx,1)-t.mesh.tx,t.mesh.L,t.mesh.L)

tx_nan[:,2::3] = np.nan

ax.plot(tx_nan[:,:,0].ravel(),tx_nan[:,:,1].ravel())
ax.scatter(t.mesh.hp_k[:,:,0],t.mesh.hp_k[:,:,1])

fig.show()

