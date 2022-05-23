from simulation import Simulation
import numpy as np
sim = Simulation()
sim.set_box()
sim.set_initial_positions(nx=20,ny=20,noise = 0.5)
sim.set_initial_radii(0.6)
sim.set_mechanical_parameters(lambda_A=1,
                   lambda_P=1,
                   lambda_M=0.1,
                   P0=(sim.R0[0]*np.pi*1.65),
                   A0=(sim.R0[0]**2 * np.pi)*0.8)
sim.noise = []
sim.set_active_parameters(v0=0.05,
                          Dr=10)
sim.set_soft_parameters(a=0.2,
                        k=0)
sim.set_t_span(tfin=20,dt=0.02)
# sim.simulate()
# sim.animate(n_frames=30)
sim.dtA0 = 0.3
sim.A_crit = 1
sim.simulate_division(n_C_init=1,div_time=4,dx_div=0.3)
print("Simulation complete")
sim.animate_division(n_frames=30)
print("Animation complete")
# sim.lineage.plot_phylogeny()

lin = sim.lineage
lin.assemble_phylogeny()
lin.color_lineage()
sim.animate_division(n_frames=40)

"""
Not working. Could either be the bfs search is wrong (e.g. going the wrong direction)

Or the calling of cell ids is wrong. 
"""


g = lin.phylogeny

tfin = sim.tfin

for cll in lin.all_cells:
    if cll.division_time is None:
        cll.division_time = tfin

def get_branch_length(cell):
    return cell.division_time - cell.birth_time

for cll in lin.all_cells:
    cll.branch_length = get_branch_length(cll)

import networkx as nx
import matplotlib.pyplot as plt

tfin = sim.tfin
g = nx.DiGraph()
lin.all_cells[0].birth_time = -1
for cll in lin.all_cells:
    cllid = "%d:%.3f"%(cll.hash,cll.branch_length)
    # cllid = "%d.%d:%.3f"%(cll.sim_cell_id,cll.n_division,cll.branch_length)
    parent = cll.parent
    if parent is not None:
        parentid = "%d:%.3f"%(parent.hash,parent.branch_length)
        # parentid = "%d.%d:%.3f"%(parent.sim_cell_id,parent.n_division,parent.branch_length)
        birthtime = cll.birth_time
        divisiontime = cll.division_time
        if divisiontime is None:
            divisiontime = tfin
        branch_length = divisiontime - birthtime
        # print(birthtime)
        print(branch_length)
        g.add_edge(parentid,cllid)



def recursive_search(dict, key):
    if key in dict:
        return dict[key]
    for k, v in dict.items():
        item = recursive_search(v, key)
        if item is not None:
            return item

def bfs_edge_lst(graph, n):
    return list(nx.bfs_edges(graph, n))

def tree_from_edge_lst(elst,founderid):
    tree = {founderid: {}}
    for src, dst in elst:
        subt = recursive_search(tree, src)
        subt[dst] = {}
    return tree

def tree_to_newick(tree):
    items = []
    for k in tree.keys():
        s = ''
        if len(tree[k].keys()) > 0:
            subt = tree_to_newick(tree[k])
            if subt != '':
                s += '(' + subt + ')'
        s += k
        items.append(s)
    return ','.join(items)


founder = lin.all_cells[0]
founderid = "%d:%.3f"%(founder.hash,founder.branch_length)


elst = bfs_edge_lst(g, founderid)
tree = tree_from_edge_lst(elst,founderid)
newick = tree_to_newick(tree) + ';'


t = Tree(newick)
ts = TreeStyle()
ts.show_leaf_name = False
ts.mode = "c" # draw tree in circular mode
ts.scale = 20
# ts.arc_start = -180 # 0 degrees = 3 o'clock
# ts.arc_span = 180
t.render("trees/mytree_large.png", w=183, units="mm", tree_style=ts)

