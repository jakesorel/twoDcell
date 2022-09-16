import networkx as nx
from ete3 import Tree,TreeStyle

class Cell:
    def __init__(self,sim_cell_id,birth_time):
        self.hash = self.__hash__() ##unique identifier
        self.sim_cell_id = sim_cell_id ##row number in the simulation.
        self.birth_time = birth_time
        self.division_time = None
        self.parent = None
        self.n_division = 0
        self.color = "grey"

    def assign_parent(self,parent_cell):
        self.parent = parent_cell

    @property
    def branch_length(self):
        return self.division_time - self.birth_time


class Lineage:
    def __init__(self,tfin):
        self.all_cells = []
        self.live_cells = []
        self.tfin = tfin
        self.phylogeny = None


    @property
    def sim_cell_ids(self):
        return [cll.sim_cell_id for cll in self.live_cells]

    def get_live_cell(self,sim_cell_id):
        return self.live_cells[self.sim_cell_ids.index(sim_cell_id)]

    @property
    def cell_hashes(self):
        return [cll.hash for cll in self.all_cells]

    def get_cell_from_hash(self,hash):
        return self.all_cells[self.cell_hashes.index(hash)]

    def initialize_lineage(self,sim_cell_ids,t0=0):
        for i in sim_cell_ids:
            new_cell = Cell(sim_cell_id=i,birth_time=t0)
            self.all_cells.append(new_cell)
            self.live_cells.append(new_cell)

    def track_division(self,time,new_sim_cell_id,parent=None,parent_cell_id=None):
        assert (parent_cell_id is not None)or(parent is not None), \
            "Must give either a Cell object 'parent' or a parent_cell_id (int)"

        if parent is None: ##can over-ride the parent_cell_id
            parent = self.get_live_cell(parent_cell_id)
        parent.division_time = time
        daughter1 = Cell(sim_cell_id=parent.sim_cell_id,birth_time=time)
        daughter2 = Cell(sim_cell_id=new_sim_cell_id,birth_time=time)
        daughter1.parent = parent
        daughter2.parent = parent
        parent.daughter1 = daughter1
        parent.daughter2 = daughter2
        daughter1.n_division =parent.n_division + 1
        daughter2.n_division =parent.n_division + 1
        daughter1.sister = daughter2
        daughter2.sister = daughter1
        self.all_cells.extend([daughter1,daughter2])
        self.live_cells.remove(parent)
        self.live_cells.extend([daughter1,daughter2])

    def assemble_phylogeny(self):
        for cll in self.all_cells:
            if cll.division_time is None:
                cll.division_time = self.tfin
        g = nx.DiGraph()
        self.all_cells[0].birth_time = -1
        for cll in self.all_cells:
            cllid = "%d:%.3f" % (cll.hash, cll.branch_length)
            parent = cll.parent
            if parent is not None:
                parentid = "%d:%.3f" % (parent.hash, parent.branch_length)
                g.add_edge(parentid, cllid)
        self.phylogeny = g

    def plot_phylogeny(self,filename="cell_tree"):
        if self.phylogeny is None:
            self.assemble_phylogeny()
        founder = self.all_cells[0]
        founderid = "%d:%.3f" % (founder.hash, founder.branch_length)

        elst = bfs_edge_lst(self.phylogeny, founderid)
        tree = tree_from_edge_lst(elst, founderid)
        newick = tree_to_newick(tree) + ';'

        t = Tree(newick)
        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.mode = "c"  # draw tree in circular mode
        ts.scale = 20
        # ts.arc_start = -180 # 0 degrees = 3 o'clock
        # ts.arc_span = 180
        t.render("trees/%s.png"%filename, w=183, units="mm", tree_style=ts)

    def color_lineage(self,cll_start=None,trace_color = "red"):
        if self.phylogeny is None:
            self.assemble_phylogeny()
        if cll_start is None:
            cll_start = self.all_cells[5]
        cll_start_id = "%d:%.3f" % (cll_start.hash, cll_start.branch_length)
        edges = list(nx.bfs_edges(self.phylogeny, cll_start_id))
        hashes = [int(start.split(":")[0]) for (start, end) in edges]
        hashes.extend([int(end.split(":")[0]) for (start, end) in edges])
        hashes = list(set(hashes))
        for hash in hashes:
            cll = self.get_cell_from_hash(hash)
            cll.color = trace_color

def recursive_search(dict, key):
    if key in dict:
        return dict[key]
    for k, v in dict.items():
        item = recursive_search(v, key)
        if item is not None:
            return item

def bfs_edge_lst(graph, n):
    return list(nx.bfs_edges(graph, n))

def tree_from_edge_lst(elst, founderid):
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