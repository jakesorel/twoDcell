import numpy as np
import os
import power_triangulation as pt
from cell_geometries import geometries
from tri_functions import hexagonal_lattice, square_boundary
from matplotlib import animation
from division_axis import get_minor_major_axes
import matplotlib.pyplot as plt
import time
from tri_functions import _roll, _roll3
from lineage import Lineage
from tri_functions import _CV_matrix
from division_axis import get_centroid,sort_coords
from numba import jit
from cell_geometries import _tV
from matplotlib.patches import Polygon
from shapely.geometry import Polygon, Point
from descartes import PolygonPatch

class Simulation:
    def __init__(self):

        ##box parameters
        self.xlim, self.ylim = [], []
        self.rb = []
        self.n_b = []

        ##tissue properties
        self.n_C = []
        self.n_c = []

        # tissue parameters
        self.lambda_A, self.lambda_P, self.lambda_M, self.P0, self.A0 = [], [], [], [], []

        # active mechanics
        self.noise = []
        self.Dr = []
        self.v0 = []

        ##initial conditions
        self.rc0 = []
        self.r0 = []
        self.R0 = []

        ##time properties
        self.dt = []
        self.t_span = []

        ##Saving matrices:
        self.r_save, self.R_save = [], []

        ##Division parameters
        self.dtA0 = 0.3
        self.A_crit = 1
        self.hash_dict = {}


        ##Plotting params
        self.cell_cols = plt.cm.plasma(np.random.random(300))

    def set_box(self, xlim=(-20, 20), ylim=(-20, 20)):
        self.xlim, self.ylim = xlim, ylim
        self.rb = square_boundary(xlim, ylim)
        self.n_b = self.rb.shape[0]

    def set_initial_positions(self, r0=None, nx=5, ny=5, noise=0.1):
        if r0 is None:
            self.rc0 = hexagonal_lattice(nx,ny, noise=noise)
            self.n_C = self.rc0.shape[0]
            self.r0 = np.vstack((self.rc0, self.rb))
            self.n_c = self.r0.shape[0]

    def set_initial_radii(self, R0=1):
        self.R0 = add_boundary_zeros(make_vector(R0, self.n_C), self.n_b)

    def set_mechanical_parameters(self, lambda_A, lambda_P, lambda_M, P0, A0):
        self.lambda_A = add_boundary_zeros(make_vector(lambda_A, self.n_C), self.n_b)
        self.lambda_P = add_boundary_zeros(make_vector(lambda_P, self.n_C), self.n_b)
        self.lambda_M = add_boundary_zeros(make_vector(lambda_M, self.n_C), self.n_b)
        self.P0 = add_boundary_zeros(make_vector(P0, self.n_C), self.n_b)
        self.A0 = add_boundary_zeros(make_vector(A0, self.n_C), self.n_b)

    def set_active_parameters(self,v0,Dr):
        self.v0 = add_boundary_zeros(make_vector(v0, self.n_C), self.n_b)
        self.Dr = add_boundary_zeros(make_vector(Dr, self.n_C), self.n_b)

    def set_soft_parameters(self,a,k):
        self.a,self.k = a,k

    def set_t_span(self, tfin, dt):
        self.tfin = tfin
        self.dt = dt
        self.t_span = np.arange(0, tfin, dt)
        self.n_t = self.t_span.shape[0]


    def generate_noise(self,reset=True):
        if reset is True:
            self.noise = []
        if type(self.noise) is list:
            theta_noise = np.cumsum(np.sqrt(2 * self.Dr.reshape(1,-1) * self.dt)*np.random.normal(0, 1, (self.n_t, self.n_c)), axis=0) + np.random.uniform(0,np.pi*2,self.n_c)
            self.noise = np.dstack((np.cos(theta_noise), np.sin(theta_noise)))
            self.noise[:,-self.n_b:] = 0

    def simulate(self):
        self.generate_noise()

        r, R = self.r0, self.R0

        r_save = np.zeros((self.n_t, self.n_C, 2))
        R_save = np.zeros((self.n_t, self.n_C))

        for i, t in enumerate(self.t_span):
            tri_list, V, n_v = pt.get_power_triangulation(r, R)
            geom = geometries(r, R, V, tri_list, n_v, n_b=self.n_b)
            geom.P0 = self.P0
            geom.A0 = self.A0
            geom.lambda_P = self.lambda_P
            geom.lambda_A = self.lambda_A
            geom.lambda_M = self.lambda_M
            geom.a,geom.k = self.a,self.k
            ##rather inelegant way of doing this. Consider overhauling geom code.

            geom.get_F()
            geom.get_F_soft()
            F = geom.FP + geom.F_soft
            r += F * self.dt
            r += self.v0.reshape(-1,1)*self.noise[i]*self.dt
            r_save[i] = r[:self.n_C]
            R_save[i] = R[:self.n_C]
        self.geom = geom

        self.r_save = r_save
        self.R_save = R_save


    def simulate_division(self,div_time=1,n_C_init = 4,dx_div = 0.2):
        self.generate_noise()

        t_span_div = self.t_span[int((div_time/self.dt)/2)::int((div_time/self.dt))]
        r, R = self.r0, self.R0
        r[n_C_init:-self.n_b] = np.nan
        R[n_C_init:-self.n_b] = np.nan

        r_save = np.ones((self.n_t, self.n_C, 2))*np.nan
        R_save = np.ones((self.n_t, self.n_C))*np.nan

        n_C_true = n_C_init

        self.lineage = Lineage(self.tfin)
        self.lineage.initialize_lineage(range(n_C_init))

        hash_matrix = np.zeros(len(self.lineage.live_cells),dtype=np.int64)
        for i, cll in enumerate(self.lineage.live_cells):
            hash_matrix[0] = cll.hash
        self.hash_dict[0] = hash_matrix

        for i, t in enumerate(self.t_span):
            r_red = np.vstack((r[:n_C_true], self.rb))

            R_red = add_boundary_zeros(R[:n_C_true], self.n_b)

            tri_list, V, n_v = pt.get_power_triangulation(r_red, R_red)
            geom = geometries(r_red, R_red, V, tri_list, n_v, n_b=self.n_b)
            geom.P0 = add_boundary_zeros(self.P0[:n_C_true], self.n_b)
            geom.A0 = add_boundary_zeros(self.A0[:n_C_true], self.n_b)
            geom.lambda_P = add_boundary_zeros(self.lambda_P[:n_C_true], self.n_b)
            geom.lambda_A = add_boundary_zeros(self.lambda_A[:n_C_true], self.n_b)
            geom.lambda_M = add_boundary_zeros(self.lambda_M[:n_C_true], self.n_b)
            geom.a,geom.k = self.a,self.k
            ##rather inelegant way of doing this. Consider overhauling geom code.

            geom.get_F()
            geom.get_F_soft()
            F = geom.FP + geom.F_soft
            r_red += F * self.dt
            r_red[:n_C_true] += self.v0[:n_C_true].reshape(-1,1)*self.noise[i,:n_C_true]*self.dt

            r[:n_C_true] = r_red[:n_C_true]
            R[:n_C_true] = R_red[:n_C_true]
            r_save[i,:n_C_true] = r[:n_C_true]
            R_save[i,:n_C_true] = R[:n_C_true]

            self.do_A0_step(n_C_true)
            A_div_crit = geom.A[:n_C_true]>self.A_crit
            if (A_div_crit).any():
                for k in np.nonzero(A_div_crit)[0]:
                    major_axis,__ = get_minor_major_axes(k, r, R, geom.hp_j, geom.hm_j, geom.CV_matrix, geom.no_touch_j, alpha_small=0.05)
                    if np.isnan(major_axis[0]):
                        theta = np.random.uniform(0,np.pi*2)
                        major_axis = np.array((np.cos(theta),np.sin(theta)))
                    r[k] = r[k] - dx_div*major_axis/2
                    r[n_C_true] = r[k] + dx_div*major_axis/2
                    R[n_C_true] = R[k]
                    self.A0[n_C_true] = self.A0[k]/2
                    self.A0[k] = self.A0[k]/2
                    self.lineage.track_division(time=t,new_sim_cell_id=n_C_true,parent_cell_id=k)



                    n_C_true += 1

                hash_matrix = np.zeros(len(self.lineage.live_cells))
                for i, cll in enumerate(self.lineage.live_cells):
                    hash_matrix[i] = cll.hash
                self.hash_dict[i] = hash_matrix

        self.geom = geom
        self.r_save = r_save
        self.R_save = R_save


    def do_A0_step(self,n_C_true):
        """
        Basic linear integration of A0

        Follows dA0/dt = "self.dtA0"

        Called once per simulation step.

        :param n_C_true:
        :return:
        """
        self.A0[:n_C_true] += self.dtA0*self.dt

    def cell_cells_at_time(self,ti):
        hash_keys = np.array(list(self.hash_dict.keys()))
        tj = hash_keys[np.nonzero(hash_keys<=ti)[0][-1]]
        hash_mat = self.hash_dict[tj]
        clls = [self.lineage.get_cell_from_hash(hash) for hash in hash_mat]
        return clls

    def animate(self,n_frames=100, file_name=None, dir_name="animations",xlim=None,ylim=None):
        make_directory(dir_name)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        skip = int(self.n_t/ n_frames)

        min_corner = np.amin(self.r_save, axis=(0,1)) - np.max(self.R_save) * 3
        max_corner = np.amax(self.r_save, axis=(0,1)) + np.max(self.R_save) * 3
        if xlim is None:
            xlim = (min_corner[0], max_corner[0])
        if ylim is None:
            ylim = (min_corner[1], max_corner[1])

        def animate(i):
            ax1.cla()

            r = np.vstack((self.r_save[i*skip], self.rb))

            R = add_boundary_zeros(self.R_save[i*skip],self.n_b)

            tri_list, V, n_v = pt.get_power_triangulation(r, R)
            voronoi_cell_map = pt.get_voronoi_cells(r, V, tri_list)
            pt.display(ax1, r, R, tri_list, voronoi_cell_map, tri_alpha=0.0, n_b=self.n_b, xlim=xlim, ylim=ylim,
                       line_col="white")
            ax1.set(aspect=1, xlim=xlim, ylim=ylim)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=1800)
        if file_name is None:
            file_name = "animation %d" % time.time()
        an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
        an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)


    def animate_division(self,n_frames=100, file_name=None, dir_name="animations",xlim=None,ylim=None):
        make_directory(dir_name)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        skip = int(self.n_t/ n_frames)

        min_corner = np.nanmin(self.r_save, axis=(0,1)) - np.nanmax(self.R_save) * 3
        max_corner = np.nanmax(self.r_save, axis=(0,1)) + np.nanmax(self.R_save) * 3
        if xlim is None:
            xlim = (min_corner[0], max_corner[0])
        if ylim is None:
            ylim = (min_corner[1], max_corner[1])

        def animate(i):
            ax1.cla()
            self.plot_cells(ax1,i*skip)
            #
            # r = np.vstack((self.r_save[i*skip], self.rb))
            #
            # R = add_boundary_zeros(self.R_save[i*skip],self.n_b)
            #
            # r = r[~np.isnan(R)]
            # R = R[~np.isnan(R)]
            #
            # tri_list, V, n_v = pt.get_power_triangulation(r, R)
            # voronoi_cell_map = pt.get_voronoi_cells(r, V, tri_list)
            # pt.display(ax1, r, R, tri_list, voronoi_cell_map, tri_alpha=0.0, n_b=self.n_b, xlim=xlim, ylim=ylim,
            #            line_col="white")
            ax1.set(aspect=1, xlim=xlim, ylim=ylim)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=1800)
        if file_name is None:
            file_name = "animation %d" % time.time()
        an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
        an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)


    def plot_cells(self,ax1,ti):
        """
        Plot every cell type.
        Parameters
        ----------
        ax1 : matplotlib axis
            Parse in mpl axis argument
        i : int
            Frame number of **x_save** (i.e. for now, of **t_span**)
        """
        ax1.clear()
        ax1.axis('off')
        r = np.vstack((self.r_save[ti], self.rb))

        R = add_boundary_zeros(self.R_save[ti], self.n_b)

        r = r[~np.isnan(R)]
        R = R[~np.isnan(R)]
        clls = self.cell_cells_at_time(ti)
        n_C_true = np.sum(~np.isnan(r[:,0]))
        tri_list, V, n_v = pt.get_power_triangulation(r, R)
        tV = _tV(V)
        CV_matrix = _CV_matrix(tri_list, n_v, n_C_true)
        pt.get_voronoi_cells(r,V,tri_list)
        for cid in range(n_C_true - self.n_b):
            vs = get_power_polygon(cid, CV_matrix, tV)
            poly = Polygon(vs)
            circle = Point(r[cid]).buffer(R[cid])
            cell_poly = circle.intersection(poly)
            if cell_poly.area !=0:
                ax1.add_patch(PolygonPatch(cell_poly, ec="white", fc=clls[cid].color))
            # ax1.add_patch(Polygon(polygon, fill=False, edgecolor="white"))
        # ax1.set(xlim=[self.r_save[:,:,0].min() - self.R_save.max()*2,self.x_save[:,:,0].max()+ self.R.max()*2],ylim=[self.x_save[:,:,1].min() - self.R.max()*2,self.x_save[:,:,1].max()+ self.R.max()*2],aspect=1)



def make_vector(val, n):
    if type(val) is np.ndarray:
        out = val
    else:
        if type(val) is int:
            val = np.float(val)
        out = np.ones(n) * val
    return out


def add_boundary_zeros(val, n_b):
    return np.concatenate((val, np.zeros(n_b)))


def make_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# @jit(nopython=True)
def get_power_polygon(cll_i,CV_matrix,tV):
    cll_mask = CV_matrix[cll_i] == 1
    cll_mask_flat = cll_mask.ravel()
    vs = np.column_stack((tV[:, :, 0].ravel()[cll_mask_flat], tV[:, :, 1].ravel()[cll_mask_flat]))
    centroid = np.mean(vs,axis=0)
    vs = sort_coords(vs,centroid)
    return vs

