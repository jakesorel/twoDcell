import two_dimensional_cell.tri_functions as trf
from two_dimensional_cell.mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt
L = 9.5
A0 = 1
init_noise = 1e-1
x = trf.hexagonal_lattice(int(L), int(np.ceil(L)), noise=init_noise, A=A0)
x += 1e-3

x = x[np.argsort(x.max(axis=1))[:int(L ** 2 / A0)]]

x += 0.25
np.argsort(x.max(axis=1))
radius = 0.6
buffer = 1.2

self = Mesh(x,radius,L,buffer)
self.update()

import matplotlib.pyplot as plt

def plot_circle(centre, radius,ax,color="black"):
    theta = np.linspace(0,np.pi*2,100)
    ax.plot(centre[0]+ radius*np.cos(theta),centre[1]+radius*np.sin(theta),color=color)
#
# buffer = radius *6
# on_boundary, on_edge, on_corner = get_edge_masks(x,L,buffer)
#
# y, M,idxs = extend_domain(x, L, radius, N, corner_shifts, mid_shifts)
#
# fig, ax = plt.subplots()
# radius = 0.3
# self = Mesh(x, radius, L, buffer)
# self.update()
# for i, xx in enumerate(self.x):
#     plot_circle(xx,radius,ax)
# for i, xx in enumerate(self.x[on_boundary]):
#     plot_circle(xx,radius,ax,color="blue")
# for oe in on_edge:
#     for i, xx in enumerate(self.x[oe]):
#         plot_circle(xx,radius,ax,color="orange")
# for oc in on_corner:
#     for i, xx in enumerate(self.x[oc]):
#         plot_circle(xx,radius,ax,color="purple")
# ax.set(aspect=1)
#
# """
# Shifts don't work...
# """
# fig.show()
#
fig, ax = plt.subplots()
radius = 0.5
self = Mesh(x, radius, L, radius * 2)
self.update()
for i, xx in enumerate(self.y[:self.n_c]):
    plot_circle(xx,radius,ax)
for i, xx in enumerate(self.y[self.n_c:]):
    plot_circle(xx, radius, ax,color="red")
ax.set(aspect=1)
# plot_circle(self.y[4], radius, ax,color="green")
fig.show()


rad_range = np.linspace(0.1,3,100)
As,Ps = np.zeros((len(rad_range),len(x))),np.zeros((len(rad_range),len(x)))
for i, radius in enumerate(rad_range):
    self = Mesh(x, radius, L, radius*2)
    self.update()
    print(self.y.shape)
    As[i] = self.A[:x.shape[0]]
    Ps[i] = self.P[:x.shape[0]]

plt.plot(As)
plt.show()

self.triangulate()
self.tri_format()
self.get_displacements()
self.classify_edges()
self.get_angles()

t0 = time.time()
for i in range(int(4000)):
    self.triangulate()
t1 = time.time()

print(t1 - t0)