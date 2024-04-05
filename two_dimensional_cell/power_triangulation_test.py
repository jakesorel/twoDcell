import two_dimensional_cell.power_triangulation as pt
import numpy as np
import time
import triangle as tr

x = np.random.uniform(0,17,(100,2))
R = np.ones((100))
pt.get_power_triangulation(x, R)

t0 = time.time()
for i in range(int(1e3)):
    pt.get_power_triangulation(x, R)

t1 =time.time()
print(t1-t0)


t0 = time.time()
for i in range(int(1e3)):
    triangulation = tr.triangulate({"vertices": x}, "n")
t1 =time.time()
print(t1-t0)
