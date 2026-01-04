import numpy as np
import matplotlib.pyplot as plt

surf = np.loadtxt("build/surface.csv", delimiter=",")
path = np.loadtxt("build/newton.csv", delimiter=",")

x = surf[:,0]
y = surf[:,1]
z = surf[:,2]

plt.figure(figsize=(7,6))
plt.tricontour(x, y, z, levels=30)
plt.plot(path[:,0], path[:,1], 'ro-', label="Newton path")
plt.scatter(path[0,0], path[0,1], c='green', label="start")
plt.scatter(path[-1,0], path[-1,1], c='black', label="end")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Backtracking Regularized Newton (2D)")
plt.grid()
plt.show()
