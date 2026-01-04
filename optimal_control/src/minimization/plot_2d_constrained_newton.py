import numpy as np
import matplotlib.pyplot as plt

# Load data
cost = np.loadtxt("build/cost.csv", delimiter=",")
constraint = np.loadtxt("build/constraint.csv", delimiter=",")
newton = np.loadtxt("build/newton.csv", delimiter=",")

# Cost grid
x = cost[:,0]
y = cost[:,1]
z = cost[:,2]

# Reshape for contour
N = int(np.sqrt(len(z)))
X = x.reshape(N, N)
Y = y.reshape(N, N)
Z = z.reshape(N, N)

plt.figure(figsize=(7,6))

# Cost contours
plt.contour(X, Y, Z, levels=30)
plt.colorbar(label="f(x)")

# Constraint
plt.plot(constraint[:,0], constraint[:,1], 'r', linewidth=2, label="c(x)=0")

# Newton iterates
plt.plot(newton[:,0], newton[:,1], 'ko-', label="Newton steps")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.axis("equal")
plt.title("Constrained Newton (KKT)")
plt.show()
