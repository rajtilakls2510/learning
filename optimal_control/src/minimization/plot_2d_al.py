import numpy as np
import matplotlib.pyplot as plt

# Load data
cost = np.loadtxt("build/cost.csv", delimiter=",")

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

# Equality constraints
eq = np.loadtxt("build/equalities.csv", delimiter=",")
plt.contour(
    eq[:,0].reshape(N,N),
    eq[:,1].reshape(N,N),
    eq[:,2].reshape(N,N),
    levels=[0],
    colors='red',
    linewidths=2
)
plt.contour(
    eq[:,0].reshape(N,N),
    eq[:,1].reshape(N,N),
    eq[:,3].reshape(N,N),
    levels=[0],
    colors='orange',
    linewidths=2
)

# Inequality boundaries
ineq = np.loadtxt("build/inequalities.csv", delimiter=",")
plt.contour(
    ineq[:,0].reshape(N,N),
    ineq[:,1].reshape(N,N),
    ineq[:,2].reshape(N,N),
    levels=[0],
    colors='green',
    linestyles='dashed'
)
plt.contour(
    ineq[:,0].reshape(N,N),
    ineq[:,1].reshape(N,N),
    ineq[:,3].reshape(N,N),
    levels=[0],
    colors='blue',
    linestyles='dashed'
)
try:
    al_steps = np.loadtxt("build/al_steps.csv", delimiter=",")

    # AL outer iterations
    plt.plot(al_steps[:,0], al_steps[:,1], 'ko-', linewidth=2, label="AL steps")
except:
    pass
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend([
    "h₁(x)=0",
    "h₂(x)=0",
    "g₁(x)=0",
    "g₂(x)=0",
    "AL iterates"
])
plt.grid()
plt.axis("equal")
plt.title("Augmented Lagrangian Optimization Path")
plt.show()
