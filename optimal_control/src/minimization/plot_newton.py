import numpy as np
import matplotlib.pyplot as plt

fdata = np.loadtxt("build/function.csv", delimiter=",")
ndata = np.loadtxt("build/newton.csv", delimiter=",")

x = fdata[:,0]
y = fdata[:,1]

xn = ndata[:,0]
yn = ndata[:,1]

plt.figure(figsize=(8,5))
plt.plot(x, y, label="f(x)")
plt.scatter(xn, yn, color="red", zorder=3, label="Newton iterates")
plt.plot(xn, yn, "--", color="red", alpha=0.6)

plt.axhline(0, color="black", linewidth=0.5)
plt.legend()
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Newton's Method on f(x)")
plt.show()
