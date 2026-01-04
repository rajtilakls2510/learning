import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_dynamics.py <data.csv>")
    sys.exit(1)

filename = sys.argv[1]

data = np.loadtxt(filename, delimiter=",")

theta  = data[:, 0]
thetad = data[:, 1]

plt.figure()
plt.plot(theta, label="theta")
plt.plot(thetad, label="theta_dot")
plt.legend()
plt.grid()
plt.xlabel("Time step")
plt.ylabel("State")
plt.title(filename)
plt.show()
