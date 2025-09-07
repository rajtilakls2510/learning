import matplotlib.pyplot as plt

# Load data
path = []
with open("./assets/rs_path.txt", "r") as f:
    for line in f:
        x, y, theta = map(float, line.strip().split())
        path.append((x, y, theta))

# Unpack x, y for plotting
x_vals, y_vals, _ = zip(*path)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Reeds-Shepp Path", linewidth=2)
plt.scatter(x_vals, y_vals, c='green')
# plt.scatter(x_vals[-1], y_vals[-1], c='red', label='Goal')
plt.title("Reeds-Shepp Path")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
