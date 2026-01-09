# ============================================================
# Augmented Lagrangian – 2D Visualization
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl


mpl.rcParams.update({
    "font.size": 20,           # base font size
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 20,
})

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
cost  = np.loadtxt("build/cost.csv", delimiter=",")
eq    = np.loadtxt("build/equalities.csv", delimiter=",")
ineq  = np.loadtxt("build/inequalities.csv", delimiter=",")
steps = np.loadtxt("build/al_steps.csv", delimiter=",")#[:50]

# ------------------------------------------------------------
# Grid reconstruction
# ------------------------------------------------------------
N = int(np.sqrt(len(cost)))

X = cost[:, 0].reshape(N, N)
Y = cost[:, 1].reshape(N, N)
Z = cost[:, 2].reshape(N, N)

h1 = eq[:, 2].reshape(N, N)
h2 = eq[:, 3].reshape(N, N)

g1 = ineq[:, 2].reshape(N, N)
g2 = ineq[:, 3].reshape(N, N)

# ------------------------------------------------------------
# Feasibility mask
# ------------------------------------------------------------
feasible = (g1 <= 0) & (g2 <= 0)

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 20))

ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_title("Augmented Lagrangian Optimization (2D)")

# ------------------------------------------------------------
# Cost contours (full domain)
# ------------------------------------------------------------
levels = np.linspace(np.percentile(Z, 5), np.percentile(Z, 95), 30)

cf = ax.contourf(
    X, Y, Z,
    levels=levels,
    cmap="viridis",
    alpha=0.85
)

plt.colorbar(cf, ax=ax, label="f(x)")

# ------------------------------------------------------------
# Infeasible region shading
# ------------------------------------------------------------
ax.contourf(
    X, Y, feasible,
    levels=[-0.5, 0.5, 1.5],
    colors=["none", "lightgray"],
    alpha=0.6
)

# ------------------------------------------------------------
# Constraints
# ------------------------------------------------------------
# Equality constraints
ax.contour(X, Y, h1, levels=[0], colors="green", linewidths=2)
ax.contour(X, Y, h2, levels=[0], colors="orange", linewidths=2)

# Inequality boundaries
ax.contour(X, Y, g1, levels=[0], colors="red",   linestyles="dashed", linewidths=2)
ax.contour(X, Y, g2, levels=[0], colors="blue",  linestyles="dashed", linewidths=2)

# ------------------------------------------------------------
# Text annotations (no LaTeX dependency)
# ------------------------------------------------------------
ax.text(
    0.01, 0.99,
    "min  0.1(1−x₁)² + 0.001(x₂−x₁²)² + 0.1 sin(3x₁) sin(3x₂)\n"
    "s.t.\n"
    "  sin(1.75 x₁) + x₂² − 1 = 0\n"
    "  x₁ x₂ − 0.25 = 0\n"
    "  x₁² + x₂² − 2 ≤ 0\n"
    "  x₂ − 0.1 ≤ 0",
    transform=ax.transAxes,
    fontsize=20,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

# ------------------------------------------------------------
# Optimization trajectory (fading)
# ------------------------------------------------------------
points = []
lines  = []

pause_seconds = 0        # how long to wait
fps = 4                  # animation FPS
pause_frames = int(pause_seconds * fps)  # number of frames to wait

def update(frame):
    if frame < pause_frames:
        # Only show the static plot, no points
        ax.set_title(f"Augmented Lagrangian starting soon...")
        return []
    
    # Step index after pause
    step_idx = frame - pause_frames
    x, y = steps[step_idx]

    # Fade previous
    for p in points:
        p.set_alpha(max(p.get_alpha() * 0.25, 0.001))
    for l in lines:
        l.set_alpha(max(l.get_alpha() * 0.25, 0.001))
        l.set_linewidth(max(l.get_linewidth() * 0.9, 0.5))

    # Plot current step
    p = ax.scatter(x, y, color="red", s=100, alpha=1.0, zorder=5)
    points.append(p)

    if step_idx > 0:
        x0, y0 = steps[step_idx - 1]
        l, = ax.plot([x0, x], [y0, y], color="black", linewidth=3, alpha=1.0, zorder=4)
        lines.append(l)

    ax.set_title(f"Augmented Lagrangian iteration {step_idx+1}/{len(steps)}")
    return points + lines

# Total frames = pause_frames + actual steps
anim = FuncAnimation(
    fig,
    update,
    frames=pause_frames + len(steps),
    interval=10,
    repeat=False
)


ax.set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()

# from matplotlib.animation import FFMpegWriter

# writer = FFMpegWriter(
#     fps=4,
#     metadata=dict(artist="Rajtilak Pal"),
#     bitrate=1800
# )

# anim.save("augmented_lagrangian_2d.mp4", writer=writer)

# plt.show()
