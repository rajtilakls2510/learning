import numpy as np
import matplotlib.pyplot as plt

# Estimation of (x,y,yaw, velocity) (bicycle model) with (x,y) measurements using EKF


# -----------------------
# Simulation parameters
# -----------------------
dt = 0.1         # time step
T  = 40          # total time [s]
N  = int(T/dt)   # number of steps
L__  = 2.0         # wheelbase

# Noise (measurement)
pos_noise_std = 0.5       # GPS noise [m]
heading_noise_std = 0.2   # heading noise [rad]

# Process noise (applied to state after the deterministic step)
# std deviations for [px, py, theta, v]
proc_noise_std = np.array([0.05, 0.05, 0.01, 0.05])

# -----------------------
# Helpers
# -----------------------
def normalize_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2*np.pi) - np.pi

# -----------------------
# True bicycle dynamics
# -----------------------
def bicycle_step_deterministic(state, u):
    """
    Deterministic kinematic bicycle step (no process noise).
    state = [x, y, theta, v]
    u = [a, delta] (acceleration, steering)
    """
    x, y, theta, v = state
    a, delta = u

    x_next     = x + v * np.cos(theta) * dt
    y_next     = y + v * np.sin(theta) * dt
    theta_next = theta + (v / L__) * np.tan(delta) * dt
    v_next     = v + a * dt

    theta_next = normalize_angle(theta_next)
    return np.array([x_next, y_next, theta_next, v_next])

def measure(state, noise):
    """
    Measurement model
    """
    meas_x, meas_y = state.ravel()[:2]
    meas_x += noise[0]
    meas_y += noise[1]
    # meas_theta = state[2] + np.random.randn() * heading_noise_std
    # meas_theta = normalize_angle(meas_theta)
    return np.array([meas_x, meas_y])

# -----------------------
# Control sequence (hard trajectory)
# -----------------------
def control_policy(k):
    """Return [a, delta] at step k."""
    if k < N*0.15:
        return [0.2, 0.0]       # accelerate straight
    elif k < N*0.30:
        return [0.0, 0.3]       # left turn
    elif k < N*0.45:
        return [-0.1, 0.0]      # decelerate straight
    elif k < N*0.60:
        return [0.1, -0.3]      # accelerate into right turn
    elif k < N*0.75:
        return [0.0, 0.0]       # coast straight
    else:
        return [0.0, 0.25]      # gentle left curve

# -----------------------
# Generate trajectory with process noise
# -----------------------
np.random.seed(1)

state = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, theta, v]
states = [state.copy()]
measurements = []

for k in range(N):
    u = control_policy(k)

    # deterministic update
    state_det = bicycle_step_deterministic(state, u)

    # additive process noise (applied to state)
    process_noise = np.random.randn(4) * proc_noise_std
    state = state_det + process_noise
    state[2] = normalize_angle(state[2])  # keep theta in [-pi,pi]

    states.append(state.copy())

    # noisy measurement: (x, y, theta)
    measure_noise = np.array([
         np.random.randn() * pos_noise_std, 
         np.random.randn() * pos_noise_std
    ])
    measured_y = measure(state, measure_noise)
    measurements.append(measured_y)

states = np.array(states)            # shape (N+1, 4)
measurements = np.array(measurements)  # shape (N, 2)

# -----------------------
# Extended Kalman Filter
# -----------------------


# Filter initialization (State: [x, y, theta, v], Control: [a, delta])
state_dim = N_ = 4
measurement_dim = M_ = 2
x_pos = np.array([0,0,0,0]).reshape(-1,1)  # Intial State
initial_process_noise_std = 10.0
P_pos = np.eye(N_) * initial_process_noise_std**2

process_noise_std_estimate = 0.01
process_noise_estimate = Q = np.eye(N_) * process_noise_std_estimate**2
measurement_noise_std_estimate = 0.1
measurement_noise_estimate = R = np.eye(M_) * measurement_noise_std_estimate**2

filtered_states = [x_pos.ravel()]

def system_jacobian(state_pos, u, w):
    x, y, theta, v = state_pos.ravel()
    a, delta = u.ravel()
    return np.array([
        [1, 0, -v * np.sin(theta) * dt, np.cos(theta) * dt],
        [0, 1, v * np.cos(theta) * dt, np.sin(theta) * dt],
        [0, 0, 1, np.tan(delta) * dt / L__],
        [0, 0, 0, 1]
    ])

def process_noise_jacobian(state_pos, u, w):
    return np.eye(N_)

def measurement_jacobian(state_prior, v):
    return np.array([
        [1,0,0,0],
        [0,1,0,0]
    ])

def measurement_noise_jacobian(state_prior, v):
    return np.array([
        [1,0],
        [0,1]
    ])

for k in range(N):
    u = np.array(control_policy(k))

    # Compute Partial Derivatives of System Model and Process Noise
    F = system_jacobian(x_pos, u, 0)
    L = process_noise_jacobian(x_pos, u, 0)

    # EKF Prediction
    x_prior = bicycle_step_deterministic(x_pos.ravel(), u).reshape(-1,1)
    P_prior = F @ P_pos @ F.T + L @ Q @ L.T

    # Compute Partial Derivatives of Measurement Model and Measurement Noise
    H = measurement_jacobian(x_prior, 0)
    M = measurement_noise_jacobian(x_prior, 0)

    # EKF Update
    y = measurements[k].reshape(-1,1)
    K_gain = P_prior @ H.T @ np.linalg.inv(H @ P_prior @ H.T + M @ R @ M.T)
    y_estimate = measure(x_prior, np.array([0,0])).reshape(-1, 1)
    x_pos = x_prior + K_gain @ (y - y_estimate)
    L_ = (np.eye(N_) - K_gain @ H)
    P_pos = L_ @ P_prior @ L_.T + K_gain @ M @ R @ M.T @ K_gain.T

    filtered_states.append(x_pos.ravel()) 

filtered_states = np.array(filtered_states)


# -----------------------
# Visualization: trajectory + velocities
# -----------------------
true_velocities = states[:,3]  
filtered_velocities = filtered_states[:,3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,12))
ax1.axis("equal")

# plot true trajectory arrows (every M steps)
M = 1
scale_arrow = 0.1
for k in range(0, N+1, M):
    x, y, theta, v = states[k]
    ax1.arrow(x, y, scale_arrow*np.cos(theta), scale_arrow*np.sin(theta),
              head_width=scale_arrow, head_length=scale_arrow, fc='blue', ec='blue')

# # plot noisy measurement arrows (every M steps)
# for k in range(0, N, M):
#     x, y, theta = measurements[k]
#     ax1.arrow(x, y, scale_arrow*np.cos(theta), scale_arrow*np.sin(theta),
#               head_width=0.18, head_length=0.22, fc='red', ec='red', alpha=0.6)

# plot filtered arrows (every M steps)
for k in range(0, N, M):
    x, y, theta = filtered_states[k, :3]
    ax1.arrow(x, y, scale_arrow*np.cos(theta), scale_arrow*np.sin(theta),
              head_width=scale_arrow, head_length=scale_arrow, fc='green', ec='green', alpha=0.6)


# plots
ax1.plot(states[:,0], states[:,1], 'b-', label="True path")
ax1.scatter(measurements[:,0], measurements[:,1], c='red', s=12, alpha=0.6, label="Noisy measurements")
ax1.plot(filtered_states[:,0], filtered_states[:,1], 'g-', label="Filtered path")
ax1.set_title("Bicycle dynamics: complex trajectory (process noise added)")
ax1.legend()
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")

# velocity subplot
time = np.arange(N+1) * dt
ax2.plot(time, true_velocities, 'b-', label="True velocity (v)")
ax2.plot(time, filtered_velocities, 'r-', label="Filtered velocity (v)")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Velocity [m/s]")
ax2.set_title("Velocity: true vs noisy (finite-diff)")
ax2.legend()

plt.tight_layout()
plt.show()
