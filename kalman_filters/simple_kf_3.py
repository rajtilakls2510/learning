import numpy as np
import matplotlib.pyplot as plt

# Estimation of 2D position, velocity and acceleration from GPS and IMU

# ============================
# Simulation parameters
# ============================
dt = 0.1               # time step (s)
num_steps = 200        # number of steps
time = np.arange(0, num_steps*dt, dt)

# Process noise (true dynamics)
process_noise_std = 0.05   # noise in true acceleration
# Sensor noise
gps_noise_std = 2.0        # GPS position noise
imu_noise_std = 0.2        # IMU acceleration noise

# ============================
# True trajectory arrays
# ============================
true_positions = []
true_velocities = []
true_accelerations = []

# Sensor measurements
gps_measurements = []
imu_measurements = []

# ============================
# Initial true state
# State: [x, y, vx, vy]
# ============================
x, y = 0.0, 0.0
vx, vy = 1.0, 0.5   # initial velocities

# ============================
# Simulation loop
# ============================
for k in range(num_steps):
    t = k * dt

    # Time-varying acceleration (sinusoidal + process noise)
    ax_true = 1.0 * np.sin(0.2*t) + np.random.normal(0, process_noise_std)
    ay_true = 0.6 * np.cos(0.1*t) + np.random.normal(0, process_noise_std)

    # Update true state
    x = x + vx*dt + 0.5*ax_true*dt**2
    y = y + vy*dt + 0.5*ay_true*dt**2
    vx = vx + ax_true*dt
    vy = vy + ay_true*dt

    # Sensor measurements
    gps_x = x + np.random.normal(0, gps_noise_std)
    gps_y = y + np.random.normal(0, gps_noise_std)
    imu_ax = ax_true + np.random.normal(0, imu_noise_std)
    imu_ay = ay_true + np.random.normal(0, imu_noise_std)

    # Store data
    true_positions.append([x, y])
    true_velocities.append([vx, vy])
    true_accelerations.append([ax_true, ay_true])
    gps_measurements.append([gps_x, gps_y])
    imu_measurements.append([imu_ax, imu_ay])

# Convert to arrays
true_positions = np.array(true_positions)
true_velocities = np.array(true_velocities)
true_accelerations = np.array(true_accelerations)
gps_measurements = np.array(gps_measurements)
imu_measurements = np.array(imu_measurements)

# ============================
# Filtering loop
# ============================

# Initial KF estimate (STATE: [x, y, vx, vy, ax, ay])
state_dim = N = 6
measurement_dim = M = 4

x_pos = np.array([0, 0, 5, -5, 0, 0]).reshape(-1,1)
P_pos = np.eye(N) * 100.0

# System matrix
F = np.array([
    [1, 0, dt, 0, 0.5*dt**2, 0],
    [0, 1, 0, dt, 0, 0.5*dt**2],
    [0, 0, 1, 0, dt, 0],
    [0, 0, 0, 1, 0, dt],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

# Input matrix
# G = np.array([
#     [0.5*dt**2, 0],
#     [0, 0.5*dt**2],
#     [dt, 0],
#     [0, dt]
# ])

# Output matrix
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

# Process noise cov estimate
process_noise_std_estimate = 0.05
Q = np.eye(N) * process_noise_std_estimate**2

# Measurement noise cov estimate
gps_noise_std_estimate = 5.0 # From manufacturer
imu_noise_std_estimate = 0.5 # From manufacturer
R = np.eye(M)
R[:2] *= gps_noise_std_estimate**2
R[2:] *= imu_noise_std_estimate**2

filtered_positions = []
filtered_velocities = []
filtered_accelerations = []

for k in range(num_steps):

    # KF Prediction
    #ax, ay = imu_measurements[k]
    #u = np.array([ax, ay]).reshape(-1, 1)
    x_prior = F @ x_pos #+ G @ u
    P_prior = F @ P_pos @ F.T + Q

    # KF Update
    x, y = gps_measurements[k]
    ax, ay = imu_measurements[k]
    y = np.array([x, y, ax, ay]).reshape(-1, 1)
    K_gain = P_prior @ H.T @ np.linalg.inv(H @ P_prior @ H.T + R)
    x_pos = x_prior + K_gain @ (y - H @ x_prior)
    L = (np.eye(N) - K_gain @ H)
    P_pos = L @ P_prior @ L.T + K_gain @ R @ K_gain.T

    filtered_positions.append(x_pos[:2])
    filtered_velocities.append(x_pos[2:4])
    filtered_accelerations.append(x_pos[4:])

filtered_positions = np.array(filtered_positions)
filtered_velocities = np.array(filtered_velocities)
filtered_accelerations = np.array(filtered_accelerations)


# ============================
# Plot results
# ============================

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Position (X and Y)
axs[0].plot(time, true_positions[:,0], label="True X pos")
axs[0].plot(time, true_positions[:,1], label="True Y pos")
axs[0].scatter(time, gps_measurements[:,0], s=10, color="red", alpha=0.5, label="GPS X means")
axs[0].scatter(time, gps_measurements[:,1], s=10, color="orange", alpha=0.5, label="GPS Y means")
axs[0].plot(time, filtered_positions[:,0], color="red", alpha=0.5, label="Filtered X means")
axs[0].plot(time, filtered_positions[:,1], color="orange", alpha=0.5, label="Filtered Y means")
axs[0].set_ylabel("Position")
axs[0].legend()
axs[0].grid()

# Velocity (X and Y)
axs[1].plot(time, true_velocities[:,0], label="True Vx", color="blue")
axs[1].plot(time, true_velocities[:,1], label="True Vy", color="green")
axs[1].plot(time, filtered_velocities[:,0], label="Filtered Vx", color="red")
axs[1].plot(time, filtered_velocities[:,1], label="Filtered Vy", color="orange")
axs[1].set_ylabel("Velocity")
axs[1].legend()
axs[1].grid()

# Acceleration (X and Y)
axs[2].plot(time, true_accelerations[:,0], label="True Ax", color="purple")
axs[2].plot(time, true_accelerations[:,1], label="True Ay", color="brown")
axs[2].scatter(time, imu_measurements[:,0], s=10, color="cyan", alpha=0.5, label="IMU Ax")
axs[2].scatter(time, imu_measurements[:,1], s=10, color="magenta", alpha=0.5, label="IMU Ay")
axs[2].plot(time, filtered_accelerations[:,0], color="purple", alpha=0.5, label="Filtered Ax")
axs[2].plot(time, filtered_accelerations[:,1], color="brown", alpha=0.5, label="Filtered Ay")
axs[2].set_ylabel("Acceleration")
axs[2].set_xlabel("Time [s]")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()
