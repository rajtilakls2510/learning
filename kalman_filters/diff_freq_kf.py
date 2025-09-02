import numpy as np
import matplotlib.pyplot as plt

# ============================
# Simulation parameters
# ============================
dt = 0.01              # time step (s) -> 100 Hz
num_steps = 2000       # simulate for 20s
time = np.arange(0, num_steps*dt, dt)

# Process noise (true dynamics)
process_noise_std = 0.05   # noise in true acceleration
# Sensor noise
gps_noise_std = 2.0        # GPS position noise
imu_noise_std = 0.2        # IMU acceleration noise

gps_rate = 5    # Hz
imu_rate = 100  # Hz
gps_interval = int(imu_rate / gps_rate)  # update GPS every 20 steps

# ============================
# True trajectory arrays
# ============================
true_positions = []
true_velocities = []
true_accelerations = []

# Sensor measurements
gps_measurements = [None] * num_steps   # GPS at 5Hz
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

    # GPS only every gps_interval steps
    if k % gps_interval == 0:
        gps_x = x + np.random.normal(0, gps_noise_std)
        gps_y = y + np.random.normal(0, gps_noise_std)
        gps_measurements[k] = np.array([gps_x, gps_y])

    # IMU every step
    imu_ax = ax_true + np.random.normal(0, imu_noise_std)
    imu_ay = ay_true + np.random.normal(0, imu_noise_std)

    # Store data
    true_positions.append([x, y])
    true_velocities.append([vx, vy])
    true_accelerations.append([ax_true, ay_true])
    imu_measurements.append([imu_ax, imu_ay])

# Convert to arrays
true_positions = np.array(true_positions)
true_velocities = np.array(true_velocities)
true_accelerations = np.array(true_accelerations)
imu_measurements = np.array(imu_measurements)

# ============================
# Filtering loop
# ============================

# Initial KF estimate (STATE: [x, y, vx, vy, ax, ay])
state_dim = N = 6
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

process_noise_std_estimate = 0.05
Q = np.eye(N) * process_noise_std_estimate**2  # process noise estimate


gps_noise_std_estimate = 5.0 # From manufacturer
imu_noise_std_estimate = 0.5 # From manufacturer

filtered_positions = []
filtered_velocities = []
filtered_accelerations = []

for k in range(num_steps):
    # KF Prediction
    x_prior = F @ x_pos
    P_prior = F @ P_pos @ F.T + Q

    # Measurement update depends on available sensor
    H_list, z_list, R_list = [], [], []

    # GPS update only every gps_interval steps
    if gps_measurements[k] is not None:
        z_gps = gps_measurements[k].reshape(-1,1)
        H_gps = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])
        R_gps = np.eye(2) * gps_noise_std_estimate**2
        H_list.append(H_gps); z_list.append(z_gps); R_list.append(R_gps)

    # IMU update (every step)
    z_imu = imu_measurements[k].reshape(-1,1)
    H_imu = np.array([[0,0,0,0,1,0],[0,0,0,0,0,1]])
    R_imu = np.eye(2) * imu_noise_std_estimate**2
    H_list.append(H_imu); z_list.append(z_imu); R_list.append(R_imu)

    # Stack measurements if multiple
    if len(H_list) > 0:
        H = np.vstack(H_list)
        z = np.vstack(z_list)
        R = np.block([[R_list[i] if i==j else np.zeros_like(R_list[i])
                       for j in range(len(R_list))] for i in range(len(R_list))])

        K_gain = P_prior @ H.T @ np.linalg.inv(H @ P_prior @ H.T + R)
        x_pos = x_prior + K_gain @ (z - H @ x_prior)
        P_pos = (np.eye(N) - K_gain @ H) @ P_prior
    else:
        x_pos, P_pos = x_prior, P_prior

    filtered_positions.append(x_pos[:2])
    filtered_velocities.append(x_pos[2:4])
    filtered_accelerations.append(x_pos[4:])

filtered_positions = np.array(filtered_positions).squeeze()
filtered_velocities = np.array(filtered_velocities).squeeze()
filtered_accelerations = np.array(filtered_accelerations).squeeze()

# ============================
# Plot results
# ============================
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(time, true_positions[:,0], label="True X")
axs[0].plot(time, true_positions[:,1], label="True Y")
axs[0].scatter(time[::gps_interval], [gps[0] for gps in gps_measurements if gps is not None],
               s=10, color="red", alpha=0.5, label="GPS X")
axs[0].scatter(time[::gps_interval], [gps[1] for gps in gps_measurements if gps is not None],
               s=10, color="orange", alpha=0.5, label="GPS Y")
axs[0].plot(time, filtered_positions[:,0], label="Filtered X", color="red", alpha=0.7)
axs[0].plot(time, filtered_positions[:,1], label="Filtered Y", color="orange", alpha=0.7)
axs[0].set_ylabel("Position")
axs[0].legend(); axs[0].grid()

axs[1].plot(time, true_velocities[:,0], label="True Vx")
axs[1].plot(time, true_velocities[:,1], label="True Vy")
axs[1].plot(time, filtered_velocities[:,0], label="Filtered Vx", color="red", alpha=0.7)
axs[1].plot(time, filtered_velocities[:,1], label="Filtered Vy", color="orange", alpha=0.7)
axs[1].set_ylabel("Velocity")
axs[1].legend(); axs[1].grid()

axs[2].plot(time, true_accelerations[:,0], label="True Ax")
axs[2].plot(time, true_accelerations[:,1], label="True Ay")
axs[2].scatter(time, imu_measurements[:,0], s=5, color="cyan", alpha=0.5, label="IMU Ax")
axs[2].scatter(time, imu_measurements[:,1], s=5, color="magenta", alpha=0.5, label="IMU Ay")
axs[2].plot(time, filtered_accelerations[:,0], label="Filtered Ax", color="purple", alpha=0.7)
axs[2].plot(time, filtered_accelerations[:,1], label="Filtered Ay", color="brown", alpha=0.7)
axs[2].set_ylabel("Acceleration")
axs[2].set_xlabel("Time [s]")
axs[2].legend(); axs[2].grid()

plt.tight_layout()
plt.show()
