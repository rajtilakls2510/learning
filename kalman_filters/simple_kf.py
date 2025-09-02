import numpy as np
import matplotlib.pyplot as plt

# Estimation of position using constant velocity

# Simulation parameters
dt = 1.0          # time step (s)
num_steps = 50    # number of steps
true_velocity = 1.0  # constant velocity (m/s)

# Process and measurement noise
process_noise_std = 1.0
measurement_noise_std = 5.0

# Arrays to store true values and measurements
true_positions = []
true_velocities = []
measurements = []
filtered_positions = []
filtered_velocities = []

# Initial state
x = 0.0
v = true_velocity

# Kalman Filter initialization
state_dim = N = 2
measurement_dim = M = 1
initial_process_noise_var = 100.0
x_posteriori = np.array([x, v]).reshape(-1,1)
process_noise_posteriori = P_posteriori = np.eye(N) * initial_process_noise_var 
system_matrix = F = np.array([[1, dt], [0, 1]])
# input_matrix = G = np.array([0, 0])
output_matrix = H = np.array([1, 0]).reshape(1,-1)
process_noise_estimate = Q = np.eye(N) * 1.0
measurement_noise_estimate = R = np.eye(M) * 100

# Simulate robot motion
for k in range(num_steps):
    # True state update (constant velocity + small process noise)
    x = x + v*dt + np.random.normal(0, process_noise_std)
    v = true_velocity + np.random.normal(0, process_noise_std)
    
    # Measurement (position only, noisy)
    y = x + np.random.normal(0, measurement_noise_std)

    # KF prediction
    x_priori = F @ x_posteriori # We don't have an input here
    P_priori = F @ P_posteriori @ F.T + Q

    # KF Update
    print("H: ", H, H.shape)
    H_T = H.T#reshape(-1,1)
    print("H_T: ", H_T, H_T.shape)

    K_gain = P_priori @ H_T @ np.linalg.inv(H @ P_priori @ H_T + R)
    x_posteriori = x_priori + K_gain @ (y - H @ x_priori)
    L = (np.eye(N) - K_gain @ H)
    P_posteriori = L @ P_priori @ L.T + K_gain @ R @ K_gain.T

    # Store values
    true_positions.append(x)
    true_velocities.append(v)
    measurements.append(y)
    filtered_positions.append(x_posteriori[0])
    filtered_velocities.append(x_posteriori[1])

# Plot true position and noisy measurements
plt.figure(figsize=(10, 5))
plt.plot(range(num_steps), true_positions, label="True Position", linewidth=2)
plt.plot(range(num_steps), filtered_positions, label="Filtered Position", linewidth=2)
plt.plot(range(num_steps), true_velocities, label="True Velocity", linewidth=2)
plt.plot(range(num_steps), filtered_velocities, label="Filtered Velocity", linewidth=2)
plt.scatter(range(num_steps), measurements, color="red", label="Measurements", alpha=0.6)
plt.xlabel("Time step")
plt.ylabel("Position")
plt.title("1D Robot Motion with Noisy Position Measurements")
plt.legend()
plt.grid(True)
plt.show()
