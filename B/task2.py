############################################################
# Task 2
############################################################

import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 1.0  # Damping coefficient
D = 1.0  # Diffusion coefficient
v0 = 0.0  # Initial velocity
T = 10  # Total time
N = 1000  # Number of time steps
dt = T / N  # Time step size
np.random.seed(42)  # Random seed for reproducibility

# Initialize arrays
t = np.arange(0, T, dt)  # Time array
v = np.zeros(N)  # Velocity array
v[0] = v0  # Set initial velocity

# Generate Wiener process increments (noise term)
dW = np.random.normal(0, np.sqrt(dt), N)  # Wiener process increments

# Solve the Langevin equation using Ito stochastic integrator
for i in range(1, N):
    dv = -gamma * v[i - 1] * dt + np.sqrt(2 * D) * dW[i]  # Ito stochastic differential
    v[i] = v[i - 1] + dv  # Update velocity

# Plot the velocity trajectory
plt.plot(t, v, label="Velocity v(t)")
plt.xlabel("Time (t)")
plt.ylabel("Velocity (v(t))")
plt.title("Langevin Equation: Velocity Trajectory")
plt.legend()
plt.savefig('plots/task2_langevin_velocity.png', bbox_inches='tight')
plt.show()

# Calculate mean and variance of the velocity
mean_v = np.mean(v)
var_v = np.var(v)

print(f"Mean of velocity: {mean_v}")
print(f"Variance of velocity: {var_v}")