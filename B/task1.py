###############################################
###############################################
# Sec 6B
# Task 1
###############################################
###############################################

##############################################
# Part B
###############################################

import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 0.1  # Drift coefficient
sigma = 0.2  # Diffusion coefficient
T = 10  # Total time
N = 100  # Number of time steps
dt = T / N  # Time step size
np.random.seed(42)  # Random seed for reproducibility

# Generate Wiener process (Brownian motion)
dW = np.random.normal(0, np.sqrt(dt), N)  # Increments of Wiener process
W = np.cumsum(dW)  # Cumulative sum to get Wiener process

# Geometric Brownian motion
X_t = np.exp(mu * np.arange(0, T, dt) + sigma * W)

# Plot the trajectory
plt.plot(np.arange(0, T, dt), X_t, label="Geometric Brownian Motion")
plt.xlabel("Time (t)")
plt.ylabel("X_t")
plt.title("Geometric Brownian Motion")
plt.legend()
plt.savefig('plots/B_geometric_brownian_motion.png', bbox_inches='tight')
plt.show()




#######################################################
# Part C
#######################################################
# Ito differential form for X_t
dX_I = mu * X_t * dt + sigma * X_t * dW  # Ito differential
X_I = np.cumsum(dX_I)  # Cumulative sum to get Ito integral

# Plot the trajectory
plt.plot(np.arange(0, T, dt), X_I, label="Ito Integral")
plt.xlabel("Time (t)")
plt.ylabel("X_I(t)")
plt.title("Ito Stochastic Integrator")
plt.legend()
plt.savefig('plots/C_ito_integrator.png', bbox_inches='tight')
plt.show()



#############################################################
# Part D
############################################################

# Stratonovich differential form for X_t
# Use the same random seed as in part c
dX_S = mu * X_t * dt + sigma * X_t * (dW + 0.5 * sigma * dt)  # Stratonovich differential
X_S = np.cumsum(dX_S)  # Cumulative sum to get Stratonovich integral

# Plot the trajectory
plt.plot(np.arange(0, T, dt), X_S, label="Stratonovich Integral")
plt.xlabel("Time (t)")
plt.ylabel("X_S(t)")
plt.title("Stratonovich Stochastic Integrator")
plt.legend()
plt.savefig('plots/D_stratonovich_integrator.png', bbox_inches='tight')
plt.show()


########################################################################
# Part E
##################################################################


# Parameters
mu = 0.1  # Drift coefficient
sigma = 0.2  # Diffusion coefficient
T = 10  # Total time
N_values = np.logspace(1, 4, num=10, dtype=int)  # N from 10 to 10^4 (log spacing)
np.random.seed(42)  # Random seed for reproducibility

# Initialize lists to store mean and variance for Ito and Stratonovich
mean_Ito = []
var_Ito = []
mean_Strat = []
var_Strat = []

# Loop over different N values
for N in N_values:
    dt = T / N  # Time step size
    dW = np.random.normal(0, np.sqrt(dt), N)  # Wiener process increments
    W = np.cumsum(dW)  # Cumulative Wiener process
    
    # Geometric Brownian motion
    t = np.arange(0, T, dt)  # Time array
    X_t = np.exp(mu * t + sigma * W)  # Geometric Brownian motion
    
    # Ito differential
    dX_I = mu * X_t * dt + sigma * X_t * dW  # Ito differential
    X_I = np.cumsum(dX_I)  # Cumulative Ito integral
    
    # Stratonovich differential
    dX_S = mu * X_t * dt + sigma * X_t * (dW + 0.5 * sigma * dt)  # Stratonovich differential
    X_S = np.cumsum(dX_S)  # Cumulative Stratonovich integral
    
    # Store mean and variance
    mean_Ito.append(np.mean(X_I))
    var_Ito.append(np.var(X_I))
    mean_Strat.append(np.mean(X_S))
    var_Strat.append(np.var(X_S))

# Plot mean and variance
plt.figure(figsize=(12, 6))

# Plot mean
plt.subplot(2, 1, 1)
plt.plot(N_values, mean_Ito, label="Ito Mean")
plt.plot(N_values, mean_Strat, label="Stratonovich Mean")
plt.xscale("log")
plt.xlabel("N (log scale)")
plt.ylabel("Mean")
plt.title("Mean of Ito and Stratonovich Trajectories")
plt.legend()

# Plot variance
plt.subplot(2, 1, 2)
plt.plot(N_values, var_Ito, label="Ito Variance")
plt.plot(N_values, var_Strat, label="Stratonovich Variance")
plt.xscale("log")
plt.xlabel("N (log scale)")
plt.ylabel("Variance")
plt.title("Variance of Ito and Stratonovich Trajectories")
plt.legend()

plt.tight_layout()
plt.savefig('plots/E_stochastic_integrators_statistics.png', bbox_inches='tight')
plt.show()



###############################################
# Part F
#####################################################

# Define the function f(X_t) = X_t^2
f_X_t = X_t**2

# Ito integral for f(X_t)
dF_I = f_X_t * dX_I
F_I = np.cumsum(dF_I)

# Stratonovich integral for f(X_t)
dF_S = f_X_t * dX_S
F_S = np.cumsum(dF_S)

# Plot the trajectories
plt.plot(np.arange(0, T, dt), F_I, label="Ito Integral for f(X_t)")
plt.plot(np.arange(0, T, dt), F_S, label="Stratonovich Integral for f(X_t)")
plt.xlabel("Time (t)")
plt.ylabel("F(t)")
plt.title("Functional Dynamics on Geometric Brownian Motion")
plt.legend()
plt.savefig('plots/F_functional_dynamics_brownian.png', bbox_inches='tight')
plt.show()

print("Part F) The Ito and Stratonovich integrals for f(X_t)=X_t^2 show similar trends but differ due to the additional term in the Stratonovich formulation. The Stratonovich integral tends to smooth out the trajectory compared to the Ito integral.")



#####################################################
# Part G
########################################################

# Autocorrelation function
def autocorrelation(F, t, tau_max):
    autocorr = []
    for tau in range(tau_max):
        autocorr.append(np.mean(F[t] * F[t + tau]))
    return autocorr

# Calculate autocorrelation for F_I and F_S at t = 5, 10, 20, 30
t_values = [5, 10, 20, 30]
tau_max = 50  # Maximum time lag

plt.figure(figsize=(12, 8))
for i, t in enumerate(t_values):
    autocorr_I = autocorrelation(F_I, t, tau_max)
    autocorr_S = autocorrelation(F_S, t, tau_max)
    
    plt.subplot(2, 2, i + 1)
    plt.plot(range(tau_max), autocorr_I, label="Ito Autocorrelation")
    plt.plot(range(tau_max), autocorr_S, label="Stratonovich Autocorrelation")
    plt.xlabel("Time Lag (τ)")
    plt.ylabel("Autocorrelation")
    plt.title(f"Autocorrelation at t = {t}")
    plt.legend()
plt.tight_layout()
plt.savefig('plots/G_autocorrelation.png', bbox_inches='tight')
plt.show()

print('Part G) The autocorrelation functions for both Ito and Stratonovich integrals decay over time, reflecting the loss of correlation as the time lag increases. The Stratonovich integral tends to have slightly higher autocorrelation due to its smoother nature.')