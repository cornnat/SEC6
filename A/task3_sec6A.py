################################################
# Question 1
###########################################################

import numpy as np
from itertools import product

# # Explanation:
# # We consider a 3-spin chain with periodic boundary conditions.
# # Each spin can be either up (+1) or down (-1). The basis states are represented as tuples.
# # The Hamiltonian for the Heisenberg XXX model (with J=1) is given by:
# #   H = 3/4 - [ 1/2 (S+_1 S-_2 + S-_1 S+_2) + S^z_1 S^z_2 +
# #               1/2 (S+_2 S-_3 + S-_2 S+_3) + S^z_2 S^z_3 +
# #               1/2 (S+_3 S-_1 + S-_3 S+_1) + S^z_3 S^z_1 ]
# #
# # We use the “hopping” (off-diagonal) elements to define the transitions in a Markov chain.
# # Specifically, the transition probability from state i to state j is proportional to
# # the absolute value of the corresponding off-diagonal Hamiltonian element, normalized by the sum of such weights in that row.

# Define the basis states as all combinations of three spins (+1 for up, -1 for down).
basis = list(product([1, -1], repeat=3))
n_states = len(basis)  # should be 2^3 = 8

# Define single-spin operators.
def Splus(s):
    # S+ flips a down spin (-1) to up (+1), zero if already up.
    return 1 if s == -1 else 0

def Sminus(s):
    # S- flips an up spin (+1) to down (-1), zero if already down.
    return 1 if s == 1 else 0

def Sz(s):
    # S^z gives spin value divided by 2.
    return s / 2

# Function to apply the hopping operators on a pair (i,j) in a given state.
def apply_operator_pair(state, i, j):
    # This function applies (1/2)*(S+_i S-_j + S-_i S+_j) on a given state.
    new_states = []
    # First term: S+_i S-_j, which requires spin at i is down and at j is up.
    if state[i] == -1 and state[j] == 1:
        new_state = list(state)
        new_state[i] = 1
        new_state[j] = -1
        new_states.append((tuple(new_state), 0.5))
    # Second term: S-_i S+_j, which requires spin at i is up and at j is down.
    if state[i] == 1 and state[j] == -1:
        new_state = list(state)
        new_state[i] = -1
        new_state[j] = 1
        new_states.append((tuple(new_state), 0.5))
    return new_states

# Build the Hamiltonian matrix H (8 x 8) in the site basis.
H = np.zeros((n_states, n_states), dtype=float)
J = 1.0  # coupling constant
# The pairs with periodic boundary conditions: (0,1), (1,2), (2,0)
pairs = [(0, 1), (1, 2), (2, 0)]

# Loop over basis states to fill H.
for idx, state in enumerate(basis):
    # Diagonal part from Sz_i Sz_j contributions.
    diag_val = 0.0
    for (i, j) in pairs:
        diag_val += Sz(state[i]) * Sz(state[j])
    # Include the constant shift (3/4) and the sign: note that H = 3/4 - J*( ... )
    H[idx, idx] += 0.75 - J * diag_val

    # Off-diagonal contributions from the hopping terms.
    for (i, j) in pairs:
        contributions = apply_operator_pair(state, i, j)
        for new_state, amp in contributions:
            jdx = basis.index(new_state)
            H[idx, jdx] += -J * amp  # multiply by -J

# # Construct the Markov chain transition matrix P from the Hamiltonian.
# # For each state (row), use the absolute value of off-diagonal elements as weights,
# # then normalize so that the sum of probabilities in that row is 1.
P_site = np.zeros_like(H)
for i in range(n_states):
    # Collect weights for off-diagonal elements.
    weights = np.array([abs(H[i, j]) for j in range(n_states) if j != i])
    total_weight = weights.sum()
    if total_weight > 0:
        for j in range(n_states):
            if j != i:
                P_site[i, j] = abs(H[i, j]) / total_weight
        # Optionally, set the self-transition to zero (or leave it out since the row is normalized).
        P_site[i, i] = 0.0
    else:
        # If there are no off-diagonal transitions, the state is absorbing.
        P_site[i, i] = 1.0

# Print the resulting transition matrix.
print("# Transition Matrix P (site basis):")
print(P_site)

################################################
# Question 2
####################################################

# We wish to solve π P = π, subject to ∑_i π_i = 1.
# This is equivalent to finding the eigenvector of P_site^T corresponding to eigenvalue 1.

# Compute the eigenvalues and eigenvectors of the transpose of P_site.
eigvals, eigvecs = np.linalg.eig(P_site.T)

# Identify the eigenvector corresponding to eigenvalue 1 (within a small numerical tolerance).
idx = np.argmin(np.abs(eigvals - 1))
pi_stationary = np.real(eigvecs[:, idx])

# Normalize the eigenvector so that the entries sum to 1.
pi_stationary /= np.sum(pi_stationary)

# Print the stationary distribution.
print("# Stationary distribution (site basis):")
print(pi_stationary)




#############################################################
# Question 3
#############################################################

# We use the power iteration method to compute the stationary distribution.
# The iterative update is given by:
#     π_(k+1) = π_k P_site
# where P_site is the transition matrix constructed earlier.
#
# We consider three different initial guesses:
#   1) All probability in the state |↑↑↑⟩, i.e., state (1, 1, 1).
#   2) 50% probability in |↑↑↑⟩ and 50% in |↓↑↓⟩, i.e., states (1, 1, 1) and (-1, 1, -1).
#   3) A uniform distribution over all states.

# Define the power iteration function.
def power_iteration(P, pi0, tol=1e-10, max_iter=1000):
    pi = pi0.copy()
    for _ in range(max_iter):
        pi_next = pi @ P
        if np.linalg.norm(pi_next - pi, 1) < tol:
            return pi_next
        pi = pi_next
    return pi

# Initial guess 1: all probability in |↑↑↑⟩ (state (1, 1, 1))
pi0_1 = np.zeros(n_states)
pi0_1[basis.index((1, 1, 1))] = 1.0

# Initial guess 2: 50% in |↑↑↑⟩ and 50% in |↓↑↓⟩ (states (1, 1, 1) and (-1, 1, -1))
pi0_2 = np.zeros(n_states)
pi0_2[basis.index((1, 1, 1))] = 0.5
pi0_2[basis.index((-1, 1, -1))] = 0.5

# Initial guess 3: Uniform distribution
pi0_3 = np.ones(n_states) / n_states

# Compute the stationary distributions using power iteration.
pi_site_1 = power_iteration(P_site, pi0_1)
pi_site_2 = power_iteration(P_site, pi0_2)
pi_site_3 = power_iteration(P_site, pi0_3)

# Print the results.
print("# Stationary distribution (site basis) from power iteration with initial guess |↑↑↑>:")
print(pi_site_1)
print("\n# Stationary distribution (site basis) from power iteration with initial guess |↑↑↑> and |↓↑↓> equally weighted:")
print(pi_site_2)
print("\n# Stationary distribution (site basis) from power iteration with initial uniform distribution:")
print(pi_site_3)


#############################################################
# Question 4: Markov Chain in Magnon Basis
#############################################################
# In the magnon basis, the ferromagnetic vacuum is |0> = |↑↑↑>.
# A one-magnon state is defined as:
#     |p> = ∑_n exp(i p n) S^-_n |0>
# For N=3, allowed momenta: p = 2πk/3,  k = 0,1,2.
# Their energies are given by: E_k = 2J sin²(πk/3), with J = 1.
N = 3
k_vals = np.array([0, 1, 2])
E = 2 * np.sin(np.pi * k_vals / N)**2  # energies for k=0,1,2

print("# Magnon energies:")
for k, energy in zip(k_vals, E):
    print(f"k = {k}, E = {energy:.4f}")

# Now, construct the Markov chain in the magnon basis.
# We assume a Boltzmann (or Metropolis) rule for the transitions.
# For a transition from state i to state j, if ΔE = E_j - E_i <= 0, set weight=1;
# if ΔE > 0, set weight = exp(-ΔE/T), with temperature T (k_B set to 1).
T_temperature = 1.0  # Temperature (can be varied)
n_magnon = len(k_vals)
P_magnon = np.zeros((n_magnon, n_magnon))
for i in range(n_magnon):
    for j in range(n_magnon):
        if i != j:
            dE = E[j] - E[i]
            if dE <= 0:
                P_magnon[i, j] = 1.0
            else:
                P_magnon[i, j] = np.exp(-dE / T_temperature)
    # Normalize row i so that the total probability is 1.
    row_sum = np.sum(P_magnon[i, :])
    if row_sum > 0:
        P_magnon[i, :] /= row_sum
    else:
        P_magnon[i, i] = 1.0

print("# Transition Matrix P (magnon basis):")
print(P_magnon)


#############################################################
# Question 5: Stationary Distribution in Magnon Basis
#############################################################
# Solve for π in the magnon basis by finding the eigenvector corresponding to eigenvalue 1.
eigvals_m, eigvecs_m = np.linalg.eig(P_magnon.T)
idx_m = np.argmin(np.abs(eigvals_m - 1))
pi_magnon = np.real(eigvecs_m[:, idx_m])
pi_magnon /= np.sum(pi_magnon)
print("# Stationary distribution (magnon basis) using eigenvector method:")
print(pi_magnon)


#############################################################
# Question 6: Power Iteration in Magnon Basis
#############################################################
# Define initial guesses in the magnon basis:
# 1) All probability in |k = 1> (state index 1).
pi0_m1 = np.zeros(n_magnon)
pi0_m1[1] = 1.0

# 2) 50% in |k = 1> and 50% in |k = 2> (states index 1 and 2).
pi0_m2 = np.zeros(n_magnon)
pi0_m2[1] = 0.5
pi0_m2[2] = 0.5

# 3) Uniform distribution.
pi0_m3 = np.ones(n_magnon) / n_magnon

pi_magnon_1 = power_iteration(P_magnon, pi0_m1)
pi_magnon_2 = power_iteration(P_magnon, pi0_m2)
pi_magnon_3 = power_iteration(P_magnon, pi0_m3)

print("# Stationary distributions (magnon basis) from power iteration:")
print("Initial guess 1:", pi_magnon_1)
print("Initial guess 2:", pi_magnon_2)
print("Initial guess 3:", pi_magnon_3)


#############################################################
# Question 7: Master Equation Evolution in Magnon Basis (N = 9, k = 0 through 8)
#############################################################
import numpy as np
from scipy.linalg import logm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Set the number of magnon states (N = 4, corresponding to k = 0, 1, ..., 3)
N = 4
k_vals = np.arange(N)  # k = 0, 1, ..., 3

# Compute the magnon energies:
# Energy dispersion: E(k) = 2J sin^2(pi*k/N), here with J = 1.
E = 2 * np.sin(np.pi * k_vals / N)**2

print("# Magnon energies for N=9:")
for k, energy in zip(k_vals, E):
    print(f"k = {k}, E = {energy:.4f}")

# Build the transition matrix P_magnon in the magnon basis.
# We assume Boltzmann-type transitions:
# If ΔE = E_j - E_i <= 0, assign weight = 1; if ΔE > 0, assign weight = exp(-ΔE/T).
T_temperature = 1.0  # temperature (with k_B = 1)
n_magnon = N  # 4 states
P_magnon = np.zeros((n_magnon, n_magnon))
for i in range(n_magnon):
    for j in range(n_magnon):
        if i != j:
            dE = E[j] - E[i]
            if dE <= 0:
                P_magnon[i, j] = 1.0
            else:
                P_magnon[i, j] = np.exp(-dE / T_temperature)
    # Normalize row i so that the total probability is 1.
    row_sum = np.sum(P_magnon[i, :])
    if row_sum > 0:
        P_magnon[i, :] /= row_sum
    else:
        P_magnon[i, i] = 1.0

print("\n# Transition Matrix P (magnon basis, N=4):")
print(P_magnon)

# Convert the discrete-time transition matrix into a continuous-time rate matrix Q.
# Use Q ≈ (1/(n_steps * Δt)) * ln(P^n_steps)
n_steps = 10
delta_t = 1.0

# Compute P_magnon raised to the power n_steps.
P_magnon_n = np.linalg.matrix_power(P_magnon, n_steps)
# Compute the matrix logarithm and then scale.
Q = (1.0 / (n_steps * delta_t)) * np.real(logm(P_magnon_n))
print("\n# Transition rate matrix Q (magnon basis, N=4):")
print(Q)

# Define the master equation: dπ/dt = π Q
def master_equation(t, pi):
    return pi @ Q

# Set the initial condition: for example, all probability in state |k = 0>.
pi0_master = np.zeros(n_magnon)
pi0_master[0] = 1.0

# Define the time span and evaluation times.
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 200)
solution = solve_ivp(master_equation, t_span, pi0_master, t_eval=t_eval)

# Plot the time evolution of the probabilities for each magnon state.
plt.figure(figsize=(10, 6))
for i in range(n_magnon):
    plt.plot(solution.t, solution.y[i], label=f"State k = {i}")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.title("Master Equation Evolution in Magnon Basis (N=4)")
plt.legend()
plt.grid(True)
plt.savefig('plots/Q7_master_eq_evolution_magnon_basis_N4.png', bbox_inches='tight')
plt.show()