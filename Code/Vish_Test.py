import numpy as np
import matplotlib.pyplot as plt

# model parameters
best_beta = 0.5
best_sigma = 0.1
best_gamma = 0.07777777777777778

# initial population
S0, E0, I0, R0 = 38000, 0, 1, 0
N = S0 + E0 + I0 + R0

# time grid
t_end = 200
dt = 0.1
t = np.arange(0, t_end + dt, dt)

# Intervention Timing & Efficacy
vax_day = 70
vax_total = 2000
vax_eff = 0.9

# Date-based multipliers for Beta
# Social Distancing (-10%), Masks (-20%), Schools (-15%)
# Combined effect: 0.9 * 0.8 * 0.85 = ~0.612 (approx 39% reduction)
intervention_start = 50 
reduction_factor = 0.9 * 0.8 * 0.85 

S_all = np.zeros_like(t); E_all = np.zeros_like(t)
I_all = np.zeros_like(t); R_all = np.zeros_like(t)
S_all[0], E_all[0], I_all[0], R_all[0] = S0, E0, I0, R0

for i in range(1, len(t)):
    curr_t = t[i]
    
    # Determine current Beta based on intervention status
    current_beta = best_beta
    if curr_t >= intervention_start:
        current_beta = best_beta * reduction_factor

    # SEIR Equations
    dS = -current_beta * S_all[i-1] * I_all[i-1] / N
    dE = current_beta * S_all[i-1] * I_all[i-1] / N - best_sigma * E_all[i-1]
    dI = best_sigma * E_all[i-1] - best_gamma * I_all[i-1]
    dR = best_gamma * I_all[i-1]

    S_all[i] = S_all[i-1] + dt * dS
    E_all[i] = E_all[i-1] + dt * dE
    I_all[i] = I_all[i-1] + dt * dI
    R_all[i] = R_all[i-1] + dt * dR

    # Apply one-time Vaccination Event
    if abs(curr_t - vax_day) < (dt / 2):
        actual_protected = vax_total * vax_eff
        S_all[i] -= vax_total
        R_all[i] += actual_protected
        S_all[i] += (vax_total - actual_protected) # The 10% move back to S

