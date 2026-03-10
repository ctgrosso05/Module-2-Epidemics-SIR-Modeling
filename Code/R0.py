import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get directory where THIS script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build full path to CSV
data_path = os.path.join(
    script_dir,
    "..",
    "Data",
    "mystery_virus_daily_active_counts_RELEASE#2.csv"
)

data = pd.read_csv(data_path, parse_dates=['date'])

# Select exponential window
exp_data = data[(data['day'] >= 20) & (data['day'] <= 100)]

log_I = np.log(exp_data['active reported daily cases'])

slope, intercept = np.polyfit(exp_data['day'], log_I, 1)

r = slope
D = 5
R0 = 1 + r * D

print("Estimated growth rate r:", r)
print("Estimated R0:", R0)

# Implementing Euler's method for SEIR model

# Parameters (initial guesses, will be fitted later)
beta = 0.3  # infection rate
sigma = 0.2  # incubation rate (1/sigma = incubation period)
gamma = 0.1  # recovery rate (1/gamma = infectious period)

# Initial conditions (assuming total population N=10000, initial S, E, I, R)
N = 10000
S0 = N - 1
E0 = 0
I0 = 1  # start with 1 infected
R0 = 0

# Time steps
t_start = 0
t_end = 100  # days
dt = 0.1
t = np.arange(t_start, t_end, dt)

# Step 1: Function to implement Euler's method for SEIR
def euler_seir(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    dt = timepoints[1] - timepoints[0]  # assume uniform spacing
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    for i in range(1, len(timepoints)):
        dS = -beta * S[-1] * I[-1] / N
        dE = beta * S[-1] * I[-1] / N - sigma * E[-1]
        dI = sigma * E[-1] - gamma * I[-1]
        dR = gamma * I[-1]
        
        S.append(S[-1] + dt * dS)
        E.append(E[-1] + dt * dE)
        I.append(I[-1] + dt * dI)
        R.append(R[-1] + dt * dR)
    return S, E, I, R

# Step 2: Function to fit parameters using grid search
def fit_seir_parameters(timepoints, N, S0, E0, I0, R0, data):
    # Define ranges for parameters
    beta_range = np.linspace(0.1, 1.0, 10)
    sigma_range = np.linspace(0.1, 0.5, 10)
    gamma_range = np.linspace(0.05, 0.3, 10)
    
    SSE_array = []
    param_combinations = []
    
    for b in beta_range:
        for s in sigma_range:
            for g in gamma_range:
                S, E, I_sim, R = euler_seir(b, s, g, S0, E0, I0, R0, timepoints, N)
                # Interpolate I_sim to data days
                data_days = data['day'].values
                sim_I_at_data_days = np.interp(data_days, timepoints, I_sim)
                observed_I = data['active reported daily cases'].values
                SSE = np.sum((sim_I_at_data_days - observed_I)**2)
                SSE_array.append(SSE)
                param_combinations.append((b, s, g))
    
    # Find best parameters
    min_SSE_idx = np.argmin(SSE_array)
    best_beta, best_sigma, best_gamma = param_combinations[min_SSE_idx]
    best_SSE = SSE_array[min_SSE_idx]
    
    return best_beta, best_sigma, best_gamma, best_SSE

# Fit parameters
best_beta, best_sigma, best_gamma, best_SSE = fit_seir_parameters(t, N, S0, E0, I0, R0, exp_data)

print(f"Best parameters: beta={best_beta}, sigma={best_sigma}, gamma={best_gamma}, SSE={best_SSE}")

# Step 3: Run the model longer to find the peak
# Extend timepoints to 200 days or until peak
t_extended = np.arange(t_start, 200, dt)
S_ext, E_ext, I_ext, R_ext = euler_seir(best_beta, best_sigma, best_gamma, S0, E0, I0, R0, t_extended, N)

# Find peak of I
peak_I = np.max(I_ext)
peak_day = t_extended[np.argmax(I_ext)]

print(f"Peak infected: {peak_I}")
print(f"Is this reasonable? Assuming N=10000, peak at ~{peak_I/N*100:.1f}% of population, which may be plausible for an epidemic.")
print(f"Peak occurs on day: {peak_day}")

# Implement Euler’s method for SEIR modeling (using fitted parameters)
# Plot Euler’s method solutions for I(t) and compare to your data
# Guess beta, sigma, and gamma and calculate SSE (now using fitted)

# Use best parameters for simulation (already done above)
S, E, I, R = euler_seir(best_beta, best_sigma, best_gamma, S0, E0, I0, R0, t, N)

# Interpolate simulated I to match data days
data_days = exp_data['day'].values
sim_I_at_data_days = np.interp(data_days, t, I)

# Calculate SSE with best parameters
observed_I = exp_data['active reported daily cases'].values
SSE = np.sum((sim_I_at_data_days - observed_I)**2)
print(f"SSE with fitted parameters: {SSE}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, I, label='Simulated I(t) - Fitted SEIR')
plt.scatter(exp_data['day'], exp_data['active reported daily cases'], color='red', label='Observed Data')
plt.xlabel('Day')
plt.ylabel('Infected')
plt.title('Fitted SEIR Model vs Data')
plt.legend()
# --- load release #3 and overlay on the existing plot ---------------
data3_path = os.path.join(
    script_dir,
    "..",
    "Data",
    "mystery_virus_daily_active_counts_RELEASE#3.csv"
)
data3 = pd.read_csv(data3_path, parse_dates=['date'])
exp_data3 = data3[(data3['day'] >= 20) & (data3['day'] <= 100)]

# (after you have drawn the SEIR curve and release‑#2 points:)
plt.scatter(exp_data3['day'],
            exp_data3['active reported daily cases'],
            color='green', marker='x',
            label='Observed data (release #3)')
plt.legend()          # update legend to include #3
plt.show()