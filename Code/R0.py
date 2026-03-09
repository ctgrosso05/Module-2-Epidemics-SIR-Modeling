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
    "mystery_virus_daily_active_counts_RELEASE#1.csv"
)

data = pd.read_csv(data_path, parse_dates=['date'])

# Select exponential window
exp_data = data[(data['day'] >= 20) & (data['day'] <= 40)]

log_I = np.log(exp_data['active reported daily cases'])

slope, intercept = np.polyfit(exp_data['day'], log_I, 1)

r = slope
D = 5
R0 = 1 + r * D

print("Estimated growth rate r:", r)
print("Estimated R0:", R0)

# Implementing Euler's method for SEIR model

# Parameters (guessed values)
beta = 0.357  # infection rate
sigma = 0.243  # incubation rate (1/sigma = incubation period)
gamma = 0.107  # recovery rate (1/gamma = infectious period)

# Initial conditions (assuming total population N=10000, initial S, E, I, R)
N = 10000
S0 = N - 1
E0 = 0
I0 = 1  # start with 1 infected
R0 = 0

# Time steps
t_start = 0
t_end = 50  # days
dt = 0.1
t = np.arange(t_start, t_end, dt)

# Arrays to store values
S = np.zeros(len(t))
E = np.zeros(len(t))
I = np.zeros(len(t))
R = np.zeros(len(t))

S[0] = S0
E[0] = E0
I[0] = I0
R[0] = R0

# Euler's method
for i in range(1, len(t)):
    dS = -beta * S[i-1] * I[i-1] / N
    dE = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
    dI = sigma * E[i-1] - gamma * I[i-1]
    dR = gamma * I[i-1]
    
    S[i] = S[i-1] + dt * dS
    E[i] = E[i-1] + dt * dE
    I[i] = I[i-1] + dt * dI
    R[i] = R[i-1] + dt * dR

# Interpolate simulated I to match data days
data_days = exp_data['day'].values
sim_I_at_data_days = np.interp(data_days, t, I)

# Calculate SSE between simulated I and observed data
observed_I = exp_data['active reported daily cases'].values
SSE = np.sum((sim_I_at_data_days - observed_I)**2)
print(f"SSE with guessed parameters: {SSE}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, I, label='Simulated I(t) - Euler SEIR')
plt.scatter(exp_data['day'], exp_data['active reported daily cases'], color='red', label='Observed Data')
plt.xlabel('Day')
plt.ylabel('Infected')
plt.title('SEIR Model vs Data')
plt.legend()
plt.show()

