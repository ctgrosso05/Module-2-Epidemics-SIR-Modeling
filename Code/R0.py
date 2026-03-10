import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get directory where THIS script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build full path to CSV (release #2)
data_path = os.path.join(
    script_dir,
    "..",
    "Data",
    "mystery_virus_daily_active_counts_RELEASE#2.csv"
)

data = pd.read_csv(data_path, parse_dates=['date'])

# Select exponential window
exp_data = data[(data['day'] >= 20) & (data['day'] <= 40)]

log_I = np.log(exp_data['active reported daily cases'])
slope, intercept = np.polyfit(exp_data['day'], log_I, 1)

r = slope
D = 5
R0_est = 1 + r * D                     # rename so we can still use R0 later

print("Estimated growth rate r:", r)
print("Estimated R0:", R0_est)

# ----------------------------------------------------------------------
# parameters and initial conditions (will be adjusted by the fitter)
beta = 0.3      # infection rate
sigma = 0.2     # incubation rate
gamma = 0.1     # recovery rate

N = 10000
S0 = N - 1
E0 = 0
I0 = 1          # start with one infectious
R0 = 0          # initial recovered

# time grid for the solver
t_start = 0.0
t_end = 100.0
dt = 0.1
t = np.arange(t_start, t_end + dt/2, dt)   # include t_end

# ----------------------------------------------------------------------
# Step 1: Euler's method for SEIR
def euler_seir(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    """
    return lists (S,E,I,R) evaluated at the supplied
    *uniformly spaced* timepoints using forward-Euler.
    """
    dt_local = timepoints[1] - timepoints[0]
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    for _ in range(1, len(timepoints)):
        dS = -beta * S[-1] * I[-1] / N
        dE = beta * S[-1] * I[-1] / N - sigma * E[-1]
        dI = sigma * E[-1] - gamma * I[-1]
        dR = gamma * I[-1]
        S.append(S[-1] + dt_local * dS)
        E.append(E[-1] + dt_local * dE)
        I.append(I[-1] + dt_local * dI)
        R.append(R[-1] + dt_local * dR)
    return S, E, I, R

# Step 2: grid search to minimise SSE against exp_data
def fit_seir_parameters(timepoints, N, S0, E0, I0, R0, data):
    beta_range = np.linspace(0.1, 1.0, 10)
    sigma_range = np.linspace(0.1, 0.5, 10)
    gamma_range = np.linspace(0.05, 0.3, 10)

    SSE_array = []
    params = []

    for b in beta_range:
        for s in sigma_range:
            for g in gamma_range:
                _, _, I_sim, _ = euler_seir(b, s, g, S0, E0, I0, R0, timepoints, N)
                # interpolate to the days present in the data
                days = data['day'].values
                I_at_data = np.interp(days, timepoints, I_sim)
                obs = data['active reported daily cases'].values
                SSE = np.sum((I_at_data - obs) ** 2)
                SSE_array.append(SSE)
                params.append((b, s, g))

    idx = np.argmin(SSE_array)
    return params[idx] + (SSE_array[idx],)

best_beta, best_sigma, best_gamma, best_SSE = fit_seir_parameters(
    t, N, S0, E0, I0, R0, exp_data
)

print(f"Best parameters: beta={best_beta}, sigma={best_sigma}, "
      f"gamma={best_gamma}, SSE={best_SSE}")

# Step 3: run out longer and locate the peak
t_extended = np.arange(t_start, 200.0 + dt/2, dt)
S_ext, E_ext, I_ext, R_ext = euler_seir(
    best_beta, best_sigma, best_gamma, S0, E0, I0, R0, t_extended, N
)

peak_I = np.max(I_ext)
peak_day = t_extended[int(np.argmax(I_ext))]

print(f"Peak infected: {peak_I}")
print(f"(~{peak_I/N*100:.1f}% of the population)")
print(f"Peak occurs on day: {peak_day}")

# reproducible plot against the original data
S, E, I, R = euler_seir(
    best_beta, best_sigma, best_gamma, S0, E0, I0, R0, t, N
)

days = exp_data['day'].values
sim_I_at_data_days = np.interp(days, t, I)
observed_I = exp_data['active reported daily cases'].values
SSE = np.sum((sim_I_at_data_days - observed_I) ** 2)
print(f"SSE with fitted parameters (recomputed): {SSE}")

plt.figure(figsize=(10, 6))
plt.plot(t, I, label='Simulated I(t) – fitted SEIR')
plt.scatter(days, observed_I, color='red', label='Observed data')
plt.xlabel('Day')
plt.ylabel('Infected')
plt.title('SEIR model vs. data (release #2)')
plt.legend()
plt.show()