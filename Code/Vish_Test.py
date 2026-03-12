import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# model parameters (given)
best_beta = 0.5
best_sigma = 0.1
best_gamma = 0.07777777777777778
best_SSE = 104620.93458660078           # provided for reference

# initial population
S0 = 38000
E0 = 0
I0 = 1
R0 = 0
N = S0 + E0 + I0 + R0

# time grid
t_start = 0
t_end = 200      # days
dt = 0.1
t = np.arange(t_start, t_end + dt, dt)

# ----------------------------------------------------------------------
def euler_seir(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    """Forward‑Euler integrator for the SEIR equations."""
    dt = timepoints[1] - timepoints[0]
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
    return np.array(S), np.array(E), np.array(I), np.array(R)

# ----------------------------------------------------------------------
# baseline outbreak (no intervention)
S_base, E_base, I_base, R_base = euler_seir(
    best_beta, best_sigma, best_gamma, S0, E0, I0, R0, t, N
)
peak_I_base = I_base.max()
peak_day_base = t[np.argmax(I_base)]
print(f"Baseline peak: {peak_I_base:.0f} infected on day {peak_day_base:.1f}")

# ----------------------------------------------------------------------
# intervention: 14‑day school closure starting day 70
closure_start = 70
closure_end   = closure_start + 14   # day 84

# build β profile: normal before/after, 20% during closure
beta_profile = np.ones_like(t) * best_beta
mask = (t >= closure_start) & (t < closure_end)
beta_profile[mask] *= 0.2   # only 20% of normal contacts

# integrate with time‑varying β
S_close = np.zeros_like(t)
E_close = np.zeros_like(t)
I_close = np.zeros_like(t)
R_close = np.zeros_like(t)
S_close[0], E_close[0], I_close[0], R_close[0] = S0, E0, I0, R0

for i in range(1, len(t)):
    b = beta_profile[i-1]
    dS = -b * S_close[i-1] * I_close[i-1] / N
    dE = b * S_close[i-1] * I_close[i-1] / N - best_sigma * E_close[i-1]
    dI = best_sigma * E_close[i-1] - best_gamma * I_close[i-1]
    dR = best_gamma * I_close[i-1]
    S_close[i] = S_close[i-1] + dt * dS
    E_close[i] = E_close[i-1] + dt * dE
    I_close[i] = I_close[i-1] + dt * dI
    R_close[i] = R_close[i-1] + dt * dR

peak_I_close = I_close.max()
peak_day_close = t[np.argmax(I_close)]
print(f"With school closure: {peak_I_close:.0f} infected on day {peak_day_close:.1f}")

# ----------------------------------------------------------------------
# plotting
plt.figure(figsize=(10, 6))
plt.plot(t, I_base, label='Baseline VT outbreak (β=0.5)')
plt.plot(t, I_close, '--', label='School closure day 70–84 (β=0.2×)')
plt.xlabel('Day')
plt.ylabel('Number infected')
plt.title('SEIR simulation – VT population of 38 000')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()