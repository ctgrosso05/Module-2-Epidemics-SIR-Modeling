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
# intervention: vaccine event on day 70 (2000 people, 90% efficacy)
vaccine_day = 70
vaccine_total = 2000
vaccine_efficacy = 0.9
vaccinated_protected = vaccine_total * vaccine_efficacy
vaccinated_unprotected = vaccine_total * (1 - vaccine_efficacy)

# integrate with vaccination event
S_vax = np.zeros_like(t)
E_vax = np.zeros_like(t)
I_vax = np.zeros_like(t)
R_vax = np.zeros_like(t)
S_vax[0], E_vax[0], I_vax[0], R_vax[0] = S0, E0, I0, R0

for i in range(1, len(t)):
    # standard SEIR dynamics
    dS = -best_beta * S_vax[i-1] * I_vax[i-1] / N
    dE = best_beta * S_vax[i-1] * I_vax[i-1] / N - best_sigma * E_vax[i-1]
    dI = best_sigma * E_vax[i-1] - best_gamma * I_vax[i-1]
    dR = best_gamma * I_vax[i-1]

    S_vax[i] = S_vax[i-1] + dt * dS
    E_vax[i] = E_vax[i-1] + dt * dE
    I_vax[i] = I_vax[i-1] + dt * dI
    R_vax[i] = R_vax[i-1] + dt * dR

    # apply vaccination at the specified day
    current_day = t[i]
    if abs(current_day - vaccine_day) < dt:
        # move successfully vaccinated people to R
        S_vax[i] -= vaccine_total
        # 90% become immune
        R_vax[i] += vaccinated_protected
        # 10% stay in S
        S_vax[i] += vaccinated_unprotected
        print(f"Vaccine event on day {vaccine_day}: {vaccine_total} vaccinated")
        print(f"  → {vaccinated_protected:.0f} protected (90%)")
        print(f"  → {vaccinated_unprotected:.0f} unprotected (10%)")

peak_I_vax = I_vax.max()
peak_day_vax = t[np.argmax(I_vax)]
print(f"With vaccine event: {peak_I_vax:.0f} infected on day {peak_day_vax:.1f}")

# ----------------------------------------------------------------------
# plotting
plt.figure(figsize=(10, 6))
plt.plot(t, I_base, label='Baseline VT outbreak (β=0.5)')
plt.plot(t, I_vax, '--', label='Vaccine event day 70 (2000 people, 90% efficacy)')
plt.xlabel('Day')
plt.ylabel('Number infected')
plt.title('SEIR simulation – VT population of 38 000')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()