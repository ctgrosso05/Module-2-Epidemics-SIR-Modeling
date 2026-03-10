import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to Data folder
data_path = os.path.join(script_dir, "..", "Data",
                         "mystery_virus_daily_active_counts_RELEASE#1.csv")

data = pd.read_csv(data_path, parse_dates=['date'])

# Select exponential window for R0 calculation
exp_data = data[(data['day'] >= 20) & (data['day'] <= 40)]

log_I = np.log(exp_data['active reported daily cases'])

slope, intercept = np.polyfit(exp_data['day'], log_I, 1)

r = slope
D = 5
R0 = 1 + r * D

# Generate fitted exponential curve
fitted_days = np.linspace(20, 40, 100)
fitted_I = np.exp(intercept + slope * fitted_days)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(data["day"], data["active reported daily cases"], 'o-', label='Actual Data', linewidth=2)
plt.plot(fitted_days, fitted_I, 'r--', label=f'Exponential Fit (R0={R0:.2f})', linewidth=2)
plt.xlabel("Day")
plt.ylabel("Active Infections")
plt.title("Day vs Active Infections (Data Release #1)")
plt.legend()
plt.grid(True, alpha=0.3)

print("Estimated growth rate r:", r)
print("Estimated R0:", R0)

plt.show()
