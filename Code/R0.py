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