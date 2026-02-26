import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to Data folder
data_path = os.path.join(script_dir, "..", "Data",
                         "mystery_virus_daily_active_counts_RELEASE#1.csv")

data = pd.read_csv(data_path, parse_dates=['date'])

plt.figure()
plt.plot(data["day"], data["active reported daily cases"])
plt.xlabel("Day")
plt.ylabel("Active Infections")
plt.title("Day vs Active Infections (Data Release #1)")
plt.show()