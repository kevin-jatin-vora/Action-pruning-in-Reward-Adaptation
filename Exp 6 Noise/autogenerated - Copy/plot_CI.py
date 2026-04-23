# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 08:54:35 2025

@author: kevin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 2025

@author: trive
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_style("darkgrid")

# -------------------------------------------------------------------------
# Load all CSV files and extract pruned action averages for labeling
# -------------------------------------------------------------------------

# Match CSV files ending with _4, _9, _14, _19, _24, or _29.csv
all_files = glob.glob("*.csv")
target_files = [f for f in all_files if re.search(r'_(4|9|14|19|24|29)\.csv$', f)]

# Load and combine CSVs
df_list = [pd.read_csv(file) for file in target_files]
df = pd.concat(df_list, ignore_index=True)

# Extract domain suffix and compute average "Actions Pruned"
df['DomainLabel'] = df['Domain'].apply(lambda x: x.split('_')[-1].replace('.csv', ''))
avg_pruned = df.groupby('DomainLabel')['Actions Pruned'].mean().reset_index()
pruned = avg_pruned['Actions Pruned'].tolist()

# -------------------------------------------------------------------------
# Define plot parameters and files for a single subplot
# -------------------------------------------------------------------------

files = [f'ours_0_{0}.csv', f'ours_0_{0.015}.csv', f'ours_0_{0.03}.csv']
params = {"w3": 1, "length": 90}
colors = ['#191970', '#6A5ACD', '#4169E1']  # Custom color map

# -------------------------------------------------------------------------
# Confidence interval helper
# -------------------------------------------------------------------------

def confidence_interval(data, confidence=0.9):
    n = len(data)
    if n < 2:
        return np.zeros_like(data)
    se = stats.sem(data, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

# -------------------------------------------------------------------------
# Plot function: Single subplot with CI instead of std
# -------------------------------------------------------------------------

def plot_single_subplot(files, params, pruned_values):
    plt.figure(figsize=(7, 5))

    for idx, file in enumerate(files):
        data = pd.read_csv(file).drop("Unnamed: 0", axis=1)
        rewards_array = np.array(data)[:, :params["length"]]

        mean_rewards = np.mean(rewards_array, axis=0)
        ci = confidence_interval(rewards_array)

        smoothed_mean = np.convolve(mean_rewards, np.ones(params["w3"]), 'valid') / params["w3"]
        smoothed_upper = np.convolve(mean_rewards + ci, np.ones(params["w3"]), 'valid') / params["w3"]
        smoothed_lower = np.convolve(mean_rewards - ci, np.ones(params["w3"]), 'valid') / params["w3"]

        label = f"Actions pruned = {np.round(pruned_values[idx], 2)}"
        # x_axis = np.arange(len(smoothed_mean))
        x_axis = np.arange(smoothed_mean.shape[0]) * 40

        plt.plot(x_axis, smoothed_mean, label=label, color=colors[idx])
        plt.fill_between(x_axis, smoothed_lower, smoothed_upper, alpha=0.2, color=colors[idx])

    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Average Return", fontsize=14)
    # plt.title("Average Reward per Episode", fontsize=16)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig("noisyR_CI_plot.png", dpi=600)
    plt.show()

# -------------------------------------------------------------------------
# Execute plot
# -------------------------------------------------------------------------

plot_single_subplot(files, params, pruned)
