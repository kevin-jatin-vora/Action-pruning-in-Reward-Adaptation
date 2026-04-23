import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from scipy import stats
import seaborn as sns
import os

sns.set_style("darkgrid")

# ----------------------
# Load and prepare data
# ----------------------
all_files = glob.glob("*.csv")

# Original target files
target_files = [f for f in all_files if re.search(r'_(9|19|29)\.csv$', f)]

# New target files (with '_new' in the filename)
new_target_files = [f for f in all_files if re.search(r'_(9|19|29)\_new.csv$', f)]

# Read data from the original CSV files
df_list = [pd.read_csv(file) for file in target_files]
df = pd.concat(df_list, ignore_index=True)

df['Total'] = df['|S|'] * df['|A|']
df['Actions Pruned %'] = (df['Actions Pruned'] / df['Total']) * 100

df['DomainLabel'] = df['Domain'].apply(lambda x: f"{(float(x.split('_')[1]) / 1.2) * 100:.2f}%")
df = df[df['DomainLabel'] != '0.00%'].reset_index(drop=True)

avg_pruned = df.groupby('DomainLabel')['Actions Pruned %'].mean().reset_index()
avg_pruned = avg_pruned.sort_values(by='Actions Pruned %', ascending=False)

# Read data from the new CSV files
new_df_list = [pd.read_csv(file) for file in new_target_files]
new_df = pd.concat(new_df_list, ignore_index=True)

new_df['Total'] = new_df['|S|'] * new_df['|A|']
new_df['Actions Pruned %'] = (new_df['Actions Pruned'] / new_df['Total']) * 100

new_df['DomainLabel'] = new_df['Domain'].apply(lambda x: f"{(float(x.split('_')[1]) / 1.2) * 100:.2f}%")
new_df = new_df[new_df['DomainLabel'] != '0.00%'].reset_index(drop=True)

avg_pruned_new = new_df.groupby('DomainLabel')['Actions Pruned %'].mean().reset_index()
avg_pruned_new = avg_pruned_new.sort_values(by='Actions Pruned %', ascending=False)

# ----------------------
# Reward plot setup
# ----------------------
reward_files = [f'ours_0_{v}.csv' for v in [0, 0.003, 0.006,0.009, 0.012,0.015]][1:]  # excluding 0
reward_files_new = [f.replace('.csv', '_new_new.csv') for f in reward_files]  # New version

params = {"w3": 50, "length": 190}

# Base colors for original data (orange replaced with magenta)
colors_orig = [
    "#1f77b4",  # Blue
    "#e377c2",  # Magenta (replaces orange)
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b"   # Brown
]

# Lighter versions of the same colors for _new data (light orange replaced with light magenta)
colors_new = [
    "#7fa9d9",  # Medium Light Blue
    "#d377af",  # Medium Light Magenta (instead of very pale pink)
    "#6eb661",  # Medium Light Green
    "#e0726c",  # Medium Light Red
    "#a892bd",  # Medium Light Purple
    "#a07f70"   # Medium Light Brown
]


# Use pruned % from original and new datasets for legends in reward plots
pruned_orig = avg_pruned['Actions Pruned %'].iloc[:len(reward_files)].tolist()
pruned_new = avg_pruned_new['Actions Pruned %'].iloc[:len(reward_files_new)].tolist()

def confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return np.zeros_like(data)
    se = stats.sem(data, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

# ----------------------
# Combined Plot (3 subplots)
# ----------------------
# fig, axs = plt.subplots(1, 3, figsize=(22, 7))
fig, axs = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={'width_ratios': [1.5, 1, 1]})


# --- Left: Grouped Bar chart (Side-by-side) ---
labels = avg_pruned['DomainLabel'].tolist()
group_spacing = 1.8  # increase this number for more space
x = np.arange(len(labels)) * group_spacing
width = 0.7
delta = 0.035

# Align new data with the same domain labels
merged_df = pd.merge(avg_pruned, avg_pruned_new, on='DomainLabel', suffixes=('_orig', '_new'))

bars_orig = axs[0].bar(x - width/2 - delta, merged_df['Actions Pruned %_orig'], width,
                       label='Original', color=colors_orig[:len(merged_df)])

bars_new = axs[0].bar(x + width/2 + delta, merged_df['Actions Pruned %_new'], width,
                      label='New', color=colors_new[:len(merged_df)])

# Add bar labels for both sets, with vertical shift to prevent overlap
for bar in bars_orig:
    height = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width() / 2, height + 0.15, f'{height:.1f}%', 
                ha='center', va='bottom', fontsize=16)

for bar in bars_new:
    height = bar.get_height()
    # Shift label higher than original bars to avoid overlap
    axs[0].text(bar.get_x() + bar.get_width() / 2, height + 0.15, f'{height:.1f}%', 
                ha='center', va='bottom', fontsize=16)

axs[0].set_xticks(x)
axs[0].set_xticklabels(merged_df['DomainLabel'], fontsize=16)
axs[0].set_xlabel('+/- Percentage Noise wrt Rmax=1.2', fontsize=20)
axs[0].set_ylabel('Average Actions Pruned (%)', fontsize=20)
axs[0].tick_params(axis='both', labelsize=16)
axs[0].set_title("Actions Pruned with respect to uniform noise", fontsize=22)
# axs[0].legend(fontsize=16)

# --- Middle: Reward curves (original) ---
for idx, file in enumerate(reward_files):
    data = pd.read_csv(file).drop("Unnamed: 0", axis=1)
    rewards_array = np.array(data)[:, :params["length"]]
    mean_rewards = np.mean(rewards_array, axis=0)
    ci = confidence_interval(rewards_array)

    smoothed_mean = np.convolve(mean_rewards, np.ones(params["w3"]), 'valid') / params["w3"]
    smoothed_upper = np.convolve(mean_rewards + ci, np.ones(params["w3"]), 'valid') / params["w3"]
    smoothed_lower = np.convolve(mean_rewards - ci, np.ones(params["w3"]), 'valid') / params["w3"]
    label = f"Pruned = {pruned_orig[idx]:.1f}%"  # 1 decimal place here

    x_axis = np.arange(smoothed_mean.shape[0]) * 40
    axs[1].plot(x_axis, smoothed_mean, label=label, color=colors_orig[idx])
    axs[1].fill_between(x_axis, smoothed_lower, smoothed_upper, alpha=0.2, color=colors_orig[idx])

# --- Right: Reward curves (_new_new files) ---
for idx, file in enumerate(reward_files_new):
    if not os.path.exists(file):
        continue  # Skip missing files
    data = pd.read_csv(file).drop("Unnamed: 0", axis=1)
    rewards_array = np.array(data)[:, :params["length"]]
    mean_rewards = np.mean(rewards_array, axis=0)
    ci = confidence_interval(rewards_array)

    smoothed_mean = np.convolve(mean_rewards, np.ones(params["w3"]), 'valid') / params["w3"]
    smoothed_upper = np.convolve(mean_rewards + ci, np.ones(params["w3"]), 'valid') / params["w3"]
    smoothed_lower = np.convolve(mean_rewards - ci, np.ones(params["w3"]), 'valid') / params["w3"]
    label = f"Pruned = {pruned_new[idx]:.1f}%"  # 1 decimal place here

    x_axis = np.arange(smoothed_mean.shape[0]) * 40
    axs[2].plot(x_axis, smoothed_mean, label=label, color=colors_new[idx])
    axs[2].fill_between(x_axis, smoothed_lower, smoothed_upper, alpha=0.2, color=colors_new[idx])
    
# Add QL baseline to subplot 2
ql_file = "QL_0_0.csv"
if os.path.exists(ql_file):
    ql_data = pd.read_csv(ql_file).drop("Unnamed: 0", axis=1)
    ql_array = np.array(ql_data)[:, :params["length"]]
    ql_mean = np.mean(ql_array, axis=0)
    ql_ci = confidence_interval(ql_array)

    smoothed_ql_mean = np.convolve(ql_mean, np.ones(params["w3"]), 'valid') / params["w3"]
    smoothed_ql_upper = np.convolve(ql_mean + ql_ci, np.ones(params["w3"]), 'valid') / params["w3"]
    smoothed_ql_lower = np.convolve(ql_mean - ql_ci, np.ones(params["w3"]), 'valid') / params["w3"]

    axs[1].plot(x_axis, smoothed_ql_mean, label="QL", color='#ff7f0e')
    axs[1].fill_between(x_axis, smoothed_ql_lower, smoothed_ql_upper, alpha=0.2, color='#ff7f0e')

# Add QL baseline to subplot 3
if os.path.exists(ql_file):
    axs[2].plot(x_axis, smoothed_ql_mean, label="QL", color='#ff7f0e')
    axs[2].fill_between(x_axis, smoothed_ql_lower, smoothed_ql_upper, alpha=0.2, color='#ff7f0e')

axs[1].set_xlabel("Step", fontsize=20)
axs[1].set_ylabel("Average Return", fontsize=20)
axs[1].tick_params(axis='both', labelsize=20)
axs[1].legend(loc='lower right', fontsize=18)
axs[1].set_title("Action Pruning using M-Q-M", fontsize=22)

axs[2].set_xlabel("Step", fontsize=20)
axs[2].set_ylabel("Average Return", fontsize=20)
axs[2].tick_params(axis='both', labelsize=20)
axs[2].legend(loc='lower right', fontsize=18)
axs[2].set_title("Action Pruning using Q-M", fontsize=22)

# ----------------------
# Save and show
# ----------------------
plt.tight_layout()
plt.savefig("combined_bar_reward_plot2.jpg", bbox_inches='tight', dpi=600)
plt.show()
