import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from scipy import stats
import seaborn as sns
sns.set_style("darkgrid")

# ----------------------
# Load and prepare data
# ----------------------
# Get all CSV files matching the pattern
all_files = glob.glob("*.csv")
target_files = [f for f in all_files if re.search(r'_(9|19|29)\.csv$', f)]

# Read and combine all selected CSV files into a single DataFrame
df_list = [pd.read_csv(file) for file in target_files]
df = pd.concat(df_list, ignore_index=True)

# Compute the 'Total' (|S| * |A|) and 'Actions Pruned %' for percentage calculation
df['Total'] = df['|S|'] * df['|A|']
df['Actions Pruned %'] = (df['Actions Pruned'] / df['Total']) * 100

# Add readable domain label as percentage of noise
df['DomainLabel'] = df['Domain'].apply(lambda x: f"{(float(x.split('_')[1]) / 1.2) * 100:.2f}%")
# Remove 0.00% noise labels
df = df[df['DomainLabel'] != '0.00%'].reset_index(drop=True)

# Group by 'DomainLabel' and calculate the average 'Actions Pruned %' for each domain
avg_pruned = df.groupby('DomainLabel')['Actions Pruned %'].mean().reset_index()

# Sort the data by 'Actions Pruned %' in descending order
avg_pruned = avg_pruned.sort_values(by='Actions Pruned %', ascending=False)
# avg_pruned = avg_pruned[avg_pruned['Actions Pruned %'] > 0].reset_index(drop=True)


# ----------------------
# Reward plot setup
# ----------------------
# List of reward files for plotting
reward_files = [f'ours_0_{0}.csv', f'ours_0_{0.006}.csv', f'ours_0_{0.012}.csv', f'ours_0_{0.018}.csv', f'ours_0_{0.024}.csv', f'ours_0_{0.03}.csv']
reward_files = reward_files[1:]
params = {"w3": 1, "length": 170}
colors = ['#000000', '#191970', '#6A5ACD', '#4169E1', '#32CD32', '#FF6347']  # Colors for bars and lines
colors=colors[1:]
# Get the first 5 'Actions Pruned %' values for labeling
pruned = avg_pruned['Actions Pruned %'].iloc[:len(reward_files)].tolist()

# Function to compute the confidence interval
def confidence_interval(data, confidence=0.9):
    n = len(data)
    if n < 2:
        return np.zeros_like(data)
    se = stats.sem(data, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

# ----------------------
# Combined Plot
# ----------------------
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# --- Left: Bar chart ---
bars = axs[0].bar(avg_pruned['DomainLabel'], avg_pruned['Actions Pruned %'], color=colors[:len(avg_pruned)], width=0.3)

# Annotate bars with the values
for i, bar in enumerate(bars):
    yval = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontsize=18)

# Set labels and ticks for the left plot
axs[0].set_xlabel('+/- Percentage Noise wrt Rmax=1.2', fontsize=18)
axs[0].set_ylabel('Average Actions Pruned (%)', fontsize=18)
axs[0].tick_params(axis='x', rotation=0)
axs[0].tick_params(axis='both', labelsize=18)

# --- Right: Reward curves with confidence intervals ---
for idx, file in enumerate(reward_files):
    # Load the reward data
    data = pd.read_csv(file).drop("Unnamed: 0", axis=1)
    rewards_array = np.array(data)[:, :params["length"]]  # Only consider the first 'length' columns

    # Calculate mean rewards and confidence intervals
    mean_rewards = np.mean(rewards_array, axis=0)
    ci = confidence_interval(rewards_array)

    # Apply smoothing to the mean rewards and confidence intervals
    smoothed_mean = np.convolve(mean_rewards, np.ones(params["w3"]), 'valid') / params["w3"]
    smoothed_upper = np.convolve(mean_rewards + ci, np.ones(params["w3"]), 'valid') / params["w3"]
    smoothed_lower = np.convolve(mean_rewards - ci, np.ones(params["w3"]), 'valid') / params["w3"]

    # Label with corresponding pruned values
    label = f"Actions pruned = {np.round(pruned[idx], 2)}%"
    x_axis = np.arange(smoothed_mean.shape[0]) * 40  # Steps

    # Plot the reward curve with confidence intervals
    axs[1].plot(x_axis, smoothed_mean, label=label, color=colors[idx])
    axs[1].fill_between(x_axis, smoothed_lower, smoothed_upper, alpha=0.2, color=colors[idx])

# --- Additional QL baseline curve ---
ql_file = "QL_0_0.csv"
ql_data = pd.read_csv(ql_file).drop("Unnamed: 0", axis=1)
ql_array = np.array(ql_data)[:, :params["length"]]

# Compute mean and confidence intervals
ql_mean = np.mean(ql_array, axis=0)
ql_ci = confidence_interval(ql_array)

# Apply smoothing
smoothed_ql_mean = np.convolve(ql_mean, np.ones(params["w3"]), 'valid') / params["w3"]
smoothed_ql_upper = np.convolve(ql_mean + ql_ci, np.ones(params["w3"]), 'valid') / params["w3"]
smoothed_ql_lower = np.convolve(ql_mean - ql_ci, np.ones(params["w3"]), 'valid') / params["w3"]

# Plot the QL curve
x_axis = np.arange(smoothed_ql_mean.shape[0]) * 40  # Steps
axs[1].plot(x_axis, smoothed_ql_mean, label="QL", color='#ff7f0e', linestyle='--')
axs[1].fill_between(x_axis, smoothed_ql_lower, smoothed_ql_upper, alpha=0.2, color='#ff7f0e')


# Set labels and ticks for the right plot
axs[1].set_xlabel("Step", fontsize=18)
axs[1].set_ylabel("Average Return", fontsize=18)
axs[1].legend(loc='lower right', ncol=1, fontsize=18)
axs[1].tick_params(axis='both', labelsize=18)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("combined_bar_reward_plot2.jpg", bbox_inches='tight', dpi=600)
plt.show()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import re
# from scipy import stats
# import seaborn as sns
# sns.set_style("darkgrid")

# # ----------------------
# # Load and prepare data
# # ----------------------
# # Get all CSV files matching the pattern
# all_files = glob.glob("*.csv")
# target_files = [f for f in all_files if re.search(r'_(9|19|29)\.csv$', f)]

# # Read and combine all selected CSV files into a single DataFrame
# df_list = [pd.read_csv(file) for file in target_files]
# df = pd.concat(df_list, ignore_index=True)

# # Compute the 'Total' (|S| * |A|) and 'Actions Pruned %' for percentage calculation
# df['Total'] = df['|S|'] * df['|A|']
# df['Actions Pruned %'] = (df['Actions Pruned'] / df['Total']) * 100

# # Add readable domain label as percentage of noise
# df['DomainLabel'] = df['Domain'].apply(lambda x: f"{(float(x.split('_')[1]) / 1.2) * 100:.2f}%")
# # Remove 0.00% noise labels
# df = df[df['DomainLabel'] != '0.00%'].reset_index(drop=True)

# # Group by 'DomainLabel' and calculate the average 'Actions Pruned %' for each domain
# avg_pruned = df.groupby('DomainLabel')['Actions Pruned %'].mean().reset_index()

# # Sort the data by 'Actions Pruned %' in descending order
# avg_pruned = avg_pruned.sort_values(by='Actions Pruned %', ascending=False)
# # avg_pruned = avg_pruned[avg_pruned['Actions Pruned %'] > 0].reset_index(drop=True)


# # ----------------------
# # Reward plot setup
# # ----------------------
# # List of reward files for plotting
# reward_files = [f'ours_0_{0}.csv', f'ours_0_{0.006}.csv', f'ours_0_{0.012}.csv', f'ours_0_{0.018}.csv', f'ours_0_{0.024}.csv', f'ours_0_{0.03}.csv']
# reward_files = reward_files[1:]
# params = {"w3": 1, "length": 90}
# colors = ['#000000', '#191970', '#6A5ACD', '#4169E1', '#32CD32', '#FF6347']  # Colors for bars and lines
# colors=colors[1:]
# # Get the first 5 'Actions Pruned %' values for labeling
# pruned = avg_pruned['Actions Pruned %'].iloc[:len(reward_files)].tolist()

# # Function to compute the confidence interval
# def confidence_interval(data, confidence=0.9):
#     n = len(data)
#     if n < 2:
#         return np.zeros_like(data)
#     se = stats.sem(data, axis=0)
#     h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
#     return h

# # ----------------------
# # Combined Plot
# # ----------------------
# fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# # --- Left: Bar chart ---
# bars = axs[0].bar(avg_pruned['DomainLabel'], avg_pruned['Actions Pruned %'], color=colors[:len(avg_pruned)], width=0.3)

# # Annotate bars with the values
# for i, bar in enumerate(bars):
#     yval = bar.get_height()
#     axs[0].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontsize=18)

# # Set labels and ticks for the left plot
# axs[0].set_xlabel('+/- Percentage Noise wrt Rmax=1.2', fontsize=18)
# axs[0].set_ylabel('Average Actions Pruned (%)', fontsize=18)
# axs[0].tick_params(axis='x', rotation=0)
# axs[0].tick_params(axis='both', labelsize=18)

# # --- Right: Reward curves with confidence intervals ---
# for idx, file in enumerate(reward_files):
#     # Load the reward data
#     data = pd.read_csv(file).drop("Unnamed: 0", axis=1)
#     rewards_array = np.array(data)[:, :params["length"]]  # Only consider the first 'length' columns

#     # Calculate mean rewards and confidence intervals
#     mean_rewards = np.mean(rewards_array, axis=0)
#     ci = confidence_interval(rewards_array)

#     # Apply smoothing to the mean rewards and confidence intervals
#     smoothed_mean = np.convolve(mean_rewards, np.ones(params["w3"]), 'valid') / params["w3"]
#     smoothed_upper = np.convolve(mean_rewards + ci, np.ones(params["w3"]), 'valid') / params["w3"]
#     smoothed_lower = np.convolve(mean_rewards - ci, np.ones(params["w3"]), 'valid') / params["w3"]

#     # Label with corresponding pruned values
#     label = f"Actions pruned = {np.round(pruned[idx], 2)}%"
#     x_axis = np.arange(smoothed_mean.shape[0]) * 40  # Steps

#     # Plot the reward curve with confidence intervals
#     axs[1].plot(x_axis, smoothed_mean, label=label, color=colors[idx])
#     axs[1].fill_between(x_axis, smoothed_lower, smoothed_upper, alpha=0.2, color=colors[idx])

# # Set labels and ticks for the right plot
# axs[1].set_xlabel("Step", fontsize=18)
# axs[1].set_ylabel("Average Return", fontsize=18)
# axs[1].legend(loc='lower right', ncol=1, fontsize=18)
# axs[1].tick_params(axis='both', labelsize=18)

# # Adjust layout and save the figure
# plt.tight_layout()
# plt.savefig("combined_bar_reward_plot2.jpg", bbox_inches='tight', dpi=600)
# plt.show()
