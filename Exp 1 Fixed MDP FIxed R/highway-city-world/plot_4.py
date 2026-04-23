import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import math

# Load the .npy files
qm_sqfl = np.load(r"C:\Users\kvora1\Downloads\highway-city-world\memorized rewards\test_combined.npy")
sqfl = np.load("SFQL_og.npy")
ql = np.load("QL_traditonal.npy")
qm = np.load(r"C:\Users\kvora1\Downloads\highway-city-world\memorized rewards\QM+SFQL\QM+SFQL_ours_MR_learning.npy")

# For simplicity, we assume each file is already formatted correctly
# If not, you might need to preprocess these files into a compatible format.

# Define subplot parameters
subplot_params = [
    {
        "w3": 200,  # window size for smoothing
        "length": 2700  # ensuring the same length for all data
    }
]

# Initialize seaborn style
sns.set_style("darkgrid")

# Create subplots dynamically
num_cols = 1
num_rows = 1
fig, axs = plt.subplots(math.ceil(num_rows / num_cols), num_cols, figsize=(8, 5))

# Ensure axs is always a list
if not isinstance(axs, np.ndarray):
    axs = [axs]

# Track colors assigned to each label
color_map = {
    'QM+SFQL': 'red',
    'QM': plt.cm.tab10(0),
    'SFQL': plt.cm.tab10(2),
    'QL': plt.cm.tab10(1)
}

# Prepare data files as a list
data_files = [
    ('QM+SFQL', qm_sqfl),
    ('QM', qm),
    ('SFQL', sqfl),
    ('QL', ql)
]

# Function to plot data for a single subplot
def plot_subplot(ax, data_files, params, color_map):
    i = 0
    for label, data in data_files:
        d = np.mean(data, axis=0).flatten()[:params["length"]]
        # Smooth the data using a moving average (convolution)
        smoothed_data = np.convolve(d, np.ones(params["w3"]), 'valid') / params["w3"]
        
        # x-axis represents the steps
        x_axis = np.arange(smoothed_data.shape[0]) * 2
        
        # Plot the data
        ax.plot(x_axis, smoothed_data, label=label, color=color_map[label])
        ax.fill_between(x_axis, smoothed_data - np.std(d), smoothed_data + np.std(data), 
                        alpha=0.25, color=color_map[label])  # Show std deviation
        
        i += 1
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

# Plot data for the subplot
for ax in axs:
    plot_subplot(ax, data_files, subplot_params[0], color_map)

# Add legend
plt.figlegend(
    handles=[
        plt.Line2D([0], [0], linestyle='-', color=color_map['QM+SFQL'], label='Q-M+SFQL'),
        plt.Line2D([0], [0], linestyle='-', color=color_map['QM'], label='Q-M'),
        plt.Line2D([0], [0], linestyle='-', color=color_map['SFQL'], label='SFQL'),
        plt.Line2D([0], [0], linestyle='-', color=color_map['QL'], label='QL')
    ],
    labels=['Q-M+SFQL','Q-M', 'SFQL', 'QL'],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.08),
    ncol=4,
    labelspacing=0.
)

# Set common x and y labels for the entire plot
fig.text(0.5, 0.014, 'Step', ha='center', va='center')
fig.text(0.014, 0.5, 'Average Return', ha='center', va='center', rotation='vertical')
plt.rcParams['font.size'] = '20'

plt.tight_layout()
# plt.savefig('HC_memoriezdR_learning.png', bbox_inches='tight', dpi=600)
plt.show()
