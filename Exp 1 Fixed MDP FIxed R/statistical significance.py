import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# Define base directories for each row of the 3x3 grid
base_dirs = [
    'Dollar-Euro',  # Path to first folder
    'Racetrack'   # Path to third folder
]

# List of files for each subplot in a 3x3 grid
subplot_files = [
    [['ours_0_1.csv', 'QL_0_1.csv', 'SFQL_0_1.csv'],
     ['ours_0_2.csv', 'QL_0_2.csv', 'SFQL_0_2.csv'],
     ['ours_0_3.csv', 'QL_0_4.csv', 'SFQL_0_4.csv']],
    
    [['ours_0_1.csv', 'QL_0_1.csv', 'SFQL_0_1.csv'],
     ['ours_0_5.csv', 'QL_0_5.csv', 'SFQL_0_5.csv'],
     ['ours_0_7.csv', 'QL_0_7.csv', 'SFQL_0_7.csv']],
]

# Define subplot parameters for each domain and method (3 domains, each with 3 different sets of params)
subplot_params = [
    [ {"w3": 150, "length": 1000}, {"w3": 150, "length": 1000}, {"w3": 150, "length": 1000} ],  # Domain 1
    [ {"w3": 150, "length": 1300}, {"w3": 150, "length": 1300}, {"w3": 150, "length": 1300} ]  # Domain 3
]


# Function to perform Wilcoxon test on data for a single set of methods
def perform_wilcoxon_test(files, base_dir, params):
    rewards = {}
    
    # Read and process data for each file
    for file in files:
        file_path = os.path.join(base_dir, file)  # Get full path
        data = pd.read_csv(file_path).drop("Unnamed: 0", axis=1)
        all_runs_rewards = np.array(data)[:, :params["length"]]
        mean_rewards = np.mean(all_runs_rewards, axis=0)
        
        # Store rewards for statistical comparison
        label = file.split('_')[0]
        rewards[label] = mean_rewards

    # Perform Wilcoxon tests between methods (if there are at least two methods)
    if 'ours' in rewards and 'QL' in rewards and 'SFQL' in rewards:
        print(f"Wilcoxon test results for files: {files}")
        
        # Compare Q-M (ours) with QL
        stat, p_value = wilcoxon(rewards['ours'], rewards['QL'])
        print(f"  Q-M vs QL: p-value = {p_value:.4f}")
        
        # Compare Q-M (ours) with SFQL
        stat, p_value = wilcoxon(rewards['ours'], rewards['SFQL'])
        print(f"  Q-M vs SFQL: p-value = {p_value:.4f}")
        
        # Compare QL with SFQL
        stat, p_value = wilcoxon(rewards['QL'], rewards['SFQL'])
        print(f"  QL vs SFQL: p-value = {p_value:.4f}")

# Iterate through each set of files and perform the Wilcoxon test
for col in range(2):  # Iterate through different domains
    for row in range(3):  # Iterate through different sets of files within each domain
        perform_wilcoxon_test(subplot_files[col][row], base_dirs[col], subplot_params[col][row])
