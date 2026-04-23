import os
import pandas as pd
import numpy as np

def calculate_average_csv_files(file_list):
    """Calculates the average of multiple CSV files.

    Args:
        file_list: A list of CSV file paths.

    Returns:
        A pandas DataFrame containing the average of the CSV files.
    """
    dataframes = []
    for file_path in file_list:
        df = pd.read_csv(file_path)
        dataframes.append(df)

    average_df = pd.concat(dataframes)  # Concatenate all dataframes

    return average_df

# Define the number of files (avg = 30)
avg = 30

# Define the noise levels as a list
noise_levels = [0] #[0, 0.015, 0.02, 0.025, 0.03]

# Create lists of CSV file paths dynamically for each noise level
ours_files = {
    f'QL_0_{noise}': [f'{i}//QL_0_{noise}.csv' for i in range(avg)] 
    for noise in noise_levels
}

# Calculate averages for each group of files dynamically
averages = {}
for noise, file_list in ours_files.items():
    averages[noise] = calculate_average_csv_files(file_list)

# Save the average DataFrames to CSV files (optional)
for noise, avg_df in averages.items():
    avg_filename = f"{noise}.csv"
    avg_df.to_csv(avg_filename, index=False)

print("Averages calculated and saved for each noise level.")
