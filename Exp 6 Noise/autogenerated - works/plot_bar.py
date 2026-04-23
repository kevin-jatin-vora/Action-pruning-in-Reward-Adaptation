import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
# df = pd.read_csv("Data_RA_29.csv")  # Replace with your actual filename
import glob
import re

# Pattern to match all CSV files in the current directory
all_files = glob.glob("*.csv")

# Filter files that end with the specified numbers before the '.csv'
target_files = [f for f in all_files if re.search(r'_(4|9|14|19|24|29)\.csv$', f)]

# Concatenate all selected CSVs
df_list = [pd.read_csv(file) for file in target_files]
df = pd.concat(df_list, ignore_index=True)
# Clean domain names for x-axis labels
df['DomainLabel'] = 'R +/- ' + df['Domain'].apply(lambda x: x.split('_')[-1].replace('.csv', ''))

# Group by DomainLabel and compute average Actions Pruned
avg_pruned = df.groupby('DomainLabel')['Actions Pruned'].mean().reset_index()

# Sort for better visualization
avg_pruned = avg_pruned.sort_values(by='Actions Pruned', ascending=False)

# Plot
plt.figure(figsize=(5, 5))
bars = plt.bar(avg_pruned['DomainLabel'], avg_pruned['Actions Pruned'], color='slateblue', width=0.3)

# Add value labels (counts) on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom', fontsize=9)

# Labels and formatting
plt.xlabel('Noise')
plt.ylabel('Average Actions Pruned')
plt.title('Average Actions Pruned')
plt.xticks(ha='center')
plt.tight_layout()
plt.savefig("avg_actions_pruned_per_domain.jpg", dpi=600)

plt.show()
