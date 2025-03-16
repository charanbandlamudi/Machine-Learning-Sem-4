
import numpy as np
import pandas as pd

# List of file paths
file_paths = ['255_s.xlsx', '259_s.xlsx', '261_s.xlsx', '285_s.xlsx', '287_s.xlsx',
              '219_student.xlsx', '220_student.xlsx', '221_student.xlsx', '222_student.xlsx', '223_student.xlsx']

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each file path, read the Excel file, and append to the list
for file_path in file_paths:
    try:
        df = pd.read_excel(file_path)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Function to calculate Gini index
def gini_index(column):
    values, counts = np.unique(column, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

# Specify the target column
target_column = 'Number'

# Calculate and print Gini index
gini_value = gini_index(combined_df[target_column])
print(f"Gini Index of '{target_column}': {gini_value}")
