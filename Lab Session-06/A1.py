import numpy as np
import pandas as pd


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

# A1: Calculate entropy
def calculate_entropy(column):
    # Calculate the probability of each class
    probabilities = column.value_counts(normalize=True)
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Example usage: Calculating entropy for the 'Number' column (or any other categorical target)
target_column = 'Number'
entropy_value = calculate_entropy(combined_df[target_column])
print(f"Entropy of '{target_column}':", entropy_value)

