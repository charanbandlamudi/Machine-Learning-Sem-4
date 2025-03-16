import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ✅ List of dataset file paths
file_paths = ['255_s.xlsx', '259_s.xlsx', '261_s.xlsx', '285_s.xlsx', '287_s.xlsx',
              '219_student.xlsx', '220_student.xlsx', '221_student.xlsx', '222_student.xlsx', '223_student.xlsx']

# ✅ Load and combine all datasets
df_list = [pd.read_excel(file, sheet_name='Sheet1', engine='openpyxl') for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

print("Datasets loaded and combined successfully.")
print(df.head())

# ✅ Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

print("\nCategorical columns encoded successfully.")

# ✅ Function to calculate entropy
def calculate_entropy(target):
    counts = target.value_counts()
    probabilities = counts / len(target)
    entropy = -sum(probabilities * np.log2(probabilities))
    return entropy

# ✅ Function to calculate Information Gain
def calculate_information_gain(df, feature, target):
    total_entropy = calculate_entropy(df[target])
    values = df[feature].unique()
    feature_entropy = 0
    for value in values:
        subset = df[df[feature] == value]
        feature_entropy += (len(subset) / len(df)) * calculate_entropy(subset[target])
    information_gain = total_entropy - feature_entropy
    return information_gain

# ✅ Example usage of calculate_information_gain
target_column = 'Number'      # Replace with your target column name
feature_column = 'Start time' # Replace with the feature column name

# Check if columns exist before calculating Information Gain
if target_column in df.columns and feature_column in df.columns:
    information_gain = calculate_information_gain(df, feature_column, target_column)
    print(f"\nInformation Gain for '{feature_column}' relative to '{target_column}': {information_gain}")
else:
    print("\nFeature or target column not found in the dataset.")

