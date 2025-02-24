# Import necessary libraries
import pandas as pd
import numpy as np
from google.colab import files

# Step 1: Upload the Excel file
uploaded = files.upload()

# Get the uploaded filename
file_name = list(uploaded.keys())[0]

# Load the Excel file
xls = pd.ExcelFile(file_name)

# Load the "thyroid0387_UCI" sheet
thyroid_data = pd.read_excel(xls, sheet_name="thyroid0387_UCI")

# Step 2: Extract First Two Observations
first_two_vectors = thyroid_data.iloc[:2]  # Select first two rows

# Identify binary columns (0/1 values) - Modified to handle mixed types
binary_cols = [
    col
    for col in thyroid_data.columns
    if thyroid_data[col].dropna().apply(lambda x: x in (0, 1, 0.0, 1.0)).all()
    and thyroid_data[col].dropna().nunique() <= 2  # Check for at most 2 unique values
]

# Extract binary attributes
binary_data = first_two_vectors[binary_cols]

# Convert to NumPy arrays for calculations
vector_1 = binary_data.iloc[0].values
vector_2 = binary_data.iloc[1].values

# Step 3: Compute Jaccard & Simple Matching Coefficient
f11 = np.sum((vector_1 == 1) & (vector_2 == 1))  # Both 1
f00 = np.sum((vector_1 == 0) & (vector_2 == 0))  # Both 0
f10 = np.sum((vector_1 == 1) & (vector_2 == 0))  # (1,0) mismatch
f01 = np.sum((vector_1 == 0) & (vector_2 == 1))  # (0,1) mismatch

# Compute JC & SMC
JC = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0  # Avoid division by zero
SMC = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

# Step 4: Print the Results
print("\nBinary Attributes Considered:", binary_cols)
print("\nJaccard Coefficient (JC):", round(JC, 4))
print("Simple Matching Coefficient (SMC):", round(SMC, 4))

# Step 5: Compare JC & SMC
if JC < SMC:
    print("\n SMC is higher than JC because it considers both matches (1,1 and 0,0).")
    print(" JC is useful when we only care about 1s (e.g., features presence).")
else:
    print("\n JC and SMC values are close, meaning the feature vectors are similar.")
