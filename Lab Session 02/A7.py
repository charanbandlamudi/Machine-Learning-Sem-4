# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from google.colab import files

# Step 1: Upload the Excel file
uploaded = files.upload()

# Get the uploaded filename
file_name = list(uploaded.keys())[0]

# Load the Excel file
xls = pd.ExcelFile(file_name)

# Load the "thyroid0387_UCI" sheet
thyroid_data = pd.read_excel(xls, sheet_name="thyroid0387_UCI")

# Identify numeric columns
numeric_cols = thyroid_data.select_dtypes(include=["int64", "float64"]).columns

# Step 2: Plot Before Normalization
plt.figure(figsize=(12, 6))
thyroid_data[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("Before Normalization: Distribution of Numeric Attributes")
plt.show()

# Initialize scalers
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Create a copy for normalization
normalized_data = thyroid_data.copy()

# Step 3: Apply Normalization (Min-Max for skewed, Z-score for normal)
for col in numeric_cols:
    if thyroid_data[col].skew() < 1:  # Normal distribution → Standard Scaling
        normalized_data[col] = standard_scaler.fit_transform(thyroid_data[[col]])
    else:  # Skewed distribution → Min-Max Scaling
        normalized_data[col] = minmax_scaler.fit_transform(thyroid_data[[col]])

print("\nNormalization Completed!")

# Step 4: Plot After Normalization
plt.figure(figsize=(12, 6))
normalized_data[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("After Normalization: Distribution of Numeric Attributes")
plt.show()



