import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from google.colab import files

file = pd.read_excel(r"/content/Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")


numeric_cols = thyroid_data.select_dtypes(include=["int64", "float64"]).columns

plt.figure(figsize=(12, 6))
thyroid_data[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("Before Normalization: Distribution of Numeric Attributes")
plt.show()

minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

normalized_data = thyroid_data.copy()

for col in numeric_cols:
    if thyroid_data[col].skew() < 1:  # Normal distribution → Standard Scaling
        normalized_data[col] = standard_scaler.fit_transform(thyroid_data[[col]])
    else:  # Skewed distribution → Min-Max Scaling
        normalized_data[col] = minmax_scaler.fit_transform(thyroid_data[[col]])

print("\nNormalization Completed!")

plt.figure(figsize=(12, 6))
normalized_data[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("After Normalization: Distribution of Numeric Attributes")
plt.show()



