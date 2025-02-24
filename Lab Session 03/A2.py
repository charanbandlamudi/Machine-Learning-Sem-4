import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

# Upload the Excel file
uploaded = files.upload()

# Load the Excel file
file_name = list(uploaded.keys())[0] # Get the uploaded file name
xls = pd.ExcelFile(file_name)

# Load the "IRCTC Stock Price" sheet
df_stock = pd.read_excel(xls, sheet_name="IRCTC Stock Price")

# Convert 'Price' column to numeric
df_stock["Price"] = pd.to_numeric(df_stock["Price"], errors="coerce")

# Drop NaN values
df_stock = df_stock.dropna(subset=["Price"])

# Compute histogram
hist, bins = np.histogram(df_stock["Price"], bins=10)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(df_stock["Price"], bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Stock Price")
plt.ylabel("Frequency")
plt.title("Histogram of IRCTC Stock Prices")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Calculate mean and variance
mean_price = df_stock["Price"].mean()
variance_price = df_stock["Price"].var()

print(f"Mean Stock Price: {mean_price}")
print(f"Variance: {variance_price}")
