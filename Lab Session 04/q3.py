# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Upload the Excel file
from google.colab import files
uploaded = files.upload()

# Read the Excel file
file_name = list(uploaded.keys())[0]
df = pd.ExcelFile(file_name)

# Load the 'IRCTC Stock Price' sheet
df_irctc = pd.read_excel(file_name, sheet_name='IRCTC Stock Price')

# Prepare the data
# Using 'Open' as X and 'Price' as Y for scatter plot
df_irctc = df_irctc[['Open', 'Price']].dropna()

# Convert columns to numeric
df_irctc['Open'] = pd.to_numeric(df_irctc['Open'], errors='coerce')
df_irctc['Price'] = pd.to_numeric(df_irctc['Price'], errors='coerce')

# Drop rows with NaN values
df_irctc = df_irctc.dropna()

# Take first 20 data points for visualization
X = df_irctc['Open'].values[:20]
Y = df_irctc['Price'].values[:20]

# Assign classes based on price movement
# Class 0 (Blue) -> Price < Open (Stock went down)
# Class 1 (Red) -> Price >= Open (Stock went up or stayed the same)
classes = np.where(Y < X, 0, 1)

# Plot the data points with color coding
plt.figure(figsize=(8, 6))
for i in range(20):
    if classes[i] == 0:
        plt.scatter(X[i], Y[i], color='blue', label='Class 0 (Down - Blue)' if i == 0 else "")
    else:
        plt.scatter(X[i], Y[i], color='red', label='Class 1 (Up - Red)' if i == 0 else "")

# Add labels and legend
plt.title("IRCTC Stock Price Movement (Class 0 - Blue, Class 1 - Red)")
plt.xlabel("Open Price")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.show()
