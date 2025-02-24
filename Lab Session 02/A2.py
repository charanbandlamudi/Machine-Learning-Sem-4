# Import necessary libraries
import pandas as pd
import numpy as np

# Load the Excel file (Upload manually in Colab and update file path)
from google.colab import files
uploaded = files.upload()

# Get the uploaded filename
file_name = list(uploaded.keys())[0]

# Load the Excel file
xls = pd.ExcelFile(file_name)

# Load the "Purchase data" sheet
purchase_data = pd.read_excel(xls, sheet_name="Purchase data")

# Selecting relevant columns and dropping empty ones
purchase_data = purchase_data.iloc[:, :5]  # First five columns are relevant
purchase_data = purchase_data.drop(columns=["Customer"])  # Remove Customer ID

# Rename columns for clarity
purchase_data.columns = ["Candies", "Mangoes", "Milk_Packets", "Payment"]

# Extract feature matrix (X) and target vector (Y)
X = purchase_data[["Candies", "Mangoes", "Milk_Packets"]].values
Y = purchase_data["Payment"].values

# Add a bias column (intercept term) to X
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Compute the model vector using the Moore-Penrose Pseudo-inverse
X_pseudo_inverse = np.linalg.pinv(X)  # Compute Pseudo-inverse
model_vector = X_pseudo_inverse @ Y  # Calculate model parameters

# Display the model vector
print("Model Vector (Intercept and Coefficients):")
print(model_vector)
