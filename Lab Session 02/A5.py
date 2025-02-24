# Import necessary libraries
import pandas as pd
from google.colab import files

# Upload the Excel file
uploaded = files.upload()

# Get the uploaded filename
file_name = list(uploaded.keys())[0]

# Load the Excel file
xls = pd.ExcelFile(file_name)

# Load the "thyroid0387_UCI" sheet
thyroid_data = pd.read_excel(xls, sheet_name="thyroid0387_UCI")

# Display the first few rows
print(thyroid_data.head())

# Display column names and data types
print("\nData Types of Each Column:\n", thyroid_data.dtypes)

# Identify categorical attributes
categorical_cols = thyroid_data.select_dtypes(include=["object"]).columns

print("\nCategorical Columns:\n", categorical_cols)
# Identify numeric columns
numeric_cols = thyroid_data.select_dtypes(include=["int64", "float64"]).columns

# Display range (min & max) for numeric attributes
print("\nRange of Numeric Variables:\n", thyroid_data[numeric_cols].describe().loc[["min", "max"]])

# Check for missing values in each column
missing_values = thyroid_data.isnull().sum()

print("\nMissing Values in Each Attribute:\n", missing_values)
# Plot boxplots for numeric columns to detect outliers
plt.figure(figsize=(12, 6))
thyroid_data[numeric_cols].boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot to Detect Outliers")
plt.show()

# Calculate mean & standard deviation for numeric attributes
mean_values = thyroid_data[numeric_cols].mean()
std_values = thyroid_data[numeric_cols].std()

print("\nMean of Numeric Variables:\n", mean_values)
print("\nStandard Deviation of Numeric Variables:\n", std_values)

