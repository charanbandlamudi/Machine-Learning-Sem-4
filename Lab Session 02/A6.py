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

# Display missing values before imputation
print("\nMissing Values Before Imputation:\n", thyroid_data.isnull().sum())
# Identify numeric and categorical columns
numeric_cols = thyroid_data.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = thyroid_data.select_dtypes(include=["object"]).columns

print("\nNumeric Columns:", numeric_cols)
print("\nCategorical Columns:", categorical_cols)

# Plot boxplots for numeric columns to detect outliers
plt.figure(figsize=(12, 6))
thyroid_data[numeric_cols].boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot to Detect Outliers")
plt.show()
# Fill missing values in numeric columns
for col in numeric_cols:
    if thyroid_data[col].isnull().sum() > 0:  # Check if there are missing values
        if thyroid_data[col].skew() < 1:  # No strong skewness, assume no outliers
            thyroid_data[col].fillna(thyroid_data[col].mean(), inplace=True)  # Use Mean
        else:
            thyroid_data[col].fillna(thyroid_data[col].median(), inplace=True)  # Use Median for outliers

print("\nMissing Values After Numeric Imputation:\n", thyroid_data.isnull().sum())

# Fill missing values in categorical columns using Mode
for col in categorical_cols:
    if thyroid_data[col].isnull().sum() > 0:
        thyroid_data[col].fillna(thyroid_data[col].mode()[0], inplace=True)  # Use Mode

print("\nMissing Values After Categorical Imputation:\n", thyroid_data.isnull().sum())

# Save the cleaned dataset
thyroid_data.to_excel("Imputed_Thyroid_Data.xlsx", index=False)

# Provide download link for the updated dataset
files.download("Imputed_Thyroid_Data.xlsx")
