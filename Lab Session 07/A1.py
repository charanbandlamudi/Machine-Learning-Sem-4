from google.colab import files
import pandas as pd

# Upload the dataset
uploaded = files.upload()

# Load dataset
data = pd.read_excel('219_student.xlsx')
print("Data Loaded Successfully")
print(data.head())

# A1: Continue unfinished experiments (Initial Data Exploration)
print("\nData Information:")
data.info()
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

