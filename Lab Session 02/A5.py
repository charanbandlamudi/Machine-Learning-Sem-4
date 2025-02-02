import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

thyroid_data = pd.read_excel(r"/content/Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

print(thyroid_data.head())

print("\nData Types of Each Column:\n", thyroid_data.dtypes)

categorical_cols = thyroid_data.select_dtypes(include=["object"]).columns

print("\nCategorical Columns:\n", categorical_cols)

numeric_cols = thyroid_data.select_dtypes(include=["int64", "float64"]).columns

print("\nRange of Numeric Variables:\n", thyroid_data[numeric_cols].describe().loc[["min", "max"]])

missing_values = thyroid_data.isnull().sum()

print("\nMissing Values in Each Attribute:\n", missing_values)


plt.figure(figsize=(12, 6))
thyroid_data[numeric_cols].boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot to Detect Outliers")
plt.show()

mean_values = thyroid_data[numeric_cols].mean()
std_values = thyroid_data[numeric_cols].std()

print("\nMean of Numeric Variables:\n", mean_values)
print("\nStandard Deviation of Numeric Variables:\n", std_values)
