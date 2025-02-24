import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and merge Excel files (adjust file paths as needed)
all_data = []
file_paths = ['255_s.xlsx', '259_s.xlsx', '261_s.xlsx', '285_s.xlsx', '287_s.xlsx',
              '219_student.xlsx', '220_student.xlsx', '221_student.xlsx', '222_student.xlsx', '223_student.xlsx']
for file in file_paths:
    df = pd.read_excel(file)
    df["Source_File"] = file
    all_data.append(df)
df_combined = pd.concat(all_data, ignore_index=True)
df_combined.columns = df_combined.columns.str.strip()

# Create "Label" column
if "Label" not in df_combined.columns:
    df_combined["Label"] = df_combined["Member"].apply(lambda x: 1 if x == "Student" else 0)

# Select numerical column
X = df_combined[["Start time"]].values
y = df_combined["Label"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model_one_feature = LinearRegression().fit(X_train, y_train)

# Output
print("\n Model Coefficient:", model_one_feature.coef_)
print(" Model Intercept:", model_one_feature.intercept_)
