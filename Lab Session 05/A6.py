import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

# Select numerical columns
numerical_columns = ["Start time", "End time"]
X = df_combined[numerical_columns].values
y = df_combined["Label"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Means for different k (Elbow Method)
distortions = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_train)
    distortions.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), distortions, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title(' Elbow Method for Optimal K')
plt.grid(True)
plt.show()
