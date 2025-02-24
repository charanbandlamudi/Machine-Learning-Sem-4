# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Upload the Excel file
from google.colab import files
uploaded = files.upload()

# Read the Excel file
file_name = list(uploaded.keys())[0]
df = pd.ExcelFile(file_name)

# Load the 'IRCTC Stock Price' sheet
df_irctc = pd.read_excel(file_name, sheet_name='IRCTC Stock Price')

# Prepare the data
# Using 'Open' as X and 'Price' as Y for classification
df_irctc = df_irctc[['Open', 'Price']].dropna()

# Convert columns to numeric
df_irctc['Open'] = pd.to_numeric(df_irctc['Open'], errors='coerce')
df_irctc['Price'] = pd.to_numeric(df_irctc['Price'], errors='coerce')

# Drop rows with NaN values
df_irctc = df_irctc.dropna()

# Take first 20 data points for training
X_train = df_irctc[['Open', 'Price']].values[:20]

# Class Labels:
# Class 0 (Blue) -> Price < Open (Stock went down)
# Class 1 (Red) -> Price >= Open (Stock went up or stayed the same)
y_train = np.where(X_train[:, 1] < X_train[:, 0], 0, 1)

# Generate test set data with values of X & Y between 0 and 10 with increments of 0.1
x_test_values = np.arange(0, 10.1, 0.1)
y_test_values = np.arange(0, 10.1, 0.1)
X_test, Y_test = np.meshgrid(x_test_values, y_test_values)
X_test_flat = X_test.ravel()
Y_test_flat = Y_test.ravel()
test_points = np.c_[X_test_flat, Y_test_flat]

# List of k values to observe class boundary changes
k_values = [1, 3, 5, 7, 9]

# Plot the decision boundaries for each k value
plt.figure(figsize=(20, 15))
for idx, k in enumerate(k_values):
    # Train k-NN classifier with current k value
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict the class of test points
    predicted_classes = knn.predict(test_points)
    predicted_classes = predicted_classes.reshape(X_test.shape)

    # Create subplot for current k value
    plt.subplot(2, 3, idx+1)
    plt.contourf(X_test, Y_test, predicted_classes, alpha=0.5, cmap='bwr')
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0 (Train - Blue)')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1 (Train - Red)')

    # Add labels and title for each subplot
    plt.title(f"k-NN Classification (k={k})")
    plt.xlabel("Open Price")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.grid(True)

# Display all subplots
plt.tight_layout()
plt.show()
