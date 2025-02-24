# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Excel file (Upload manually in Colab and update file path)
from google.colab import files
uploaded = files.upload()

# Get the uploaded filename
file_name = list(uploaded.keys())[0]

# Load the Excel file
xls = pd.ExcelFile(file_name)

# Load the "Purchase data" sheet
purchase_data = pd.read_excel(xls, sheet_name="Purchase data")


purchase_data = purchase_data.iloc[:, :5]
purchase_data = purchase_data.drop(columns=["Customer"])


purchase_data.columns = ["Candies", "Mangoes", "Milk_Packets", "Payment"]


purchase_data["Category"] = (purchase_data["Payment"] > 200).astype(int)

# Define features (X) and target variable (Y)
X = purchase_data[["Candies", "Mangoes", "Milk_Packets"]].values
Y = purchase_data["Category"].values

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train logistic regression model
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

# Predict on test data
Y_pred = classifier.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)
