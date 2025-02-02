import pandas as pd
import numpy as np
from google.colab import files

file = pd.read_excel(r"/content/Lab Session Data.xlsx",sheet_name="Purchase data")
xls = pd.ExcelFile(file_name)
purchase_data = pd.read_excel(xls, sheet_name="Purchase data")
purchase_data = purchase_data.iloc[:, :5]  
purchase_data = purchase_data.drop(columns=["Customer"])  
purchase_data.columns = ["Candies", "Mangoes", "Milk_Packets", "Payment"]
X = purchase_data[["Candies", "Mangoes", "Milk_Packets"]].values

Y = purchase_data["Payment"].values

X = np.hstack((np.ones((X.shape[0], 1)), X))

X_pseudo_inverse = np.linalg.pinv(X)  

model_vector = X_pseudo_inverse @ Y 

print("Model Vector (Intercept and Coefficients):")

print(model_vector)