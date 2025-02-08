import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from scipy.spatial.distance import minkowski

uploaded = files.upload()

file_name = list(uploaded.keys())[0] 
xls = pd.ExcelFile(file_name)

df_stock = pd.read_excel(xls, sheet_name="IRCTC Stock Price")

df_stock["Price"] = pd.to_numeric(df_stock["Price"], errors="coerce")
df_stock["Open"] = pd.to_numeric(df_stock["Open"], errors="coerce")

df_stock = df_stock.dropna(subset=["Price", "Open"])

price_vector = df_stock["Price"].values
open_vector = df_stock["Open"].values

r_values = range(1, 11)
distances = [minkowski(price_vector, open_vector, r) for r in r_values]

plt.figure(figsize=(8, 5))
plt.plot(r_values, distances, marker='o', linestyle='-')
plt.xlabel("r (Minkowski Parameter)")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance between Price and Open")
plt.grid(True)
plt.show()
