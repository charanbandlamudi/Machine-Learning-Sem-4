import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

file_name = list(uploaded.keys())[0] 
xls = pd.ExcelFile(file_name)

df_stock = pd.read_excel(xls, sheet_name="IRCTC Stock Price")

df_stock["Price"] = pd.to_numeric(df_stock["Price"], errors="coerce")

df_stock = df_stock.dropna(subset=["Price"])

hist, bins = np.histogram(df_stock["Price"], bins=10)

plt.figure(figsize=(8, 5))
plt.hist(df_stock["Price"], bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Stock Price")
plt.ylabel("Frequency")
plt.title("Histogram of IRCTC Stock Prices")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

mean_price = df_stock["Price"].mean()
variance_price = df_stock["Price"].var()

print(f"Mean Stock Price: {mean_price}")
print(f"Variance: {variance_price}")
