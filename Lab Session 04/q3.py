import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
df = pd.ExcelFile(file_name)

df_irctc = pd.read_excel(file_name, sheet_name='IRCTC Stock Price')

df_irctc = df_irctc[['Open', 'Price']].dropna()

df_irctc['Open'] = pd.to_numeric(df_irctc['Open'], errors='coerce')
df_irctc['Price'] = pd.to_numeric(df_irctc['Price'], errors='coerce')

df_irctc = df_irctc.dropna()

X = df_irctc['Open'].values[:20]
Y = df_irctc['Price'].values[:20]
classes = np.where(Y < X, 0, 1)

plt.figure(figsize=(8, 6))
for i in range(20):
    if classes[i] == 0:
        plt.scatter(X[i], Y[i], color='blue', label='Class 0 (Down - Blue)' if i == 0 else "")
    else:
        plt.scatter(X[i], Y[i], color='red', label='Class 1 (Up - Red)' if i == 0 else "")

plt.title("IRCTC Stock Price Movement (Class 0 - Blue, Class 1 - Red)")
plt.xlabel("Open Price")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.show()
