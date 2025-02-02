import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files

file = pd.read_excel(r"/content/Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

first_20_vectors = thyroid_data.iloc[:20]

binary_cols = [

  col

  for col in thyroid_data.columns

  if thyroid_data[col].dropna().apply(lambda x: x in (0, 1, 0.0, 1.0)).all()

  and thyroid_data[col].dropna().nunique() <= 2 # Check for at most 2 unique values

  and pd.api.types.is_numeric_dtype(thyroid_data[col]) # Check if the column is numeric

]

binary_data = first_20_vectors[binary_cols]

binary_matrix = binary_data.to_numpy()

numeric_data = first_20_vectors.select_dtypes(include=["int64", "float64"]).to_numpy()

def calculate_jc_smc(matrix):

  """Calculate Jaccard and SMC similarity matrices."""

  n = matrix.shape[0]

  jc_matrix = np.zeros((n, n))

  smc_matrix = np.zeros((n, n))

  for i in range(n):

    for j in range(n):

      f11 = np.sum((matrix[i] == 1) & (matrix[j] == 1))

      f00 = np.sum((matrix[i] == 0) & (matrix[j] == 0))

      f10 = np.sum((matrix[i] == 1) & (matrix[j] == 0))

      f01 = np.sum((matrix[i] == 0) & (matrix[j] == 1))

      jc_matrix[i, j] = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0

      smc_matrix[i, j] = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

  return jc_matrix, smc_matrix

jc_matrix, smc_matrix = calculate_jc_smc(binary_matrix)

cos_matrix = cosine_similarity(numeric_data)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(jc_matrix, annot=True, cmap="coolwarm", ax=axes[0])

axes[0].set_title("Jaccard Coefficient")

sns.heatmap(smc_matrix, annot=True, cmap="coolwarm", ax=axes[1])

axes[1].set_title("Simple Matching Coefficient")

sns.heatmap(cos_matrix, annot=True, cmap="coolwarm", ax=axes[2])

axes[2].set_title("Cosine Similarity")

plt.tight_layout()

plt.show()