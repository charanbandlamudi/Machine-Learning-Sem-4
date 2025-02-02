import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files

file = pd.read_excel(r"/content/Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")


first_two_vectors = thyroid_data.iloc[:2]  

numeric_data = first_two_vectors.select_dtypes(include=["int64", "float64"])

vector_1 = numeric_data.iloc[0].values.reshape(1, -1)
vector_2 = numeric_data.iloc[1].values.reshape(1, -1)

cosine_sim = cosine_similarity(vector_1, vector_2)[0][0]

print("\nCosine Similarity between the first two observations:", round(cosine_sim, 4))

if cosine_sim > 0.8:
    print("✅ The vectors are highly similar.")
elif cosine_sim > 0.5:
    print("✅ The vectors have moderate similarity.")
else:
    print("✅ The vectors are not very similar.")
