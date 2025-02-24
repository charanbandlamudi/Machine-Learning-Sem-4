# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files

# Step 1: Upload the Excel file
uploaded = files.upload()

# Get the uploaded filename
file_name = list(uploaded.keys())[0]

# Load the Excel file
xls = pd.ExcelFile(file_name)

# Load the "thyroid0387_UCI" sheet
thyroid_data = pd.read_excel(xls, sheet_name="thyroid0387_UCI")

# Step 2: Extract First Two Observations
first_two_vectors = thyroid_data.iloc[:2]  # Select first two rows

# Remove non-numeric columns (only keep numerical features)
numeric_data = first_two_vectors.select_dtypes(include=["int64", "float64"])

# Convert data to NumPy array
vector_1 = numeric_data.iloc[0].values.reshape(1, -1)
vector_2 = numeric_data.iloc[1].values.reshape(1, -1)

# Step 3: Compute Cosine Similarity
cosine_sim = cosine_similarity(vector_1, vector_2)[0][0]

# Step 4: Print the Result
print("\nCosine Similarity between the first two observations:", round(cosine_sim, 4))

# Step 5: Interpretation
if cosine_sim > 0.8:
    print("✅ The vectors are highly similar.")
elif cosine_sim > 0.5:
    print("✅ The vectors have moderate similarity.")
else:
    print("✅ The vectors are not very similar.")
