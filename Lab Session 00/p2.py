# Question-4

import numpy as np

# Create a matrix of zeros with 4 rows and 3 columns
matrix = np.zeros((4, 3))

# Fill the matrix with random numbers
matrix = np.random.rand(4, 3)  # Replace zeros with random values

# Convert the matrix into a single-dimensional array
flattened_array = matrix.flatten()

print("Matrix (4x3):\n", matrix)
print("\nFlattened Array:\n", flattened_array)
