# Question-11 and Question-12
import numpy as np
import matplotlib.pyplot as plt

# Create a vector V1 of 100 random numbers
V1 = np.random.rand(100)  # Generates random numbers between 0 and 1

# Sort the vector in increasing order
V1_sorted = np.sort(V1)

# Plot the sorted vector
plt.figure(figsize=(8, 5))
plt.plot(V1_sorted, marker='o', linestyle='-', color='r', label='Sorted Vector')
plt.title("Plot of Sorted Random Vector (V1)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.show()

# Square of the sorted vector
V2=V1_sorted**2


# Plot both V1 and V2
plt.figure(figsize=(10, 6))
plt.plot(V1_sorted, marker='o', linestyle='-', color='r', label='V1 (Sorted Random Numbers)')
plt.plot(V2, marker='x', linestyle='--', color='b', label='V2 (Squared Values)')
plt.title("Plot of V1 and V2")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.show()