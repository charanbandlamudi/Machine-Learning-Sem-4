def transpose_of_matrix(matrix):
    return [list(x) for x in zip(*matrix)] 
matrix = [list(map(int, input(f"Enter the elements of row {i + 1} of the matrix: ").split())) for i in range(2)]
transposed_matrix = transpose_of_matrix(matrix)
print("The transposed matrix is:")
for row in transposed_matrix:
    print(row)
