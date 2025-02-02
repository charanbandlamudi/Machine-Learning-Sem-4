def multiply_matrices(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:  
        return "Cannot multiply: Column count of A must match row count of B"
    result_matrix = [[0] * cols_B for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result_matrix[i][j] += A[i][k] * B[k][j]

    return result_matrix
A = [list(map(int, input(f"Enter the elements for row {i + 1} of matrix A: ").split())) for i in range(2)]
B = [list(map(int, input(f"Enter the elements for row {i + 1} of matrix B: ").split())) for i in range(2)]

result = multiply_matrices(A, B)

if isinstance(result, str):  
    print(result)
else:
    print("The product of matrices A and B is:")
    for row in result:
        print(row)
