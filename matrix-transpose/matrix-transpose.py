import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # code for ML pipeline
    # this backed is is C
    return np.array(A).T
    # code for optimised python code in SC -> O(1)
    # n = len(matrix)
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]