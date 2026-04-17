import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # transposne matrix using loops.
    # SC -> O(i*j)
    # TC -> O(i*j)
    i = len(A)
    j = len(A[0])
    result = []
    for r_num in range(0, j):
        lst = []
        for c_num in range(0, i):
            lst.append(A[c_num][r_num])
        result.append(lst)
    return np.array(result)
