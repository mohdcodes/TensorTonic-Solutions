import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    if len(x) != len(y):
        raise ValueError("Vectors must have the same length")
    n = len(x)
    res = 0
    for i in range(n):
        sq_i = x[i]-y[i]
        sq_i = sq_i ** 2
        res = res + sq_i
    return np.sqrt(res)