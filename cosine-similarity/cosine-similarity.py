import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.array(a)
    b = np.array(b)
    dot_prod = np.dot(a, b)
    # dot_prod = a@b
    sq_a = 0
    for i in range(len(a)):
        sq_a += a[i]**2
    res_a = np.sqrt(sq_a)
    sq_b = 0
    for i in range(len(b)):
        sq_b += b[i]**2
    res_b = np.sqrt(sq_b)
    if res_a == 0 or res_b == 0: return 0.0
    return dot_prod/(res_a*res_b)