import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    if len(x) != len(y):
        raise ValueError("Vectors must have the same length")

    res = 0
    for i in range(len(x)):
        dis_i = abs(x[i]-y[i])
        res += dis_i
    return res
        