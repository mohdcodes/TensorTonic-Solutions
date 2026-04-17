import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # created numpy array because it will be implementing the functions to each element. So no need to worry about the type of x
    x = np.array(x, dtype = float)
    return 1/(1+np.exp(-x))