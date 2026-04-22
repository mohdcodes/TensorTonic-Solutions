def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    import math
    return [math.log(1+x) for x in values]