import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    ferq_dict = {}
    for i in tokens:
        if i in ferq_dict:
            ferq_dict[i] += 1
        else:
            ferq_dict[i] = 1

    result =[]
    for word in vocab:
        if word in ferq_dict:
            result.append(ferq_dict[word])
        else:
            result.append(0)
    return np.array(result, dtype=int)