import numpy as np


def random_normed_matrix(rows, dims=768):
    """Random normalized matrix ranging from -1 to 1."""
    matrix = np.random.random_sample(size=(rows, dims))
    matrix = (matrix * 2.0) - 1.0
    norms = np.linalg.norm(matrix, axis=1).reshape((rows, 1))
    matrix = matrix / norms

    return matrix
