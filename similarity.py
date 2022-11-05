from typing import Callable, cast
import numpy as np
import pandas as pd


def quantize(arr, bits=256):
    """Scale to 0-255, cast to uint8."""
    floor = -1.0
    ceil = 1.0
    arr[arr == 1.0] = ceil - 0.00001
    assert not (arr > ceil).any()
    assert not (arr < floor).any()
    flt_per_bucket = (abs(floor) + abs(ceil)) / bits
    quantized = (arr - floor) // flt_per_bucket
    assert not (arr >= bits).any()
    return quantized.astype(np.uint8)


def get_top_n(dotted, n=100):
    n = min(n, dotted.shape[0])
    top_n = np.argpartition(-dotted, n-1)[:n]
    return top_n, dotted[top_n]


def exact_nearest_neighbors(query_vector: np.ndarray,
                            matrix: np.ndarray, n=100):
    """ nth nearest neighbors as array
        with indices of nearest neighbors"""

    dotted = np.dot(matrix, query_vector)
    return get_top_n(dotted)


def uint8_sim_matrix(query_vector, matrix):
    # This is more of a manhattan distance, because dot product will overflow
    # the uint8, and be inaccurate. But summing the absolute difference is
    # another way of getting the distance (255 - ...) for the similarity
    return np.sum(255 - (matrix - query_vector), dtype=np.uint64, axis=1)


def uint8_nearest_neighbors(query_vector: np.ndarray,
                            matrix: np.ndarray,
                            n=100):
    dotted = uint8_sim_matrix(query_vector, matrix)
    return get_top_n(dotted)


def similarity(query: str, encoder: Callable[[str], np.ndarray],
               corpus: pd.DataFrame, column: str, n=10):

    query_vector = encoder(query)
    vectors = np.array(cast(np.ndarray, corpus[column].tolist()))

    if query_vector.dtype == np.uint8:
        top_n, scores = uint8_nearest_neighbors(query_vector, vectors, n=n)
    else:
        top_n, scores = exact_nearest_neighbors(query_vector, vectors, n=n)

    top_n = pd.DataFrame({'icol': top_n, 'scores': scores})
    top_n = top_n.set_index('icol').sort_values('scores', ascending=False)
    top_n_corpus = corpus.iloc[top_n.index].copy()
    top_n_corpus['scores'] = sorted(scores, reverse=True)

    return top_n_corpus
