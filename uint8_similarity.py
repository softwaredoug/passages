from typing import Union
from similarity import get_top_n
import numpy as np
import pandas as pd


# TODO - can we encode the matrix on a logarithmic scale
#        to allow memory-efficient dot products? Such as
#        255 -> 8
#        250 -> 8
#        instead of multiplying, you would sum these -> 16
def quantize_query(arr, bits=256):
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


def quantize_idx(arr):
    return quantize_query(arr)


def uint8_nearest_neighbors(query_vector: np.ndarray,
                            matrix: Union[np.ndarray, pd.DataFrame],
                            n=100):
    # start = perf_counter()
    dotted = uint8_sim_matrix(query_vector, matrix)
    # print(f">>Dot prod took {perf_counter() - start}")
    top_n, scores = get_top_n(dotted)
    # print(f">>Top N {perf_counter() - start}")
    return top_n, scores


# Encoding as uint8 creates ~15x speedup
def uint8_sim_matrix(query_vector, matrix):
    # This is more of a manhattan distance, because dot product will overflow
    # the uint8, and be inaccurate. But summing the absolute difference is
    # another way of getting the distance (255 - ...) for the similarity
    # We perform the 255 - on the matrix when encoding it
    # start = perf_counter()
    # Can we do this sum without allocating a new matrix?
    # Like some kind of pairwise add and accumulate?
    # for each ith element
    #   sum += (mat_row[i] + query_vector[i])
    # matrix += query_vector
    # print(f"\n>> >> Diff {perf_counter() - start}")
    diff = 1 + (
        np.fmax(query_vector, matrix) -
        np.fmin(query_vector, matrix)
    )
    tot = np.sum(-diff,
                 dtype=np.uint32,
                 axis=1)
    # matrix -= query_vector
    # print(f">> >> Tot  {perf_counter() - start}")
    return tot
