from typing import Union
import numpy as np
import pandas as pd
from time import perf_counter


def get_top_n(dotted, n=100):
    n = min(n, dotted.shape[0])
    top_n = np.argpartition(-dotted, n-1)[:n]
    return top_n, dotted[top_n]


def exact_nearest_neighbors(query_vector: np.ndarray,
                            matrix: Union[np.ndarray, pd.DataFrame],
                            n=100):
    """ nth nearest neighbors as array
        with indices of nearest neighbors"""
    start = perf_counter()

    dotted = np.dot(matrix, query_vector)
    print(f">> Dot - {perf_counter() - start}")
    top_n = get_top_n(dotted)
    print(f">> Tpn - {perf_counter() - start}")
    return top_n
