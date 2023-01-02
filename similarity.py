from typing import Union
import numpy as np
import pandas as pd


def keys(top):
    return [tp[0] for tp in top]


def scores(top):
    return [tp[1] for tp in top]


def get_top_n(dotted, n=100):
    n = min(n, dotted.shape[0])
    top_n = np.argpartition(-dotted, n-1)[:n]
    scores = dotted[top_n]
    return sorted(zip(top_n, scores),
                  key=lambda scored: scored[1],
                  reverse=True)


def exact_nearest_neighbors(query_vector: np.ndarray,
                            matrix: Union[np.ndarray, pd.DataFrame],
                            n=100):
    """ nth nearest neighbors as array
        with indices of nearest neighbors"""
    dotted = np.dot(matrix, query_vector)
    # print(f">> Dot - {perf_counter() - start}")
    top_n = get_top_n(dotted, n=n)
    # print(f">> Tpn - {perf_counter() - start}")
    return top_n


def exact_farthest_neighbors(query_vector: np.ndarray,
                             matrix: Union[np.ndarray, pd.DataFrame],
                             n=100):
    """ nth farthest  as array
        with indices of nearest neighbors"""
    dotted = np.dot(matrix, query_vector)
    top_n = get_top_n(-dotted, n=n)
    return top_n


def random_neighbors(query_vector: np.ndarray,
                     matrix: Union[np.ndarray, pd.DataFrame],
                     n=100):
    dotted = np.dot(matrix, query_vector)
    randoms = np.random.randint(len(matrix), size=n)
    return list(zip(randoms, dotted[randoms]))
