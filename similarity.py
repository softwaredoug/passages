from typing import Callable, Union
import numpy as np
import pandas as pd


def get_top_n(dotted, n=100):
    n = min(n, dotted.shape[0])
    top_n = np.argpartition(-dotted, n-1)[:n]
    return top_n, dotted[top_n]


def exact_nearest_neighbors(query_vector: np.ndarray,
                            matrix: Union[np.ndarray, pd.DataFrame],
                            n=100):
    """ nth nearest neighbors as array
        with indices of nearest neighbors"""

    dotted = np.dot(matrix, query_vector)
    top_n = get_top_n(dotted)
    return top_n


def similarity(query: str, encoder: Callable[[str], np.ndarray],
               corpus: pd.DataFrame, n=10):
    """corpus is a dataframe with columns 0...n, each row with a normalized
       vector. No additional columns are present."""

    query_vector = encoder(query)
    top_n, scores = exact_nearest_neighbors(query_vector,
                                            corpus, n=n)

    top_n = pd.DataFrame({'icol': top_n, 'score': scores})
    top_n = top_n.set_index('icol').sort_values('score', ascending=False)
    top_n_corpus = corpus.iloc[top_n.index].copy()
    top_n_corpus['score'] = sorted(scores, reverse=True)

    return top_n_corpus
