import numpy as np
import pandas as pd
import pytest
from time import perf_counter
from similarity import exact_nearest_neighbors, get_top_n, keys, scores


def recall(ground_truth, top_n):
    ground_truth_set = set(ground_truth)
    top_n_set = set(top_n)
    return len(ground_truth_set & top_n_set) / len(ground_truth_set)


@pytest.fixture
def large_dataframe():
    df = []
    df = pd.DataFrame(np.random.random_sample(size=(1000000, 768)))
    return df


def test_similarity_dataframe_gives_dot_prod_scores(large_dataframe):

    query_vector = np.random.random_sample(size=768)

    def encoder(query):
        assert query == "foo"
        return query_vector

    start = perf_counter()
    top_n = exact_nearest_neighbors(query_vector,
                                    large_dataframe)
    scored = scores(top_n)
    top_n = keys(top_n)
    print(f"Similarity {perf_counter() - start}")
    matrix = large_dataframe.loc[top_n, :]
    dotted = np.dot(matrix, query_vector)
    assert np.isclose(
        scored,
        dotted
    ).all()


def test_similarity_dataframe_flt16_has_high_recall(large_dataframe):

    query_vector = np.random.random_sample(size=768)

    start = perf_counter()
    top_n = exact_nearest_neighbors(query_vector,
                                    large_dataframe)
    print(f"f64 similarity {perf_counter() - start}")
    top_n = keys(top_n)

    query16 = query_vector.astype(np.half)

    half_df = large_dataframe.astype(np.half)
    start = perf_counter()
    top_n16 = exact_nearest_neighbors(query16,
                                      half_df)
    print(f"f16 Similarity {perf_counter() - start}")
    top_n16 = keys(top_n16)

    assert recall(top_n, top_n16) >= 0.95


def test_get_top_n_sorts():
    dotted = np.array([1000, 900])
    top_n = get_top_n(dotted)
    scored = scores(top_n)
    top_n = keys(top_n)
    assert top_n == [0, 1]
    assert scored == [1000, 900]

    dotted = np.array([900, 1000])
    top_n = get_top_n(dotted)
    scored = scores(top_n)
    top_n = keys(top_n)
    assert top_n == [1, 0]
    assert scored == [1000, 900]


def test_get_top_n_sorts_up_to_n():
    dotted = np.array([900, 1000, 100, 400])
    top_n = get_top_n(dotted, n=2)
    scored = scores(top_n)
    top_n = keys(top_n)
    assert top_n == [1, 0]
    assert scored == [1000, 900]
