import numpy as np
import pandas as pd
import pytest
from time import perf_counter
from similarity import exact_nearest_neighbors


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
    top_n, scores = exact_nearest_neighbors(query_vector,
                                            large_dataframe)
    print(f"Similarity {perf_counter() - start}")
    matrix = large_dataframe.loc[top_n, :]
    dotted = np.dot(matrix, query_vector)
    assert np.isclose(
        scores,
        dotted
    ).all()


def test_similarity_dataframe_flt16_has_high_recall(large_dataframe):

    query_vector = np.random.random_sample(size=768)

    top_n, _ = exact_nearest_neighbors(query_vector,
                                       large_dataframe)

    query16 = query_vector.astype(np.half)

    half_df = large_dataframe.astype(np.half)
    start = perf_counter()
    top_n16, _ = exact_nearest_neighbors(query16,
                                         half_df)
    print(f"Similarity {perf_counter() - start}")

    assert recall(top_n, top_n16) >= 0.95
