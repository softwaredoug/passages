import numpy as np
import pandas as pd
import pytest
from time import perf_counter
from similarity import similarity


def recall(ground_truth: pd.DataFrame, top_n: pd.DataFrame):
    ground_truth_set = set(ground_truth.index)
    top_n_set = set(top_n.index)
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
    top_n = similarity("foo", encoder,
                       large_dataframe)
    print(f"Similarity {perf_counter() - start}")
    scores = np.array(top_n['score'].tolist())
    matrix = top_n.loc[:, top_n.columns != 'score']
    dotted = np.dot(matrix, query_vector)
    assert np.isclose(
        scores,
        dotted
    ).all()


def test_similarity_dataframe_same_perf_regardless_of_index(large_dataframe):
    query_vector = np.random.random_sample(size=768)

    def encoder(query):
        assert query == "foo"
        return query_vector

    start = perf_counter()
    similarity("foo", encoder,
               large_dataframe)
    print(f"Similarity {perf_counter() - start}")

    large_dataframe['doc_id'] = [idx
                                 for idx
                                 in large_dataframe.index]
    large_dataframe['passage_id'] = [idx
                                     for idx
                                     in large_dataframe.index]
    start = perf_counter()
    import pdb; pdb.set_trace()
    similarity("foo", encoder,
               large_dataframe.drop(columns=['doc_id', 'passage_id']))
    print(f"Similarity {perf_counter() - start}")


def test_similarity_dataframe_flt16_has_high_recall(large_dataframe):

    query_vector = np.random.random_sample(size=768)

    def encoder(query):
        assert query == "foo"
        return query_vector

    top_n = similarity("foo", encoder,
                       large_dataframe)

    def encoder_16(query):
        assert query == "foo"
        return query_vector.astype(np.half)

    half_df = large_dataframe.astype(np.half)
    start = perf_counter()
    top_n16 = similarity("foo", encoder,
                         half_df)
    print(f"Similarity {perf_counter() - start}")

    assert recall(top_n, top_n16) >= 0.95
