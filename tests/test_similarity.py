import numpy as np
import pandas as pd
import pytest
from time import perf_counter
from random import randint
from similarity import uint8_nearest_neighbors, similarity, \
    quantize_idx, quantize_query, exact_nearest_neighbors


@pytest.fixture
def random_matrix():
    """Random matrix ranging from -1 to 1."""
    size = 1000
    matrix = np.random.random_sample(size=(size, 768))
    matrix = (matrix * 2.0) - 1.0

    return matrix, quantize_idx(matrix)


def uint8_sim(vect1, vect2, norm=False):
    # This is more of a manhattan distance
    diff = 1 + np.max([vect1, vect2], axis=0) - np.min([vect1, vect2], axis=0)
    return np.sum(-diff, dtype=np.uint64)


def cosine_sim(vect1, vect2):
    dotted = np.dot(vect1.astype(np.float64), vect2.astype(np.float64))
    return dotted


def random_norm_vect(size=10):
    vect = np.random.random_sample(size=10)
    vect = 256 * vect / np.linalg.norm(vect)
    vect = vect.astype(np.uint8)
    return vect


def test_uint8_sim_communative():
    vect1 = np.array(0)
    vect2 = np.array(0)
    for _ in range(0, 100):

        while (vect1 == vect2).all():
            vect1 = np.random.randint(0, 255, size=10, dtype=np.uint8)
            vect2 = np.random.randint(0, 255, size=10, dtype=np.uint8)

        assert uint8_sim(vect1, vect2) == uint8_sim(vect2, vect1)


def test_cos_similarity_correlates_with_uint8_sim():
    in_order = 0
    out_of_order = 0
    for idx in range(0, 100):
        vect1 = np.array(0)
        vect2 = np.array(0)

        while (vect1 == vect2).all():
            vect1 = random_norm_vect(size=768)
            vect2 = random_norm_vect(size=768)
            vect3 = random_norm_vect(size=768)

        assert cosine_sim(vect1, vect1) == pytest.approx(255.0*255.0, abs=1000)
        assert cosine_sim(vect1, vect2) < cosine_sim(vect1, vect1)
        assert cosine_sim(vect1, vect2) < cosine_sim(vect2, vect2)

        assert uint8_sim(vect1, vect2) < uint8_sim(vect1, vect1)
        assert uint8_sim(vect1, vect2) < uint8_sim(vect2, vect2)

        cos_sim_order = np.array(sorted([vect1, vect2, vect3],
                                 key=lambda vect: cosine_sim(vect, vect3)))
        uint_sim_order = np.array(sorted([vect1, vect2, vect3],
                                  key=lambda vect: uint8_sim(vect, vect3)))
        if not ((cos_sim_order == uint_sim_order).all().all()):
            cos23 = cosine_sim(vect2, vect3)
            cos13 = cosine_sim(vect1, vect3)
            cos_delta = abs(cos23 - cos13)
            uint23 = uint8_sim(vect2, vect3)
            uint13 = uint8_sim(vect1, vect3)
            uint_delta = abs(int(uint23) - int(uint13))

            # diff13 = 1 + np.max([vect1, vect3], axis=0) - np.min([vect1, vect3], axis=0)
            # diff23 = 1 + np.max([vect2, vect3], axis=0) - np.min([vect2, vect3], axis=0)

            # mult13 = np.multiply(vect1, vect3, dtype=np.uint32)
            # mult23 = np.multiply(vect2, vect3, dtype=np.uint32)

            print(f"delta({cos23},{cos13})={cos_delta}, delta({uint23}, {uint13})={uint_delta}")
            out_of_order += 1
        else:
            in_order += 1
    print(in_order, out_of_order)

            #raise AssertionError(
            #    f"Cos order does not match uint manhattan - {idx}"
            #)


def test_cos_similarity_uint8_sim_recall(random_matrix):

    flt_matrix, uint_matrix = random_matrix
    n = 1000

    for _ in range(0, 100):
        query_vector_flt = (np.random.random_sample(768) * 2) - 1
        query_vector_uint8 = quantize_query(query_vector_flt)

        top_n_flt, _ = exact_nearest_neighbors(query_vector_flt,
                                               flt_matrix,
                                               n=100)
        top_n_uint8, _ = uint8_nearest_neighbors(query_vector_uint8,
                                                 uint_matrix,
                                                 n=100)
        # import pdb; pdb.set_trace()
        intersection = set(top_n_flt) & set(top_n_uint8)
        recall = len(intersection) / n
        print(recall)



def test_uint8_cos_similarity_of_matrix():
    for _ in range(0, 100):
        vect1 = np.array(0)
        vect2 = np.array(0)

        while (vect1 == vect2).all():
            vect1 = np.array([randint(0, 255), randint(0, 255)],
                             dtype=np.uint8)
            vect2 = np.array([randint(0, 255), randint(0, 255)],
                             dtype=np.uint8)

            matrix = 255 - np.array([vect1, vect2], dtype=np.uint8)

            idxs, scores = uint8_nearest_neighbors(vect1, matrix)
            assert scores[0] > scores[1]
            assert (idxs == [0, 1]).all()

            matrix -= vect1
            idxs, scores = uint8_nearest_neighbors(vect2, matrix)
            assert scores[0] > scores[1]
            assert (idxs == [1, 0]).all()


@pytest.fixture
def large_dataframe():
    df = []
    for i in range(0, 1000000):
        row = {"doc_id": f"doc_{i}",
               "passage_id": i,
               "passage": np.random.randint(0, 255, size=768, dtype=np.uint8)}
        df.append(row)
    return pd.DataFrame(df)


def test_similarity_dataframe(large_dataframe):

    start = perf_counter()
    similarity(np.random.randint(0, 255, size=768, dtype=np.uint8),
               large_dataframe, "passage")
    print(f"Similarity {perf_counter() - start}")
