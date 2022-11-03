import numpy as np
import pytest
from similarity import uint8_nearest_neighbors
from random import randint


def uint8_sim(vect1, vect2):
    # This is more of a manhattan distance
    return np.sum(255 - (vect1 - vect2), dtype=np.uint64)


def cosine_sim(vect1, vect2):
    norm1 = np.linalg.norm(vect1)
    norm2 = np.linalg.norm(vect2)
    dotted = np.dot(vect1.astype(np.float64), vect2.astype(np.float64))
    return dotted / (norm1 * norm2)


def test_cos_similarity_correlates_with_uint8_sim():
    for _ in range(0, 100):
        vect1 = np.array(0)
        vect2 = np.array(0)

        while (vect1 == vect2).all():
            vect1 = np.array([randint(0, 255), randint(0, 255)],
                             dtype=np.uint8)
            vect2 = np.array([randint(0, 255), randint(0, 255)],
                             dtype=np.uint8)

        assert cosine_sim(vect1, vect1) == pytest.approx(1.0)
        assert cosine_sim(vect1, vect2) < cosine_sim(vect1, vect1)
        assert cosine_sim(vect1, vect2) < cosine_sim(vect2, vect2)

        assert uint8_sim(vect1, vect2) < uint8_sim(vect1, vect1)
        assert uint8_sim(vect1, vect2) < uint8_sim(vect2, vect2)


def test_uint8_cos_similarity_of_matrix():
    for _ in range(0, 100):
        vect1 = np.array(0)
        vect2 = np.array(0)

        while (vect1 == vect2).all():
            vect1 = np.array([randint(0, 255), randint(0, 255)],
                             dtype=np.uint8)
            vect2 = np.array([randint(0, 255), randint(0, 255)],
                             dtype=np.uint8)

            matrix = np.array([vect1, vect2], dtype=np.uint8)

            idxs, scores = uint8_nearest_neighbors(vect1, matrix)
            assert scores[0] > scores[1]
            assert (idxs == [0, 1]).all()
            idxs, scores = uint8_nearest_neighbors(vect2, matrix)
            assert scores[0] > scores[1]
            assert (idxs == [1, 0]).all()
