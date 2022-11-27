import numpy as np
import random

from rand_proj_similarity import train, create_projections
from utils import random_normed_matrix
from similarity import exact_nearest_neighbors, keys
from lsh_similarity import lsh_nearest_neighbors


INT64_MAX = np.iinfo(np.int64).max


def assert_projection_bisects(projection, vect1, vect2):
    dot1 = np.dot(vect1, projection)
    dot2 = np.dot(vect2, projection)
    if dot1 > 0:
        assert dot2 < 0
    elif dot1 < 0:
        assert dot2 > 0
    else:
        assert False, "Both cant be 0"


def test_rand_projection_works_ok_in_very_low_dims():
    np.random.seed(0)
    random.seed(0)

    hash_len = 4
    dims = 8
    rows = 100000
    hashes = np.zeros(dtype=np.int64,
                      shape=(rows, hash_len))
    vectors = random_normed_matrix(rows, dims=dims)
    num_projections = hash_len * 64
    projections = create_projections(vectors, num_projections)
    train(hashes, projections, vectors)

    n = 100
    print("Trained")
    top_n_lsh = lsh_nearest_neighbors(hashes, 0, n=n)
    top_n_cos = exact_nearest_neighbors(vectors[0], vectors, n=n)
    recall = len(set(keys(top_n_cos)) & set(keys(top_n_lsh))) / n
    print(recall)
    assert recall >= 0.5


def test_rand_projection_works_ok_in_low_dims():
    np.random.seed(0)
    random.seed(0)

    hash_len = 4
    dims = 16
    rows = 100000
    hashes = np.zeros(dtype=np.int64,
                      shape=(rows, hash_len))
    vectors = random_normed_matrix(rows, dims=dims)

    num_projections = hash_len * 64
    projections = create_projections(vectors, num_projections)
    train(hashes, projections, vectors)
    print("Trained")

    n = 100
    top_n_lsh = lsh_nearest_neighbors(hashes, 0, n=n)
    top_n_cos = exact_nearest_neighbors(vectors[0], vectors, n=n)
    recall = len(set(keys(top_n_cos)) & set(keys(top_n_lsh))) / n
    assert recall >= 0.3
