import numpy as np
import random

from random_projection import projection_between
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


def create_projections(vectors, num_projections):
    projections = []
    for idx in range(num_projections):
        projection_found = False
        while not projection_found:
            vect1_idx = random.randint(0, len(vectors)-1)
            vect2_idx = random.randint(0, len(vectors)-1)
            vect1 = vectors[vect1_idx]
            vect2 = vectors[vect2_idx]
            dim = random.randint(0, vect1.shape[0]-1)
            if np.sign(vect1[dim]) == np.sign(vect2[dim]):
                try:
                    proj = projection_between(vect1, vect2, dim)
                    assert_projection_bisects(proj, vect1, vect2)

                    projections.append(proj)
                    projection_found = True
                except ValueError:
                    projection_found = False
    return projections


def set_bit(one_hash, idx):
    bit = idx % 64
    mask = np.int64(np.uint64(2 ** bit))
    one_hash[idx // 64] |= mask
    return one_hash


def clear_bit(one_hash, idx):
    bit = idx % 64
    mask = np.int64(np.uint64(2 ** bit))
    mask = ~mask
    one_hash[idx // 64] &= mask
    return one_hash


def train(hashes, vectors):
    hash_len = hashes.shape[1]
    num_projections = hash_len * 64

    projections = create_projections(vectors, num_projections)

    for vect_idx, vect in enumerate(vectors):
        for bit_idx, proj in enumerate(projections):
            dot = np.dot(vect, proj)
            if dot >= 0:
                hashes[vect_idx] = set_bit(hashes[vect_idx], bit_idx)
            else:
                hashes[vect_idx] = clear_bit(hashes[vect_idx], bit_idx)
    return hashes


def test_rand_projection_works_ok_in_very_low_dims():
    np.random.seed(0)
    random.seed(0)

    hash_len = 4
    dims = 8
    rows = 100000
    hashes = np.zeros(dtype=np.int64,
                      shape=(rows, hash_len))
    vectors = random_normed_matrix(rows, dims=dims)
    train(hashes, vectors)

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
    train(hashes, vectors)
    print("Trained")

    n = 100
    top_n_lsh = lsh_nearest_neighbors(hashes, 0, n=n)
    top_n_cos = exact_nearest_neighbors(vectors[0], vectors, n=n)
    recall = len(set(keys(top_n_cos)) & set(keys(top_n_lsh))) / n
    assert recall >= 0.3
