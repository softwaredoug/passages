import numpy as np
import random
import pytest

from rand_proj_similarity import train, create_projections, clear_bit, set_bit
from utils import random_normed_matrix
from similarity import exact_nearest_neighbors, keys
from lsh_similarity import lsh_nearest_neighbors, to_01_scale
from hamming import different_bits


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


@pytest.fixture
def very_similar_vectors_2d():
    v1 = np.array([1.0, 1.0])
    v2 = np.array([0.95, 1.0])

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    vectors = np.array([v1, v2])
    return vectors


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


def get_sim_of(top_n_lsh, doc_id):
    DOC_ID_IDX = 0
    for idx, lsh in enumerate(top_n_lsh):
        if top_n_lsh[idx][DOC_ID_IDX] == doc_id:
            return top_n_lsh[idx]
    raise AssertionError(f"Could not find {doc_id} in top LSH results")


def test_very_similar_converges_small_hash(very_similar_vectors_2d):
    vectors = very_similar_vectors_2d
    np.random.seed(11)
    random.seed(11)
    query_vector_idx = 0
    compare_vector_idx = 1

    top_n_cos = exact_nearest_neighbors(vectors[0], vectors, n=2)
    num_projections = 64
    hash_len = 1
    hashes = np.zeros(dtype=np.int64,
                      shape=(2, hash_len))
    projections = create_projections(vectors, num_projections)
    hashes = train(hashes, projections, vectors)
    top_n_lsh = lsh_nearest_neighbors(hashes, query_vector_idx, n=2)
    compare_to = get_sim_of(top_n_lsh, top_n_cos[compare_vector_idx][0])

    lsh_sim_estimate = compare_to[1]
    actual_cos_sim = top_n_cos[compare_vector_idx][1]

    assert pytest.approx(lsh_sim_estimate, 0.02) == actual_cos_sim


def test_very_similar_converges_big_hash(very_similar_vectors_2d):
    vectors = very_similar_vectors_2d
    # np.random.seed(11)
    # random.seed(11)

    query_vector_idx = 0
    compare_vector_idx = 1

    top_n_cos = exact_nearest_neighbors(vectors[0], vectors, n=2)
    actual_cos_sim = top_n_cos[compare_vector_idx][1]
    num_projections = 1024
    hash_len = 16
    hashes = np.zeros(dtype=np.int64,
                      shape=(2, hash_len))
    projections = create_projections(vectors, num_projections)
    hashes = train(hashes, projections, vectors)
    hashes_big = hashes.copy()
    projections_big = projections.copy()
    top_n_lsh = lsh_nearest_neighbors(hashes, query_vector_idx, n=2)
    compare_to = get_sim_of(top_n_lsh, top_n_cos[compare_vector_idx][0])

    lsh_sim_estimate_big_hash = compare_to[1]

    num_projections = 64
    hash_len = 1
    hashes = np.zeros(dtype=np.int64,
                      shape=(2, hash_len))
    projections = create_projections(vectors, num_projections)
    hashes = train(hashes, projections, vectors)
    top_n_lsh = lsh_nearest_neighbors(hashes, query_vector_idx, n=2)
    compare_to = get_sim_of(top_n_lsh, top_n_cos[compare_vector_idx][0])

    lsh_sim_estimate_small_hash = compare_to[1]
    delta_big = abs(actual_cos_sim - lsh_sim_estimate_small_hash),
    delta_small = abs(actual_cos_sim - lsh_sim_estimate_big_hash),

    print(actual_cos_sim,
          abs(actual_cos_sim - lsh_sim_estimate_small_hash),
          abs(actual_cos_sim - lsh_sim_estimate_big_hash))
    # Which projections produce opposite dot products?
    # Do they actually bisect the two?

    projections_different = different_bits(hashes_big[0], hashes_big[1])

    assert delta_big < delta_small


def test_more_projections_converges_to_real_similarity():
    # np.random.seed(11)
    # random.seed(11)

    hash_len = 128
    dims = 10
    rows = 2
    n = 2

    # Get two vectors
    vectors = random_normed_matrix(rows, dims=dims)
    top_n_cos = exact_nearest_neighbors(vectors[0], vectors, n=2)
    compare_sim_cos = top_n_cos[1]

    compare_sim_cos = (compare_sim_cos[0], to_01_scale(compare_sim_cos[1]))

    for hash_len in range(1, 32):
        num_projections = hash_len * 64

        hashes = np.zeros(dtype=np.int64,
                          shape=(rows, hash_len))
        projections = create_projections(vectors, num_projections)
        hashes = train(hashes, projections, vectors)

        # Cos and LSH sim of most similar should converge
        top_n_lsh = lsh_nearest_neighbors(hashes, 0, n=n)
        compare_rand_proj = None
        DOC_ID = 0
        for i in range(0, n):
            if top_n_lsh[i][DOC_ID] == compare_sim_cos[DOC_ID]:
                compare_rand_proj = top_n_lsh[i]
                break
        # delta = abs(compare_rand_proj[1] - compare_sim_cos[1])
        # assert delta < rand_proj_delta_last
        print(compare_sim_cos, compare_rand_proj)
        # rand_proj_delta_last = delta


def test_lsh_sim_converges_to_cos_similarity():

    # np.random.seed(11)
    # random.seed(11)

    hash_len = 128
    dims = 10
    rows = 1000
    n = 1000

    vectors = random_normed_matrix(rows, dims=dims)
    top_n_cos = exact_nearest_neighbors(vectors[0], vectors, n=n)
    compare_sim_cos = top_n_cos[1]
    compare_sim_cos = (compare_sim_cos[0], to_01_scale(compare_sim_cos[1]))
    print(compare_sim_cos)
    # rand_proj_delta_last = 1.0
    for hash_len in range(1, 16):
        num_projections = hash_len * 64

        hashes = np.zeros(dtype=np.int64,
                          shape=(rows, hash_len))
        projections = create_projections(vectors, num_projections)
        hashes = train(hashes, projections, vectors)

        # Cos and LSH sim of most similar should converge
        top_n_lsh = lsh_nearest_neighbors(hashes, 0, n=n)
        compare_rand_proj = None
        DOC_ID = 0
        for i in range(0, n):
            if top_n_lsh[i][DOC_ID] == compare_sim_cos[DOC_ID]:
                compare_rand_proj = top_n_lsh[i]
                break
        # delta = abs(compare_rand_proj[1] - compare_sim_cos[1])
        # assert delta < rand_proj_delta_last
        print(compare_sim_cos, compare_rand_proj)


def test_rand_projection_works_on_high_dims():
    # np.random.seed(0)
    # random.seed(0)

    hash_len = 128
    dims = 768
    rows = 1000
    hashes = np.zeros(dtype=np.int64,
                      shape=(rows, hash_len))
    vectors = random_normed_matrix(rows, dims=dims)

    num_projections = hash_len * 64
    print(num_projections)
    projections = create_projections(vectors, num_projections)
    train(hashes, projections, vectors)
    print("Trained")

    n = 100
    top_n_lsh = lsh_nearest_neighbors(hashes, 0, n=n)
    top_n_cos = exact_nearest_neighbors(vectors[0], vectors, n=n)
    recall = len(set(keys(top_n_cos)) & set(keys(top_n_lsh))) / n
    assert recall >= 0.3


def test_clear_bits():
    one_hash = np.array([-70374112886785, 17])
    after_cleared_hash = np.array([-70374112886785, 1])
    idx = 68

    hash_after = clear_bit(one_hash, idx)
    assert (after_cleared_hash == hash_after).all()


def test_clear_bits_doesnt_change_if_already_zero():
    one_hash = np.array([-70374112886785, 1])
    hash_before = one_hash.copy()
    for idx in range(65, 127):
        hash_after = clear_bit(one_hash, idx)
        assert (hash_before == hash_after).all()


def test_set_clear_bits_doesnt_change_value():
    one_hash = np.array([-70374112886785, 0])
    hash_before = one_hash.copy()
    for idx in range(65, 127):
        hash_after_set = set_bit(one_hash, idx).copy()
        hash_after_clear = clear_bit(one_hash, idx).copy()
        assert (hash_before == hash_after_clear).all()
        assert hash_after_set[1] != hash_after_clear[1]
        assert hash_after_set[0] == hash_after_clear[0]
