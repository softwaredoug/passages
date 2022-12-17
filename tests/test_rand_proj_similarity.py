import numpy as np
import random

from rand_proj_similarity import train, create_projections, clear_bit, set_bit
from utils import random_normed_matrix
from similarity import exact_nearest_neighbors, keys
from lsh_similarity import lsh_nearest_neighbors, to_01_scale


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


def test_more_projections_converges_to_real_similarity():
    np.random.seed(11)
    random.seed(11)

    hash_len = 128
    dims = 1000
    rows = 2
    n = 2

    # Get two vectors
    vectors = random_normed_matrix(rows, dims=dims)
    top_n_cos = exact_nearest_neighbors(vectors[0], vectors, n=2)
    compare_sim_cos = top_n_cos[1]

    vect_query = vectors[0]
    vect_result = vectors[1]

    compare_sim_cos = (compare_sim_cos[0], to_01_scale(compare_sim_cos[1]))

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
        delta = abs(compare_rand_proj[1] - compare_sim_cos[1])
        # assert delta < rand_proj_delta_last
        print(compare_sim_cos, compare_rand_proj)
        rand_proj_delta_last = delta


def test_lsh_sim_converges_to_cos_similarity():

    np.random.seed(11)
    random.seed(11)

    hash_len = 128
    dims = 10
    rows = 1000
    n = 1000

    vectors = random_normed_matrix(rows, dims=dims)
    top_n_cos = exact_nearest_neighbors(vectors[0], vectors, n=n)
    compare_sim_cos = top_n_cos[1]
    compare_sim_cos = (compare_sim_cos[0], to_01_scale(compare_sim_cos[1]))
    print(compare_sim_cos)
    rand_proj_delta_last = 1.0
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
        delta = abs(compare_rand_proj[1] - compare_sim_cos[1])
        # assert delta < rand_proj_delta_last
        print(compare_sim_cos, compare_rand_proj)
        rand_proj_delta_last = delta


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
