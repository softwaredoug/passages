import numpy as np
from time import perf_counter
import random

from utils import random_normed_matrix
from similarity import exact_nearest_neighbors
from hamming import hamming_sim, bit_count64
from lsh_similarity import lsh_nearest_neighbors, LshSimilarity, \
    unshare_bits, transplant_bits, choose_flips, random_mask_of_n_bits, \
    to_01_scale


INT64_MAX = np.iinfo(np.int64).max


def run_lsh_scenario(rows, dims, hash_len,
                     rounds, eval_at, train_keys=[0],
                     projections=False,
                     seed=0):
    """Run lsh scenario with optimizing to a single target."""
    np.random.seed(seed)
    random.seed(seed)
    vectors = random_normed_matrix(rows, dims=dims)

    sim = LshSimilarity(hash_len, projections=projections)

    hashes, recalls, rounds_took =\
        sim.train(vectors, rounds, eval_at, train_keys)
    return recalls, rounds_took + 1


def test_lsh_one_large_converges():
    rounds = 10000
    recalls, rounds_took = run_lsh_scenario(rows=100000, dims=768,
                                            hash_len=16,
                                            rounds=rounds, eval_at=10)
    recall = recalls[0]
    assert recall >= 0.9
    assert rounds_took < rounds


def test_lsh_one_medium_converges():
    rounds = 10000
    recalls, rounds_took = run_lsh_scenario(rows=1000, dims=768,
                                            hash_len=16,
                                            rounds=rounds, eval_at=10)
    recall = recalls[0]
    assert recall >= 0.9
    assert rounds_took < rounds


def test_lsh_one_medium_converges_faster_with_projections():
    rounds = 10000
    recalls, rounds_took = run_lsh_scenario(rows=1000, dims=768,
                                            hash_len=16,
                                            rounds=rounds,
                                            eval_at=10,
                                            projections=False)
    recall = recalls[0]
    assert recall >= 0.9
    assert rounds_took < rounds
    rounds_took_no_projections = rounds_took

    recalls, rounds_took = run_lsh_scenario(rows=1000, dims=768,
                                            hash_len=16,
                                            rounds=rounds,
                                            eval_at=10,
                                            projections=True)
    assert rounds_took < rounds_took_no_projections


def test_lsh_one_small_converges():
    rounds = 10000
    recalls, rounds_took = run_lsh_scenario(rows=100, dims=4, hash_len=16,
                                            rounds=rounds, eval_at=2)
    recall = recalls[0]
    assert rounds_took < rounds
    assert recall == 1.0


def test_lsh_one_tiny_converges():
    rounds = 10000
    recalls, rounds_took = run_lsh_scenario(rows=10, dims=4, hash_len=16,
                                            rounds=rounds, eval_at=2)
    recall = recalls[0]
    assert recall == 1.0
    assert rounds_took < rounds


def test_lsh_two_small_converges():
    rounds = 1000
    recalls, rounds_took = run_lsh_scenario(rows=100, dims=4, hash_len=16,
                                            rounds=rounds,
                                            eval_at=10,
                                            train_keys=[0, 1])
    # assert rounds_took < rounds
    assert recalls[0] >= 0.9
    assert recalls[1] >= 0.9


def test_lsh_two_large_converges():
    rounds = 1000
    recalls, rounds_took = run_lsh_scenario(rows=100000,
                                            dims=768,
                                            hash_len=32,
                                            rounds=rounds,
                                            eval_at=10,
                                            train_keys=[0, 1])
    assert rounds_took < rounds
    print(recalls)
    assert recalls[0] >= 0.9
    assert recalls[1] >= 0.9


def test_lsh_ten_large_converges():
    rounds = 3000
    recalls, rounds_took = run_lsh_scenario(rows=100000,
                                            dims=768,
                                            hash_len=18,
                                            rounds=rounds,
                                            eval_at=10,
                                            train_keys=list(range(0, 10)))
    print(recalls)
    for recall in recalls:
        assert recall >= 0.9


def test_unshare_bits_makes_less_similar():
    hash_len = 8
    hashes = np.random.randint(INT64_MAX - 1,
                               dtype=np.int64,
                               size=(10,
                                     hash_len))
    unshare_with = list(range(1, 7))
    key = 0

    bit_sim_before_round = hamming_sim(hashes,
                                       unshare_with,
                                       key)

    shrinks = 0
    same = 0
    for i in range(0, 100):

        hashes_before = hashes.copy()
        hashes = unshare_bits(hashes, key, unshare_with, i % 8, hash_len)
        changed = (hashes_before != hashes).any(axis=1)
        changed_hashes = np.argwhere(changed).reshape(1, -1)[0]
        changed_non_key_hashes = changed_hashes[changed_hashes > 0]
        bit_sim_before = hamming_sim(hashes_before,
                                     changed_non_key_hashes,
                                     key)
        bit_sim_after = hamming_sim(hashes,
                                    changed_non_key_hashes,
                                    key)
        if (bit_sim_after < bit_sim_before).any():
            shrinks += 1
        elif (bit_sim_after == bit_sim_before).all():
            same += 1

    bit_sim_after_round = hamming_sim(hashes,
                                      unshare_with,
                                      key)
    assert shrinks > same
    assert (bit_sim_after_round < bit_sim_before_round).all()


def test_mask_of_size_n():
    for requested_mask_size in range(0, 126):
        expected_mask_size = min(requested_mask_size, 64)
        mask = random_mask_of_n_bits(expected_mask_size)
        assert bit_count64(mask) == expected_mask_size


def test_unshare_all_bits_makes_opposite_sim():
    hashes = np.array([np.array([np.int64(0), np.int64(0)]),
                       np.array([np.int64(0), np.int64(0)])])

    key = 0
    unshare_with = [1]

    hashes = transplant_bits(hashes,
                             key,
                             unshare_with,
                             64,
                             0,
                             share=False)

    hashes = transplant_bits(hashes,
                             key,
                             unshare_with,
                             64,
                             1,
                             share=False)

    sim = hamming_sim(hashes, [0, 1], key)

    assert sim[0] == 1.0
    assert sim[1] == 0.0


def test_share_all_bits_makes_identical_sim():
    hashes = np.array([np.array([np.int64(0), np.int64(0)]),
                       np.array([np.int64(-1), np.int64(-1)])])

    src = 0
    dest = [1]

    hashes = transplant_bits(hashes,
                             src,
                             dest,
                             64,
                             0,
                             share=True)

    hashes = transplant_bits(hashes,
                             src,
                             dest,
                             64,
                             1,
                             share=True)

    sim = hamming_sim(hashes, [0, 1], src)

    assert sim[0] == 1.0
    assert sim[1] == 1.0


def test_choose_flips_chooses_all_bits_with_learn_rate_1():
    vectors = np.array([np.array([-0.5, -0.5]),
                        np.array([-0.5, -0.5])])

    hashes = np.array([np.array([np.int64(0), np.int64(0)]),
                       np.array([np.int64(-1), np.int64(-1)])])

    src = 0
    src_dotted = to_01_scale(np.dot(vectors, vectors[src]))
    bit_flips = choose_flips(hashes, src_dotted, src,
                             lsh_floor=0.0, learn_rate=1.0)
    assert np.sum(np.abs(bit_flips)) == 128


def test_choose_flips_chooses_learn_rate_controlled_bits():
    vectors = np.array([np.array([-0.5, -0.5]),
                        np.array([-0.5, -0.5])])

    learn_rate = 0.2

    hashes = np.array([np.array([np.int64(0), np.int64(0)]),
                       np.array([np.int64(-1), np.int64(-1)])])

    src = 0
    src_dotted = to_01_scale(np.dot(vectors, vectors[src]))
    bit_flips = choose_flips(hashes, src_dotted, src,
                             lsh_floor=0.0, learn_rate=learn_rate)
    assert np.sum(np.abs(bit_flips)) < (128 * 0.3)
    assert np.sum(np.abs(bit_flips)) > (128 * 0.1)


def test_choose_flips_chooses_no_bits_if_identical():
    vectors = np.array([np.array([0.707, 0.707]),
                        np.array([0.707, 0.707])])

    hashes = np.array([np.array([np.int64(1), np.int64(0)]),
                       np.array([np.int64(1), np.int64(0)])])

    hamming = hamming_sim(hashes, [0, 1], 0)
    cosine = np.dot(vectors, vectors[0])
    np.testing.assert_allclose(hamming, cosine, rtol=0.001)

    src = 0
    src_dotted = to_01_scale(np.dot(vectors, vectors[src]))
    bit_flips = choose_flips(hashes, src_dotted, src,
                             lsh_floor=0.0, learn_rate=1.0)
    assert np.sum(np.abs(bit_flips)) == 0


def test_choose_flips_doesnt_flip_when_cosine_below_floor():
    vectors = np.array([np.array([0.707, 0.707]),
                        np.array([0.8, 0.6]),
                        np.array([-0.707, -0.707])])

    hashes = np.array([np.array([np.int64(0), np.int64(0)]),
                       np.array([np.int64(-1), np.int64(-1)]),
                       np.array([np.int64(-1), np.int64(0)])])

    src = 0
    src_dotted = to_01_scale(np.dot(vectors, vectors[src]))
    bit_flips = choose_flips(hashes, src_dotted, src,
                             lsh_floor=0.9, learn_rate=1.0)

    assert bit_flips[0] == 0
    assert bit_flips[1] > 0
    assert bit_flips[2] == 0


def test_choose_flips_when_hamming_above_floor():
    vectors = np.array([np.array([0.707, 0.707]),
                        np.array([0.8, 0.6]),
                        np.array([0.9, 0.1])])

    hashes = np.array([np.array([np.int64(1), np.int64(1)]),
                       np.array([np.int64(1), np.int64(2 ** 2)]),
                       np.array([np.int64(1), np.int64(1)])])

    src = 0
    src_dotted = to_01_scale(np.dot(vectors, vectors[src]))
    bit_flips = choose_flips(hashes, src_dotted, src,
                             lsh_floor=0.9, learn_rate=1.0)

    assert bit_flips[0] == 0
    assert bit_flips[1] != 0
    assert bit_flips[2] != 0


def test_lsh_vs_cos_perf_similar():
    rows = 100000
    hash_len = 16
    dims = 768
    vectors = random_normed_matrix(rows, dims=dims)
    hashes = np.random.randint(INT64_MAX - 1,
                               dtype=np.int64,
                               size=(len(vectors),
                                     hash_len))

    start = perf_counter()
    n = 10
    for i in range(0, 100):
        key = i % rows
        lsh_nearest_neighbors(hashes, key, n=n)

    lsh_time = (perf_counter() - start)
    print(f"LSH took {lsh_time}")

    start = perf_counter()
    for i in range(0, 100):
        key = i % rows
        exact_nearest_neighbors(vectors[key], vectors, n=n)
    cos_time = (perf_counter() - start)
    print(f"COS took {cos_time}")

    assert lsh_time < (1.3 * cos_time)
