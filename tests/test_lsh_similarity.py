import numpy as np
from time import perf_counter
import pytest

from utils import random_normed_matrix
from similarity import exact_nearest_neighbors, \
    keys, get_top_n


INT64_MAX = np.iinfo(np.int64).max


m1 = np.int64(0x5555555555555555)
m2 = np.int64(0x3333333333333333)
m3 = np.int64(0x0F0F0F0F0F0F0F0F)
m4 = np.int64(0x0101010101010101)


# Modified from
# https://stackoverflow.com/a/68943135/8123
def bit_count64(arr):
    # Make the values type-agnostic (as long as it's integers)
    assert arr.dtype == np.int64
    mask = np.int64(-1)
    # TODO - precompute type specific hashes
    s55 = np.int64(m1 & mask)  # Add more digits for 128bit support
    s33 = np.int64(m2 & mask)
    s0F = np.int64(m3 & mask)
    s01 = np.int64(m4 & mask)

    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return (arr * s01) >> (8 * (arr.itemsize - 1))


def jaccard_sim(hashes, comp_keys, key):
    """How many shared bits are set relative to total number."""
    assert hashes.dtype == np.int64
    numer = np.bitwise_and(hashes[comp_keys], hashes[key])
    denom = np.bitwise_or(hashes[comp_keys], hashes[key])
    num_anded = bit_count64(numer).sum(axis=1)
    num_ored = bit_count64(denom).sum(axis=1)
    return num_anded / num_ored


def lsh_nearest_neighbors(hashes, key, n=10):
    sim = hamming_sim_xor(hashes, slice(0, len(hashes)), key)
    return get_top_n(sim, n=n)


def hamming_sim_naive(hashes, comp_keys, key):
    """How identical are the bitmasks."""
    assert hashes.dtype == np.int64
    shared_ones = bit_count64(np.bitwise_and(hashes[comp_keys], hashes[key]))
    shared_zeros = bit_count64(
        np.bitwise_and(~hashes[comp_keys], ~hashes[key])
    )
    sim = (shared_ones + shared_zeros) / 64

    sim = np.sum(sim, axis=1) / sim.shape[1]
    return sim


def hamming_sim_xor(hashes, comp_keys, key):
    xord = np.bitwise_xor(hashes[comp_keys], hashes[key])
    num_shared_bits = bit_count64(~xord)

    xor_sim = np.sum(num_shared_bits / 64,
                     axis=1) / num_shared_bits.shape[1]
    return xor_sim


def random_mask_of_n_bits(num_bits) -> np.int64:
    """Random mask up to 64 bits long."""
    shift_by = min(num_bits, 64)
    overlap_at = np.random.randint(65 - shift_by)

    # zero lower shift_by bits, set in dest
    mask = -1 << shift_by

    # Shift up to some random spot in the 64 bits
    # filling mask with ones
    incr_by = (2 ** overlap_at) - 1
    mask <<= overlap_at
    mask += incr_by
    mask = ~mask

    mask = np.int64(np.uint64(mask))

    return mask


def transplant_bits(hashes: np.ndarray, src: int, dest: np.ndarray,
                    num_to_change: int, hash_to_modify: int,
                    share=False):
    """ Share or unshare num_to_change bits from src -> dest
        in hashes."""
    assert src not in dest

    if num_to_change == 0:
        return hashes

    mask = random_mask_of_n_bits(num_to_change)

    if num_to_change == 64:
        assert mask == -1

    hashes[dest, hash_to_modify] &= ~mask  # Clear shared bits
    to_assign = mask & hashes[src, hash_to_modify]
    if not share:
        to_assign = mask & ~hashes[src, hash_to_modify]

    hashes[dest, hash_to_modify] |= to_assign

    return hashes


def unshare_bits(hashes, src, dest, num_to_change, hash_len):
    hash_to_modify = np.random.randint(hash_len)
    return transplant_bits(hashes, src, dest,
                           num_to_change, hash_to_modify,
                           share=False)


def share_bits(hashes, src, dest, num_to_change, hash_len):
    hash_to_modify = np.random.randint(hash_len)
    return transplant_bits(hashes, src, dest,
                           num_to_change, hash_to_modify,
                           share=True)


def choose_flips(hashes, vectors, src, sim_floor, learn_rate):
    """Pick how many bits should be flipped in hashes to approximate
       cosine similarity."""
    # These dot products could be cached
    vect = vectors[src]
    dotted = np.dot(vectors, vect)

    # dedup
    comp_keys = np.array(range(0, len(vectors)))
    comp_scores = (dotted + 1) / 2
    assert (comp_scores <= 1.01).all()
    assert (comp_scores >= -0.01).all()

    hash_len = hashes.shape[1]
    total_bits = (hash_len * 64)
    bit_sim = hamming_sim_xor(hashes, comp_keys, src)
    sim_diff = (comp_scores - bit_sim)
    print(f" >>  CS - {comp_scores}")
    print(f" >>  BS - {bit_sim}")
    print(f" >> SDF - {sim_diff}")
    bit_flips = np.int64(
        sim_diff * total_bits
    )
    # We don't care when the similarity is too far from the target,
    # in fact its pretty sub optimal to try to make these similarities
    # exact, because it uses up valuable information
    dont_move_up = comp_scores < sim_floor
    bit_flips[dont_move_up & (bit_sim < sim_floor)] = 0

    # Apply a learning rate, but with a floor of 1 bit flip
    bit_flips[bit_flips > 0] = np.ceil(learn_rate * bit_flips[bit_flips > 0])
    bit_flips[bit_flips < 0] = np.floor(learn_rate * bit_flips[bit_flips < 0])
    print(f" >>  UP - {len(bit_flips[bit_flips > 0])}")
    print(f" >>  DN - {len(bit_flips[bit_flips < 0])}")
    return bit_flips


def train_one(hashes, vectors, src, learn_rate=0.1, sim_floor=0.0):
    """ Modify hashes to be closer / farther from hashes[key] using
        'vector'."""

    comp_keys = np.array(range(0, len(vectors)))  # dup, cleanup
    bit_flips = choose_flips(hashes, vectors, src,
                             sim_floor, learn_rate)
    hash_len = hashes.shape[1]

    to_share = bit_flips[bit_flips > 0]
    to_unshare = bit_flips[bit_flips < 0]

    print(f">> {bit_flips}")
    if len(to_unshare) == 0 and len(to_share) == 0:
        return hashes, True

    if len(to_unshare) > 0:
        num_to_unshare = -np.max(to_unshare)
        num_to_unshare_min = -np.min(to_unshare)
        keys_to_unshare = comp_keys[bit_flips < 0]
        assert keys not in keys_to_unshare
        bit_sim_before = hamming_sim_xor(hashes,
                                         keys_to_unshare,
                                         src)
        assert num_to_unshare > 0
        # print("------------")
        print(f">> {src} - Unsharing {num_to_unshare} bits "
              f"/ {num_to_unshare_min}-{num_to_unshare} for {keys_to_unshare}")
        hashes = unshare_bits(hashes, src, keys_to_unshare,
                              num_to_unshare, hash_len)
        # print("------------")
        bit_sim_after = hamming_sim_xor(hashes,
                                        keys_to_unshare,
                                        src)
        assert (bit_sim_after <= bit_sim_before).all()
    if len(to_share) > 0:
        num_to_share = np.min(to_share)
        assert num_to_share > 0
        keys_to_share = comp_keys[bit_flips > 0]
        assert keys not in keys_to_share
        bit_sim_before = hamming_sim_xor(hashes,
                                         keys_to_share,
                                         src)
        print(f">> {src} -   Sharing {num_to_share} bits for {keys_to_share}")
        hashes = share_bits(hashes, src, keys_to_share,
                            num_to_share, hash_len)
        bit_sim_after = hamming_sim_xor(hashes,
                                        keys_to_share,
                                        src)
        assert (bit_sim_after >= bit_sim_before).all()
        # print(f"Shared {num_to_share} bits / {len(to_share)}")

    return hashes, False


def run_lsh_scenario(rows, dims, hash_len,
                     rounds, eval_at, train_keys=[0]):
    """Run lsh scenario with optimizing to a single target."""
    vectors = random_normed_matrix(rows, dims=dims)
    hashes = np.random.randint(INT64_MAX - 1,
                               dtype=np.int64,
                               size=(len(vectors),
                                     hash_len))

    sim_floors = {}

    last_recall = 0.0
    n = eval_at
    start = perf_counter()
    rounds_took = 0
    completes = [False] * len(train_keys)
    for i in range(rounds):
        key = train_keys[i % len(train_keys)]
        if np.array(completes).all():
            break

        try:
            sim_floor = sim_floors[key]
        except KeyError:
            exact = exact_nearest_neighbors(vectors[key], vectors, n=10)
            sim_floors[key] = (exact[-1][1] + 1) / 2
            sim_floor = sim_floors[key]

        print("---")
        print(f"{i} - {key} - {sim_floor}")
        print(lsh_nearest_neighbors(hashes, key, n=10))
        hashes, complete = train_one(hashes, vectors, key,
                                     learn_rate=0.1, sim_floor=sim_floor)

        top_n_lsh = lsh_nearest_neighbors(hashes, key, n=n)
        top_n_nn = exact_nearest_neighbors(vectors[key], vectors, n=n)
        recall = len(set(keys(top_n_nn)) & set(keys(top_n_lsh))) / n
        delta_recall = recall - last_recall
        print(f"RECALL@{eval_at} - {recall}, {delta_recall}")
        print(f"  PERF   - {perf_counter() - start}")
        print(lsh_nearest_neighbors(hashes, key, n=10))
        last_recall = recall
        print("---")
        completes[key] = complete

        rounds_took = i
    print("FINAL")
    recalls = []
    for key in train_keys:
        top_n_lsh = lsh_nearest_neighbors(hashes, key, n=n)
        top_n_nn = exact_nearest_neighbors(vectors[key], vectors, n=n)
        recall = len(set(keys(top_n_nn)) & set(keys(top_n_lsh))) / n
        print(f"RECALL@{eval_at} - {recall}")
        recalls.append(recall)
        exact = exact_nearest_neighbors(vectors[key], vectors, n=10)
        exact = [(idx, (score + 1) / 2) for idx, score in exact]
        print(f" LS {lsh_nearest_neighbors(hashes, key, n=10)}")
        print(f" GT {exact}")
    print(f"  PERF   - {perf_counter() - start}")
    return recalls, rounds_took + 1


def test_lsh_one_large_converges():
    rounds = 10000
    recalls, rounds_took = run_lsh_scenario(rows=100000, dims=768,
                                            hash_len=32,
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
    rounds = 2000
    recalls, rounds_took = run_lsh_scenario(rows=100000,
                                            dims=768,
                                            hash_len=32,
                                            rounds=rounds,
                                            eval_at=10,
                                            train_keys=list(range(0, 10)))
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

    bit_sim_before_round = hamming_sim_xor(hashes,
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
        bit_sim_before = hamming_sim_xor(hashes_before,
                                         changed_non_key_hashes,
                                         key)
        bit_sim_after = hamming_sim_xor(hashes,
                                        changed_non_key_hashes,
                                        key)
        if (bit_sim_after < bit_sim_before).any():
            shrinks += 1
        elif (bit_sim_after == bit_sim_before).all():
            same += 1

    bit_sim_after_round = hamming_sim_xor(hashes,
                                          unshare_with,
                                          key)
    assert shrinks > same
    assert (bit_sim_after_round < bit_sim_before_round).all()


def test_xor_sim_matches_naive_hamming_sim():
    hash_len = 8
    for i in range(0, 100):
        hashes = np.random.randint(INT64_MAX - 1,
                                   dtype=np.int64,
                                   size=(10,
                                         hash_len))
        to_compare = list(range(1, 7))
        xor_sim = hamming_sim_xor(hashes, to_compare, 0)

        naive_sim = hamming_sim_naive(hashes, to_compare, 0)
        assert (xor_sim == naive_sim).all()


def test_xor_sim_faster_than_naive_hamming_sim():
    hash_len = 8
    xor_time = 0
    naive_time = 0
    for i in range(0, 100):
        hashes = np.random.randint(INT64_MAX - 1,
                                   dtype=np.int64,
                                   size=(10,
                                         hash_len))
        to_compare = list(range(1, 7))
        start = perf_counter()
        hamming_sim_xor(hashes, to_compare, 0)
        stop = perf_counter()
        print(f"\nxor - {perf_counter() - start}")
        xor_time += (stop - start)

        start = perf_counter()
        hamming_sim_naive(hashes, to_compare, 0)
        stop = perf_counter()
        print(f"nai - {perf_counter() - start}")
        naive_time += (stop - start)
    assert xor_time < naive_time
    print(naive_time, xor_time)


def test_hamming_distance_zero_to_positive_one():

    hashes = np.array([np.array([np.int64(0b0)]),
                       np.array([np.int64(-0b1)])])
    sim = hamming_sim_xor(hashes, [0, 1], 0)
    assert (sim == [1.0, 0.0]).all()
    sim = hamming_sim_xor(hashes, [0, 1], 1)
    assert (sim == [0.0, 1.0]).all()


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

    sim = hamming_sim_xor(hashes, [0, 1], key)

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

    sim = hamming_sim_xor(hashes, [0, 1], src)

    assert sim[0] == 1.0
    assert sim[1] == 1.0


def test_choose_flips_chooses_all_bits_with_learn_rate_1():
    vectors = np.array([np.array([-0.5, -0.5]),
                        np.array([-0.5, -0.5])])

    hashes = np.array([np.array([np.int64(0), np.int64(0)]),
                       np.array([np.int64(-1), np.int64(-1)])])

    src = 0
    bit_flips = choose_flips(hashes, vectors, src,
                             sim_floor=0.0, learn_rate=1.0)
    assert np.sum(np.abs(bit_flips)) == 128


def test_choose_flips_chooses_learn_rate_controlled_bits():
    vectors = np.array([np.array([-0.5, -0.5]),
                        np.array([-0.5, -0.5])])

    learn_rate = 0.2

    hashes = np.array([np.array([np.int64(0), np.int64(0)]),
                       np.array([np.int64(-1), np.int64(-1)])])

    src = 0
    bit_flips = choose_flips(hashes, vectors, src,
                             sim_floor=0.0, learn_rate=learn_rate)
    assert np.sum(np.abs(bit_flips)) < (128 * 0.3)
    assert np.sum(np.abs(bit_flips)) > (128 * 0.1)


def test_choose_flips_chooses_no_bits_if_identical():
    vectors = np.array([np.array([0.707, 0.707]),
                        np.array([0.707, 0.707])])

    hashes = np.array([np.array([np.int64(1), np.int64(0)]),
                       np.array([np.int64(1), np.int64(0)])])

    hamming = hamming_sim_xor(hashes, [0, 1], 0)
    cosine = np.dot(vectors, vectors[0])
    np.testing.assert_allclose(hamming, cosine, rtol=0.001)

    src = 0
    bit_flips = choose_flips(hashes, vectors, src,
                             sim_floor=0.0, learn_rate=1.0)
    assert np.sum(np.abs(bit_flips)) == 0


def test_choose_flips_doesnt_flip_when_cosine_below_floor():
    vectors = np.array([np.array([0.707, 0.707]),
                        np.array([0.8, 0.6]),
                        np.array([-0.707, -0.707])])

    hashes = np.array([np.array([np.int64(0), np.int64(0)]),
                       np.array([np.int64(-1), np.int64(-1)]),
                       np.array([np.int64(-1), np.int64(0)])])

    src = 0
    bit_flips = choose_flips(hashes, vectors, src,
                             sim_floor=0.9, learn_rate=1.0)

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
    bit_flips = choose_flips(hashes, vectors, src,
                             sim_floor=0.9, learn_rate=1.0)

    assert bit_flips[0] == 0
    assert bit_flips[1] != 0
    assert bit_flips[2] != 0
