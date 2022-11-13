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


def bitwise_sim_naive(hashes, comp_keys, key):
    """How identical are the bitmasks."""
    assert hashes.dtype == np.int64
    shared_ones = bit_count64(np.bitwise_and(hashes[comp_keys], hashes[key]))
    shared_zeros = bit_count64(
        np.bitwise_and(~hashes[comp_keys], ~hashes[key])
    )
    sim = (shared_ones + shared_zeros) / 64

    sim = np.sum(sim, axis=1) / sim.shape[1]
    return sim


def bitwise_sim_xor(hashes, comp_keys, key):
    xord = np.bitwise_xor(hashes[comp_keys], hashes[key])
    num_shared_bits = bit_count64(~xord)

    xor_sim = np.sum(num_shared_bits / 64,
                     axis=1) / num_shared_bits.shape[1]
    return xor_sim


def unshare_n_bits(hashes, key, others, num_to_unshare, hash_len):
    """ Unshares between 0, min(bit_flips) between keys / comp_keys."""
    assert key not in others
    int64_to_modify = np.random.randint(hash_len-1)

    # Right shift will N to zero fill key hash
    # and force additional dissimilarity with comp_keys
    shift_by = min(num_to_unshare, 62)
    overlap_at = np.random.randint(62 - shift_by)

    # zero lower shift_by bits, set in others
    mask = -1 << shift_by

    # Shift up to some random spot in the 64 bits
    # filling mask with ones
    incr_by = (2 ** overlap_at) - 1
    mask <<= overlap_at
    mask += incr_by

    print(f" Hash {int64_to_modify} - {bin(np.uint64(mask))}")

    # (Pdb) bin(np.uint64((mask << 10) + (2 ** 10 - 1)))
    hashes[key, int64_to_modify] &= mask
    hashes[others, int64_to_modify] |= ~mask

    return hashes


def set_n_bits(hashes, keys, num_to_set, hash_len):
    int64_to_modify = np.random.randint(hash_len-1)
    shift_by = min(num_to_set, 62)
    hashes[keys, int64_to_modify] |= np.int64(1 << shift_by)
    return hashes


def train_one(hashes, vectors, key, hash_len, learn_rate=0.1):
    vect = vectors[key]
    # nn = exact_nearest_neighbors(vect, vectors, 100)
    # fn = exact_farthest_neighbors(vect, vectors, 100)
    # rn = random_neighbors(vect, vectors, 100)
    dotted = np.dot(vectors, vect)

    # dedup
    comp_keys = np.array(range(0, len(vectors)))
    comp_scores = dotted
    # for key, score in nn + fn + rn:
    #    if key not in comp_keys:
    #        comp_keys.append(key)
    #        comp_scores.append(score)

    total_bits = (hash_len * 64)
    bit_sim = bitwise_sim_xor(hashes, comp_keys, key)
    bit_flips = np.int64(
        learn_rate * (comp_scores - bit_sim) * total_bits
    )

    to_share = bit_flips[bit_flips > 0]
    to_unshare = bit_flips[bit_flips < 0]

    if len(to_unshare) > 0:
        num_to_unshare = -np.max(to_unshare)
        keys_to_unshare = comp_keys[bit_flips < 0]
        assert keys not in keys_to_unshare
        bit_sim_before = bitwise_sim_xor(hashes,
                                         keys_to_unshare,
                                         key)
        assert num_to_unshare > 0
        hashes = unshare_n_bits(hashes, key, keys_to_unshare,
                                num_to_unshare, hash_len)
        print(f"Unshared {num_to_unshare} bits / {num_to_unshare}")
        bit_sim_after = bitwise_sim_xor(hashes,
                                        keys_to_unshare,
                                        key)
        assert (bit_sim_after <= bit_sim_before).all()
    if len(to_share) > 0:
        num_to_set = np.min(to_share)
        keys_to_share = comp_keys[bit_flips > 0]
        hashes = set_n_bits(hashes, keys_to_share + [key],
                            num_to_set, hash_len)
        print(f"Shared {num_to_set} bits / {len(to_share)}")
    top_n = sorted(bitwise_sim_xor(hashes, comp_keys, key), reverse=True)[:10]
    print(top_n)

    return hashes


def train(vectors, learn_rate=0.1):
    hash_len = 8
    hashes = np.random.randint(INT64_MAX - 1,
                               dtype=np.int64,
                               size=(len(vectors),
                                     hash_len))

    rounds = 3000

    for key, vect in enumerate(vectors):
        # We might want this to be random dot products, not
        # closest / farthest?
        for i in range(rounds):
            hashes = train_one(hashes, vectors, key, hash_len, learn_rate)
    return hashes


def lsh_nearest_neighbors(hashes, key, n=10):
    sim = bitwise_sim_xor(hashes, slice(0, len(hashes)), key)
    return get_top_n(sim, n=10)


def test_lsh_one():
    vectors = random_normed_matrix(1000)
    hash_len = 8
    hashes = np.random.randint(INT64_MAX - 1,
                               dtype=np.int64,
                               size=(len(vectors),
                                     hash_len))

    rounds = 10000
    for i in range(rounds):
        hashes = train_one(hashes, vectors, 0, hash_len)

    n = 10
    top_n_lsh = lsh_nearest_neighbors(hashes, 0, n=n)
    top_n_nn = exact_nearest_neighbors(vectors[0], vectors, n=n)
    recall = len(set(keys(top_n_nn)) & set(keys(top_n_lsh))) / n
    assert recall > 0.1


@pytest.mark.skip("LSH similarity is experimental")
def test_lsh():
    vectors = random_normed_matrix(1000)

    lsh = train(vectors)

    # top_n_lsh = lsh_nearest_neighbors(lsh, 0, n=10)
    # top_n_nn  = exact_nearest_neighbors(vectors[0], vectors, n=10)
    # import pdb; pdb.set_trace()


def test_unshare_bits_makes_less_similar():
    hash_len = 8
    hashes = np.random.randint(INT64_MAX - 1,
                               dtype=np.int64,
                               size=(10,
                                     hash_len))
    unshare_with = list(range(1, 7))
    key = 0

    bit_sim_before_round = bitwise_sim_xor(hashes,
                                           unshare_with,
                                           key)

    shrinks = 0
    same = 0
    for i in range(0, 100):

        hashes_before = hashes.copy()
        hashes = unshare_n_bits(hashes, key, unshare_with,
                                i % 8, hash_len)
        changed = (hashes_before != hashes).any(axis=1)
        changed_hashes = np.argwhere(changed).reshape(1, -1)[0]
        changed_non_key_hashes = changed_hashes[changed_hashes > 0]
        bit_sim_before = bitwise_sim_xor(hashes_before,
                                         changed_non_key_hashes,
                                         key)
        bit_sim_after = bitwise_sim_xor(hashes,
                                        changed_non_key_hashes,
                                        key)
        if (bit_sim_after < bit_sim_before).any():
            shrinks += 1
        elif (bit_sim_after == bit_sim_before).all():
            same += 1

    bit_sim_after_round = bitwise_sim_xor(hashes,
                                          unshare_with,
                                          key)
    assert shrinks > same
    assert (bit_sim_after_round < bit_sim_before_round).all()


def test_xor_sim_matches_naive_sim():
    hash_len = 8
    for i in range(0, 100):
        hashes = np.random.randint(INT64_MAX - 1,
                                   dtype=np.int64,
                                   size=(10,
                                         hash_len))
        to_compare = list(range(1, 7))
        xor_sim = bitwise_sim_xor(hashes, to_compare, 0)

        naive_sim = bitwise_sim_naive(hashes, to_compare, 0)
        assert (xor_sim == naive_sim).all()


def test_xor_sim_faster_than_naive_sim():
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
        bitwise_sim_xor(hashes, to_compare, 0)
        stop = perf_counter()
        print(f"\nxor - {perf_counter() - start}")
        xor_time += (stop - start)

        start = perf_counter()
        bitwise_sim_naive(hashes, to_compare, 0)
        stop = perf_counter()
        print(f"nai - {perf_counter() - start}")
        naive_time += (stop - start)
    assert xor_time < naive_time
    print(naive_time, xor_time)
