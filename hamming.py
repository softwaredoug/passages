import numpy as np



m1 = np.int64(0x5555555555555555)
m2 = np.int64(0x3333333333333333)
m3 = np.int64(0x0F0F0F0F0F0F0F0F)
m4 = np.int64(0x0101010101010101)

mask = np.int64(-1)
# TODO - precompute type specific hashes
s55 = np.int64(m1 & mask)  # Add more digits for 128bit support
s33 = np.int64(m2 & mask)
s0F = np.int64(m3 & mask)
s01 = np.int64(m4 & mask)
num_bytes_64 = 8


# Modified from
# https://stackoverflow.com/a/68943135/8123
# https://stackoverflow.com/a/109025/8123
#
# Array copies removed as much as possible
#
# Apparently there's a built in CPU instruction 'popcount'
# https://github.com/numpy/numpy/issues/16325
# https://github.com/numpy/numpy/pull/21429/files
def bit_count64(arr):
    # Make the values type-agnostic (as long as it's integers)
    # baseline
    #
    # -arr = arr - ((arr >> 1) & s55)
    # +arr -= ((arr >> 1) & s55)
    #
    # before
    #     5106   32.987    0.006   32.987    0.006 hamming.py:27(bit_count64)
    #     5106   32.436    0.006   32.436    0.006 hamming.py:26(bit_count64)
    #     5106   35.277    0.007   35.277    0.007 hamming.py:26(bit_count64)

    # after
    #     5106   26.593    0.005   26.593    0.005 hamming.py:26(bit_count64)
    #     5106   26.308    0.005   26.308    0.005 hamming.py:28(bit_count64)
    #
    # reduce copies by subtract in place

    assert arr.dtype == np.int64

    arr -= ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)

    arr += (arr >> 4)
    arr &= s0F
    arr *= s01
    arr >>= (8 * (num_bytes_64 - 1))

    return arr


def jaccard_sim(hashes, comp_keys, key):
    """How many shared bits are set relative to total number."""
    assert hashes.dtype == np.int64
    numer = np.bitwise_and(hashes[comp_keys], hashes[key])
    denom = np.bitwise_or(hashes[comp_keys], hashes[key])
    num_anded = bit_count64(numer).sum(axis=1)
    num_ored = bit_count64(denom).sum(axis=1)
    return num_anded / num_ored


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
    # In place to prevent copies, not thread safe
    query = hashes[key].copy()
    hashes[comp_keys] ^= query

    # counting bits is the slowest part of this code
    num_shared_bits = bit_count64(~hashes[comp_keys])

    # Inverse xor
    hashes[comp_keys] ^= query
    assert (hashes[key] == query).all()

    xor_sim = np.sum(num_shared_bits / (64 * num_shared_bits.shape[1]),
                     axis=1)
    return xor_sim


def different_bits(hash1, hash2):
    set_if_diff = np.bitwise_xor(hash1, hash2)
    are_different = []
    # Not efficient
    for idx, diff in enumerate(set_if_diff):
        for bit_idx in range(0, 63):
            test_mask = 2 ** bit_idx
            if (diff & test_mask) != 0:
                are_different.append((idx * 64) + bit_idx)
    return are_different



hamming_sim = hamming_sim_xor
