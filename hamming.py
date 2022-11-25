import numpy as np

INT64_MAX = np.iinfo(np.int64).max


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
# Apparently there's a built in CPU instruction 'popcount'
# https://github.com/numpy/numpy/issues/16325
# https://github.com/numpy/numpy/pull/21429/files
def bit_count64(arr):
    # Make the values type-agnostic (as long as it's integers)
    assert arr.dtype == np.int64
    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return (arr * s01) >> (8 * (num_bytes_64 - 1))


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
    xord = np.bitwise_xor(hashes[comp_keys], hashes[key])

    # counting bits is the slowest part of this code
    num_shared_bits = bit_count64(~xord)

    xor_sim = np.sum(num_shared_bits / (64 * num_shared_bits.shape[1]),
                     axis=1)
    return xor_sim


hamming_sim = hamming_sim_xor
