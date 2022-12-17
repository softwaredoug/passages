import numpy as np
from time import perf_counter

from hamming import hamming_sim, hamming_sim_naive, bit_count64

INT64_MAX = np.iinfo(np.int64).max


def test_xor_sim_matches_naive_hamming_sim():
    hash_len = 8
    for i in range(0, 100):
        hashes = np.random.randint(INT64_MAX - 1,
                                   dtype=np.int64,
                                   size=(10,
                                         hash_len))
        to_compare = list(range(1, 7))
        xor_sim = hamming_sim(hashes, to_compare, 0)

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
        hamming_sim(hashes, to_compare, 0)
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
    sim = hamming_sim(hashes, [0, 1], 0)
    assert (sim == [1.0, 0.0]).all()
    sim = hamming_sim(hashes, [0, 1], 1)
    assert (sim == [0.0, 1.0]).all()


def test_hamming_distance_same_twice():

    hashes = np.array([np.array([np.int64(-0b1111)]),
                       np.array([np.int64(0b0111)])])
    hashes_before = hashes.copy()
    sim_before = hamming_sim(hashes, [0, 1], 0)
    sim_after = hamming_sim(hashes, [0, 1], 0)
    assert (sim_before == sim_after).all()
    assert (hashes == hashes_before).all()


def test_bitcount_ones():
    hashes = np.array([np.int64(0b1),
                       np.int64(0b1)])
    bc = bit_count64(hashes)
    assert (bc == np.array([1, 1])).all()


def test_bitcount_big_uint():    # bits 0444444423233401
    hashes = np.array([np.int64(0x0fffffffabcdef01)])
    bc = bit_count64(hashes)
    assert (bc == np.array([46])).all()


def test_bitcount_negative_one():
    hashes = np.array([np.int64(-0b1)])
    bc = bit_count64(hashes)
    assert (bc == np.array([64])).all()
