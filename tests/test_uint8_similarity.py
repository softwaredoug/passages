import numpy as np
import pytest
import warnings
from time import perf_counter
from uint8_similarity import quantize_idx, \
    quantize_query, log_encode
from similarity import exact_nearest_neighbors, get_top_n, keys, scores


def random_normed_matrix(rows, dims=768):
    """Random normalized matrix ranging from -1 to 1."""
    matrix = np.random.random_sample(size=(rows, dims))
    matrix = (matrix * 2.0) - 1.0
    norms = np.linalg.norm(matrix, axis=1).reshape((rows, 1))
    matrix = matrix / norms

    return matrix, quantize_idx(matrix)


def uint8_log_sim(vect1, vect2, norm=False):
    # Summing the 2^log of each component ~ cosine similarity
    #
    #   s * log(a) + s * log(b)
    summed = np.sum([vect1, vect2], dtype=np.uint8, axis=0)
    summed_biggest = (summed[(summed >= 16)] - 16)
    #   >= s * 8  <- middle
    summed_big = (summed[(summed >= 8) & (summed < 16)]) - 8
    #    < s * 8 <- lower bits
    summed_small = (summed[summed < 8])

    # 2 ** (sb - 8)
    sim = np.sum(2 ** summed_biggest, dtype=np.uint64) * 65536  # (2 ** 16)
    sim += np.sum(2 ** summed_big, dtype=np.uint64) * 256  # (2 ** 8)
    sim += np.sum(2 ** summed_small, dtype=np.uint64)
    return sim


def uint8_log_sum(arr: np.ndarray) -> np.uint64:
    summed_big = arr[arr >= 8] - 8
    summed_small = arr[arr < 8]
    sim = np.sum(2 ** summed_big, dtype=np.uint64) * np.uint64(256)  # (2 ** 8)
    sim += np.sum(2 ** summed_small, dtype=np.uint64)
    return sim


def uint8_dot_prod(vect1, vect2) -> np.uint64:
    return np.sum(np.multiply(vect1, vect2, dtype=np.uint64))


UINT64_MAX = np.iinfo(np.uint64).max


def uint8_log_diff_sim(vect1_log, vect2_log):
    """ Taking the difference of the log of the quantized."""
    # We want to recover a - b from log scale (we have log(a) and log(b))
    # log(a/b) = log(a) - log(b)       # 2  5-4 != 4-3
    # 2 ^ (log(a) - log(b)) = a/b
    #
    # But we want a/b -> how many powers of two between a and b
    #  a/b is how many powers of two on top of b to get to a
    #
    # 2^log(a/b) = a - b
    # 2^loga(b)  = a - b
    #    loga(b) = log(b) / log(a)
    diff = vect1_log.astype(np.int8) - vect2_log.astype(np.int8)
    smallest = np.min([vect1_log, vect2_log], axis=0)
    biggest = np.max([vect1_log, vect2_log], axis=0)
    diff = np.abs(diff)

    # Approximate difference in log scale
    # Large log diffs means the diff approaches the maximum
    # since we really care about the closest (most similar) we
    # could just take the maiximum in these cases
    diff_is_smallest = smallest[np.where(diff == 1)[0]]
    diff_is_biggest = biggest[np.where(diff > 1)[0]]

    distance = uint8_log_sum(diff_is_smallest) + uint8_log_sum(diff_is_biggest)
    return UINT64_MAX - distance


def cos_sim_nn(query_vector, matrix, n=100):

    def top_n_to_query_vector(arr):
        return uint8_dot_prod(arr, query_vector)

    ranked = np.apply_along_axis(top_n_to_query_vector, 1, matrix)
    return get_top_n(ranked, n=n)


def diff_sim_nearest_neighbors(query_vector, matrix, n=100):
    def top_n_to_query_vector(arr):
        return uint8_log_diff_sim(arr, query_vector)

    ranked = np.apply_along_axis(top_n_to_query_vector, 0, matrix)
    return get_top_n(ranked)


def uint8_sim(vect1, vect2, norm=False):
    # This is more of a manhattan distance
    diff = 1 + np.max([vect1, vect2], axis=0) - np.min([vect1, vect2], axis=0)
    return np.sum(-diff, dtype=np.uint64)


def cosine_sim(vect1, vect2):
    dotted = np.dot(vect1.astype(np.float64), vect2.astype(np.float64))
    return dotted


def random_norm_vect(size=10):
    vect = np.random.random_sample(size=size)
    return vect / np.linalg.norm(vect)


def random_norm_uint_vect(size=10):
    vect = 256 * random_norm_vect(size=size)
    vect = vect.astype(np.uint8)
    return vect


def test_uint8_log_sim_self_similarities_close():
    sim = uint8_log_diff_sim

    # Known tricky cases
    vect1 = np.array([38,  16,  66,  70, 104,
                      71,  54,  51, 109, 144], dtype=np.uint8)
    vect2 = np.array([112,  28,  82,  89,
                      77, 119,  30, 117,  26,  37], dtype=np.uint8)

    vect1_log = log_encode(vect1, dtype=np.uint8)
    vect2_log = log_encode(vect2, dtype=np.uint8)
    assert sim(vect1_log, vect2_log) < sim(vect1_log, vect1_log)
    assert sim(vect1_log, vect2_log) < sim(vect2_log, vect2_log)


def test_uint8_cos_sim_self_similarities_close():
    sim = uint8_dot_prod

    # Known tricky cases
    vect1 = np.array([38,  16,  66,  70, 104,
                      71,  54,  51, 109, 144], dtype=np.uint8)
    vect2 = np.array([112,  28,  82,  89,
                      77, 119,  30, 117,  26,  37], dtype=np.uint8)

    assert sim(vect1, vect2) <= sim(vect1, vect1)
    assert sim(vect1, vect2) <= sim(vect2, vect2)


def test_uint8_log_sim_communative():
    sim = uint8_log_diff_sim

    vect1 = np.array(0)
    vect2 = np.array(0)
    for _ in range(0, 100):

        while (vect1 == vect2).all():
            vect1 = np.random.randint(0, 255, size=10, dtype=np.uint8)
            vect2 = np.random.randint(0, 255, size=10, dtype=np.uint8)

        assert sim(vect1, vect2) == sim(vect2, vect1)


def test_uint8_cos_sim_communative():
    sim = uint8_dot_prod

    vect1 = np.array(0)
    vect2 = np.array(0)
    for _ in range(0, 100):

        while (vect1 == vect2).all():
            vect1 = np.random.randint(0, 255, size=10, dtype=np.uint8)
            vect2 = np.random.randint(0, 255, size=10, dtype=np.uint8)

        assert sim(vect1, vect2) == sim(vect2, vect1)


@pytest.mark.skip("Uint8 logarithmic encoding is experimental")
def test_cos_similarity_correlates_with_uint8_log_sim():
    sim = uint8_log_diff_sim

    uint_encode = log_encode
    in_order = 0
    out_of_order = 0
    for idx in range(0, 100):
        vect1 = np.array(0)
        vect2 = np.array(0)
        print(idx)

        while (vect1 == vect2).all():
            vect1 = random_norm_uint_vect(size=10)
            vect1_log = uint_encode(vect1)
            vect2 = random_norm_uint_vect(size=10)
            vect2_log = uint_encode(vect2)
            vect3 = random_norm_uint_vect(size=10)
            vect3_log = uint_encode(vect3)

        assert cosine_sim(vect1, vect1) == pytest.approx(255.0*255.0, abs=1000)
        assert cosine_sim(vect1, vect2) < cosine_sim(vect1, vect1)
        assert cosine_sim(vect1, vect2) < cosine_sim(vect2, vect2)

        assert sim(vect1_log, vect2_log) < sim(vect1_log, vect1_log)
        assert sim(vect1_log, vect2_log) < sim(vect2_log, vect2_log)

        cos_sim_order = np.array(sorted([vect1, vect2, vect3],
                                 key=lambda vect: cosine_sim(vect, vect3)))
        uint_sim_order = np.array(sorted([vect1_log, vect2_log, vect3_log],
                                  key=lambda vect: sim(vect, vect3_log)))

        vect2_first = (cos_sim_order[0] == vect2).all()
        vect1_first = (cos_sim_order[0] == vect1).all()

        correct_sort = False
        if vect2_first:
            correct_sort = (uint_sim_order[0] == vect2_log).all()
        elif vect1_first:
            correct_sort = (uint_sim_order[0] == vect1_log).all()

        if not correct_sort:
            cos23 = cosine_sim(vect2, vect3)
            cos13 = cosine_sim(vect1, vect3)
            cos_delta = abs(cos23 - cos13)
            uint23 = sim(vect2_log, vect3_log)
            uint13 = sim(vect1_log, vect3_log)
            uint_delta = abs(int(uint23) - int(uint13))

            # diff13 = (1 + np.max([vect1, vect3], axis=0)
            #           - np.min([vect1, vect3], axis=0))
            # diff23 = (1 + np.max([vect2, vect3], axis=0)
            #           - np.min([vect2, vect3], axis=0))

            # mult13 = np.multiply(vect1, vect3, dtype=np.uint32)
            # mult23 = np.multiply(vect2, vect3, dtype=np.uint32)

            print(f"delta({cos23},{cos13})={cos_delta},"
                  f"delta({uint23}, {uint13})={uint_delta}")
            out_of_order += 1
        else:
            in_order += 1
    print(in_order, out_of_order)
    assert out_of_order == 0


# @pytest.mark.skip("Uint8 encoding is experimental")
def test_cos_similarity_correlates_with_uint8_cos_sim():
    sim = uint8_dot_prod

    in_order = 0
    out_of_order = 0
    for idx in range(0, 100):
        vect1 = np.array(0)
        vect2 = np.array(0)
        print(idx)

        while (vect1 == vect2).all():
            vect1 = random_norm_uint_vect(size=10)
            vect2 = random_norm_uint_vect(size=10)
            vect3 = random_norm_uint_vect(size=10)

        assert cosine_sim(vect1, vect1) == pytest.approx(255.0*255.0, abs=1000)
        assert cosine_sim(vect1, vect2) < cosine_sim(vect1, vect1)
        assert cosine_sim(vect1, vect2) < cosine_sim(vect2, vect2)

        assert sim(vect1, vect2) < sim(vect1, vect1)
        assert sim(vect1, vect2) < sim(vect2, vect2)

        cos_sim_order = np.array(sorted([vect1, vect2, vect3],
                                 key=lambda vect: cosine_sim(vect, vect3)))
        uint_sim_order = np.array(sorted([vect1, vect2, vect3],
                                  key=lambda vect: sim(vect, vect3)))
        if not ((cos_sim_order == uint_sim_order).all().all()):
            cos23 = cosine_sim(vect2, vect3)
            cos13 = cosine_sim(vect1, vect3)
            cos_delta = abs(cos23 - cos13)
            uint23 = sim(vect2, vect3)
            uint13 = sim(vect1, vect3)
            uint_delta = abs(int(uint23) - int(uint13))

            # diff13 = (1 + np.max([vect1, vect3], axis=0)
            #           - np.min([vect1, vect3], axis=0))
            # diff23 = (1 + np.max([vect2, vect3], axis=0)
            #           - np.min([vect2, vect3], axis=0))

            # mult13 = np.multiply(vect1, vect3, dtype=np.uint32)
            # mult23 = np.multiply(vect2, vect3, dtype=np.uint32)

            print(f"delta({cos23},{cos13})={cos_delta},"
                  f"delta({uint23}, {uint13})={uint_delta}")
            out_of_order += 1
        else:
            in_order += 1
    print(in_order, out_of_order)
    assert out_of_order == 0


@pytest.mark.skip("Uint8 encoding is experimental... plus this is slow")
def test_cos_similarity_uint8_sim_recall():

    rows = 1000000
    recall_at = 10
    dims = 10
    flt_matrix, uint_matrix = random_normed_matrix(rows=rows,
                                                   dims=dims)

    for _ in range(0, 100):
        query_vector_flt = random_norm_vect(size=dims)
        query_vector_uint8 = quantize_query(query_vector_flt)

        start = perf_counter()
        top_n_flt = exact_nearest_neighbors(query_vector_flt,
                                            flt_matrix,
                                            n=recall_at)
        print(f"Exact took {perf_counter() - start}")

        start = perf_counter()
        top_n_uint8 = cos_sim_nn(query_vector_uint8,
                                 uint_matrix,
                                 n=recall_at)
        print(f"Uint8 took {perf_counter() - start}")

        intersection = keys(top_n_flt) & keys(top_n_uint8)
        recall = len(intersection) / recall_at
        print(recall)


def test_uint8_cos_similarity_of_matrix():
    # nn = # diff_sim_nearest_neighbors
    nn = cos_sim_nn
    for idx in range(0, 100):
        vect1 = np.array(0)
        vect2 = np.array(0)

        while (vect1 == vect2).all():
            vect1 = random_norm_uint_vect(size=2)
            vect2 = random_norm_uint_vect(size=2)
            matrix = np.array([vect1, vect2], dtype=np.uint8)
            # matrix_log = log_encode(matrix, dtype=np.int8)
            # vect1_log = log_encode(vect1, dtype=np.int8)
            # vect2_log = log_encode(vect2, dtype=np.int8)

            print(idx)
            top_n = nn(vect1, matrix)
            idxs = keys(top_n)
            scored = scores(top_n)
            assert scored[0] >= scored[1]
            if scored[0] > scored[1] + 255:
                assert idxs == [0, 1]
            else:
                warnings.warn("Approx dot prod differs {scores[1]-scores[0]}")

            top_n = nn(vect2, matrix)
            idxs = keys(top_n)
            scored = scores(top_n)
            assert scored[0] >= scored[1]
            if scored[0] > scored[1] + 255:
                assert idxs == [1, 0]
            else:
                warnings.warn("Approx dot prod differs {scores[1]-scores[0]}")
