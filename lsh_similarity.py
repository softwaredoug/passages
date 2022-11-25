import numpy as np

from similarity import exact_nearest_neighbors, \
    keys, get_top_n
from hamming import hamming_sim, bit_count64
from time import perf_counter


INT64_MAX = np.iinfo(np.int64).max


def lsh_nearest_neighbors(hashes, key, n=10):
    sim = hamming_sim(hashes, slice(0, len(hashes)), key)
    return get_top_n(sim, n=n)


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
    bit_sim = hamming_sim(hashes, comp_keys, src)
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
        bit_sim_before = hamming_sim(hashes,
                                     keys_to_unshare,
                                     src)
        assert num_to_unshare > 0
        # print("------------")
        print(f">> {src} - Unsharing {num_to_unshare} bits "
              f"/ {num_to_unshare_min}-{num_to_unshare} for {keys_to_unshare}")
        hashes = unshare_bits(hashes, src, keys_to_unshare,
                              num_to_unshare, hash_len)
        # print("------------")
        bit_sim_after = hamming_sim(hashes,
                                    keys_to_unshare,
                                    src)
        assert (bit_sim_after <= bit_sim_before).all()
    if len(to_share) > 0:
        num_to_share = np.min(to_share)
        assert num_to_share > 0
        keys_to_share = comp_keys[bit_flips > 0]
        assert keys not in keys_to_share
        bit_sim_before = hamming_sim(hashes,
                                     keys_to_share,
                                     src)
        print(f">> {src} -   Sharing {num_to_share} bits for {keys_to_share}")
        hashes = share_bits(hashes, src, keys_to_share,
                            num_to_share, hash_len)
        bit_sim_after = hamming_sim(hashes,
                                    keys_to_share,
                                    src)
        assert (bit_sim_after >= bit_sim_before).all()
        # print(f"Shared {num_to_share} bits / {len(to_share)}")

    return hashes, False


def train(vectors, hash_len,
          rounds, eval_at, train_keys=[0]):
    sim_floors = {}
    hashes = np.random.randint(INT64_MAX - 1,
                               dtype=np.int64,
                               size=(len(vectors),
                                     hash_len))

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
    return hashes, recalls, rounds_took + 1
