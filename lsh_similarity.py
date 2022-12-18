import numpy as np
import numpy.typing as npt
from typing import Dict, Optional, List, Tuple

from similarity import keys, get_top_n, exact_nearest_neighbors
from rand_proj_similarity import train as train_rand_proj, create_projections
from hamming import hamming_sim
from time import perf_counter


INT64_MAX = np.iinfo(np.int64).max


def lsh_nearest_neighbors(hashes, key, n=10):
    sim = hamming_sim(hashes, slice(0, len(hashes)), key)
    return get_top_n(sim, n=n)


def random_mask_of_n_bits(num_bits: np.int64) -> np.int64:
    """Random mask up to 64 bits long."""
    shift_by = np.min([num_bits, 64])
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


def transplant_bits(hashes: npt.NDArray[np.int64],
                    src: int,
                    dest: npt.NDArray[np.int64],
                    num_to_change: np.int64,
                    hash_to_modify: int,
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


def to_01_scale(val):
    return (val + 1) / 2


class ExactVectors:
    """Given an index, can get dot products."""

    def __init__(self,
                 vectors: npt.NDArray[np.float64],
                 train_keys: List[int],
                 n=10):
        self.n = n
        self.num_vectors = len(vectors)
        self.neighbors: Dict[int, List[Tuple[int, float]]] = {}
        for key in train_keys:
            neighbors = self._nearest_neighbors(key, vectors)
            self.neighbors[key] = neighbors

    def _dot(self,
             other: int,
             vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        other_vect = vectors[other]
        dots = np.dot(vectors, other_vect)
        dots = to_01_scale(dots)
        return dots

    def _nearest_neighbors(self,
                           other: int,
                           vectors: npt.NDArray[np.float64]) \
            -> List[Tuple[int, float]]:
        dotted = self._dot(other, vectors)
        return get_top_n(dotted, n=self.n)

    def dot_prod_top(self, other: int):
        """Dot prod vector to other, return top n, others 0d"""
        nn = self.nearest_neighbors(other)
        dotted = np.zeros(shape=(self.num_vectors))
        for idx, value in nn:
            dotted[idx] = value
        return dotted

    def lsh_floor(self, other: int):
        """We ignore any similarities calculated below this when doing LSH."""
        floor = self.nearest_neighbors(other)[-1][1]
        return floor

    def nearest_neighbors(self, other: int) -> List[Tuple[int, float]]:
        """Get top n exact nearest neighbors."""
        try:
            return self.neighbors[other]
        except KeyError:
            raise KeyError("Vector NN only available for train keys")
            # nn = self._nearest_neighbors(other)
            # self.neighbors[other] = nn
            # return self.neighbors[other]


def choose_flips(hashes: np.ndarray,
                 src_dotted: np.ndarray,
                 src: int,
                 lsh_floor: np.float64,
                 learn_rate: np.float64):
    """Pick how many bits should be flipped in hashes to approximate
       cosine similarity."""
    # These dot products could be cached
    # dedup
    comp_keys = np.array(range(0, len(hashes)))
    assert (src_dotted <= 1.01).all()
    assert (src_dotted >= -0.01).all()

    hash_len = hashes.shape[1]
    total_bits = (hash_len * 64)
    bit_sim = hamming_sim(hashes, comp_keys, src)
    sim_diff = (src_dotted - bit_sim)
    # print(f" >>  CS - {comp_scores}")
    # print(f" >>  BS - {bit_sim}")
    # print(f" >> SDF - {sim_diff}")
    bit_flips = (sim_diff * total_bits).astype(np.int64)
    # We don't care when the similarity is too far from the target,
    # in fact its pretty sub optimal to try to make these similarities
    # exact, because it uses up valuable information
    dont_move_up = src_dotted < lsh_floor
    bit_flips[dont_move_up & (bit_sim < lsh_floor)] = 0

    # Apply a learning rate, but with a floor of 1 bit flip
    bit_flips[bit_flips > 0] = np.ceil(learn_rate * bit_flips[bit_flips > 0])
    bit_flips[bit_flips < 0] = np.floor(learn_rate * bit_flips[bit_flips < 0])
    # print(f" >>  UP - {len(bit_flips[bit_flips > 0])}")
    # print(f" >>  DN - {len(bit_flips[bit_flips < 0])}")
    return bit_flips


def train_one(hashes: npt.NDArray[np.int64],
              src_dotted: npt.NDArray[np.float64],
              src: int,
              lsh_floor: np.float64,
              learn_rate=0.1):
    """ Modify hashes to be closer / farther from hashes[key] using
        'vector'."""

    comp_keys = np.array(range(0, len(hashes)))  # dup, cleanup
    bit_flips = choose_flips(hashes, src_dotted, src,
                             lsh_floor, learn_rate)
    hash_len = hashes.shape[1]

    to_share = bit_flips[bit_flips > 0]
    to_unshare = bit_flips[bit_flips < 0]

    # print(f">> {bit_flips}")
    if len(to_unshare) == 0 and len(to_share) == 0:
        return hashes, True

    if len(to_unshare) > 0:
        num_to_unshare = -np.max(to_unshare)
        keys_to_unshare = comp_keys[bit_flips < 0]
        assert keys not in keys_to_unshare
        bit_sim_before = hamming_sim(hashes,
                                     keys_to_unshare,
                                     src)
        assert num_to_unshare > 0
        # print("------------")
        # print(f">> {src} - Unsharing {num_to_unshare} bits "
        #   f"/ {num_to_unshare_min}-{num_to_unshare} for {keys_to_unshare}")
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
        # print(f">> {src} - Sharing {num_to_share} bits for {keys_to_share}")
        hashes = share_bits(hashes, src, keys_to_share,
                            num_to_share, hash_len)
        bit_sim_after = hamming_sim(hashes,
                                    keys_to_share,
                                    src)
        assert (bit_sim_after >= bit_sim_before).all()
        # print(f"Shared {num_to_share} bits / {len(to_share)}")

    return hashes, False


class LshSimilarity:

    def __init__(self, hash_len: int, projections: bool = True):
        self.project_on_train = projections
        self.projections: Optional[npt.NDArray[np.float64]] = None
        self.hashes = None
        self.hash_len = hash_len

    def _init_hashes(self,
                     vectors: npt.NDArray[np.float64]) \
            -> npt.NDArray[np.int64]:

        nn = exact_nearest_neighbors(vectors[0], vectors, 10)
        print(f"EXC {nn}")

        if self.project_on_train:
            start = perf_counter()
            print("Training Projections")
            hashes = np.zeros(dtype=np.int64,
                              shape=(len(vectors),
                                     self.hash_len))
            hash_len = hashes.shape[1]
            num_projections = hash_len * 64

            if self.projections is None:
                self.projections = create_projections(vectors, num_projections)

            hashes = train_rand_proj(hashes, self.projections, vectors)
            print(f"Projections Done {perf_counter() - start}")
            return hashes
        else:
            return np.random.randint(INT64_MAX - 1,
                                     dtype=np.int64,
                                     size=(len(vectors),
                                           self.hash_len))

    def log_nn(self, vectors, key, n, start):
        top_n_lsh = lsh_nearest_neighbors(self.hashes, key, n=1000)
        top_n_nn = vectors.nearest_neighbors(key)
        recall = len(set(keys(top_n_nn)) & set(keys(top_n_lsh))) / n
        print(f"RECALL@{n} - {recall}")
        print(f"  PERF   - {perf_counter() - start}")
        print(f"   LSH - {top_n_lsh}")
        print(f"    GT - {top_n_nn}")

    def train(self,
              vectors,
              rounds,
              train_keys=[0],
              log_every=10,
              n=10):

        start = perf_counter()
        self.hashes = self._init_hashes(vectors)
        rounds_took = 0
        completes = {key: False for key in train_keys}
        vectors = ExactVectors(vectors, n=n, train_keys=train_keys)
        self.log_nn(vectors, train_keys[0], n, start)
        for i in range(rounds):

            key = train_keys[i % len(train_keys)]
            if np.array(list(completes.values())).all():
                break

            if i % log_every == 0:
                print(f"{i} - {key}")

            key_dotted = vectors.dot_prod_top(key)
            lsh_floor = vectors.lsh_floor(key)
            self.hashes, complete = train_one(self.hashes,
                                              key_dotted,
                                              key,
                                              lsh_floor=lsh_floor,
                                              learn_rate=0.1)

            if i % log_every == 0:
                self.log_nn(vectors, key, n, start)
            completes[key] = complete

            rounds_took = i
        print("FINAL")
        recalls = []
        for key in train_keys:
            top_n_lsh = lsh_nearest_neighbors(self.hashes, key, n=n)
            top_n_nn = vectors.nearest_neighbors(key)
            recall = len(set(keys(top_n_nn)) & set(keys(top_n_lsh))) / n
            print(f"RECALL@{n} - {recall}")
            recalls.append(recall)
            exact = [(idx, (score + 1) / 2) for idx, score in top_n_nn]
            print(f"LSH {lsh_nearest_neighbors(self.hashes, key, n=10)}")
            print(f" GT {exact}")
        print(f"  PERF   - {perf_counter() - start}")
        return self.hashes, recalls, rounds_took + 1

    def query(self, key: int, n=10):
        return lsh_nearest_neighbors(self.hashes, key, n=n)
