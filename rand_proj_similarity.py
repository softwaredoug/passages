import random
import numpy as np
import numpy.typing as npt
from random_projection import projection_between, random_projection


def create_projections(vectors: npt.NDArray[np.float64],
                       num_projections: int,
                       between: bool = False) -> npt.NDArray[np.float64]:
    projections = []
    for idx in range(num_projections):
        projection_found = False
        while not projection_found:
            vect1_idx = random.randint(0, len(vectors)-1)
            vect2_idx = random.randint(0, len(vectors)-1)
            vect1 = vectors[vect1_idx]
            vect2 = vectors[vect2_idx]
            dim = random.randint(0, vect1.shape[0]-1)
            if np.sign(vect1[dim]) == np.sign(vect2[dim]):
                try:
                    if between:
                        proj = projection_between(vect1, vect2, dim)
                        # assert_projection_bisects(proj, vect1, vect2)
                    else:
                        proj = random_projection(vect1.shape)

                    projections.append(proj)
                    projection_found = True
                except ValueError:
                    projection_found = False
    return np.array(projections)


def set_bit(one_hash, idx):
    bit = idx % 64
    mask = np.int64(np.uint64(2 ** bit))
    one_hash[idx // 64] |= mask
    return one_hash


def clear_bit(one_hash, idx):
    bit = idx % 64
    mask = np.int64(np.uint64(2 ** bit))
    mask = ~mask
    one_hash[idx // 64] &= mask
    return one_hash


def train(hashes: npt.NDArray[np.int64],
          projections: npt.NDArray[np.float64],
          vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:

    for vect_idx, vect in enumerate(vectors):
        print(f"{vect_idx} - ", end="")
        for bit_idx, proj in enumerate(projections):
            dot = np.dot(vect, proj)
            if dot >= 0:
                print("+", end="")
                hashes[vect_idx] = set_bit(hashes[vect_idx], bit_idx)
            else:
                print("-", end="")
                hashes[vect_idx] = clear_bit(hashes[vect_idx], bit_idx)
        print()
    return hashes
