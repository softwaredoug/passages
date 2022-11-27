import random
import numpy as np
from random_projection import projection_between


def create_projections(vectors, num_projections):
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
                    proj = projection_between(vect1, vect2, dim)
                    # assert_projection_bisects(proj, vect1, vect2)

                    projections.append(proj)
                    projection_found = True
                except ValueError:
                    projection_found = False
    return projections


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


def train(hashes, vectors):
    hash_len = hashes.shape[1]
    num_projections = hash_len * 64

    projections = create_projections(vectors, num_projections)

    for vect_idx, vect in enumerate(vectors):
        for bit_idx, proj in enumerate(projections):
            dot = np.dot(vect, proj)
            if dot >= 0:
                hashes[vect_idx] = set_bit(hashes[vect_idx], bit_idx)
            else:
                hashes[vect_idx] = clear_bit(hashes[vect_idx], bit_idx)
    return hashes
