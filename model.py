from sentence_transformers import SentenceTransformer
from typing import Iterator, Union
from time import perf_counter
import numpy as np


# Possible improvements:
# If encoding of new passages can be done in batch, they will be
# about 2x faster. BUT we have to align the output array order
# to be same as input array order

def cache_encode(passages, encoder, cache_get, cache_set):
    start = perf_counter()
    single_vect = False
    if isinstance(passages, str):
        single_vect = True
        passages = [passages]
    encoded = []
    hits = 0
    misses = 0
    for passage in passages:
        cached = cache_get(passage)
        if cached is not None:
            encoded.append(cached)
            hits += 1
        else:
            enc_passage = encoder(passage)
            encoded.append(enc_passage)
            cache_set(passage, enc_passage)
            misses += 1
    arr = np.array(encoded)
    print(f"Encode done {hits}, {misses}, {perf_counter() - start}")
    if single_vect:
        return arr[0]
    else:
        return arr


class Model:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, passages: Union[str, Iterator[str]]):
        encoded = self.model.encode(passages)
        return encoded


class CacheModel:

    def __init__(self, model, vector_cache):
        self.model = model
        self.vector_cache = vector_cache

    def encode(self, passages: Union[str, Iterator[str]]):
        encoded = cache_encode(passages, self.model.encode,
                               self.vector_cache.get,
                               self.vector_cache.set)
        return encoded

    @property
    def model_name(self):
        return self.model.model_name


class EncodedModel:
    def __init__(self, model, dtype):
        self.model = model
        self.dtype = dtype

    def encode(self, passages: Union[str, Iterator[str]]):
        encoded = self.model.encode(passages).astype(self.dtype)
        return encoded

    @property
    def model_name(self):
        return self.model.model_name
