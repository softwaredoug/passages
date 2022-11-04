from sentence_transformers import SentenceTransformer
from typing import Iterator, Union
import numpy as np


def cache_encode(passages, encoder, cache_get, cache_set):
    single_vect = False
    if isinstance(passages, str):
        single_vect = True
        passages = [passages]
    encoded = []
    for passage in passages:
        cached = cache_get(passage)
        if cached is not None:
            encoded.append(cached)
        else:
            encoded = encoder(passage)
            cache_set(passage, encoded)
    arr = np.array(encoded)
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

    def __init__(self, model, cache_get, cache_set):
        self.model = model
        self.cache_get = cache_get
        self.cache_set = cache_set

    def encode(self, passages: Union[str, Iterator[str]]):
        encoded = cache_encode(passages, self.model.encode,
                               self.cache_get, self.cache_set)
        return encoded

    @property
    def model_name(self):
        return self.model.model_name
