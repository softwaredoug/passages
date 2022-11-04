from sentence_transformers import SentenceTransformer
from typing import Iterator, Union
import numpy as np


def cache_encode(passages, encoder, cache_encode):
    if isinstance(passages, str):
        passages = [passages]
    encoded = []
    for passage in passages:
        cached = cache_encode(passage)
        if cached is not None:
            encoded.append(cached)
        else:
            encoded.append(encoder(passage))
    return np.array(encoded)


class Model:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, passages: Union[str, Iterator[str]]):
        encoded = self.model.encode(passages)
        return encoded


class CacheModel:

    def __init__(self, model, read_from_cache):
        self.model = model
        self.cache_encode = read_from_cache

    def encode(self, passages: Union[str, Iterator[str]]):
        encoded = cache_encode(passages, self.model.encode, self.cache_encode)
        return encoded

    @property
    def model_name(self):
        return self.model.model_name
