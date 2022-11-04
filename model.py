from sentence_transformers import SentenceTransformer
from typing import Iterator, Union
import numpy as np
import pickle
from time import perf_counter

try:
    print("Loading cache...")
    start = perf_counter()
    with open('.cache/all-mpnet-base-v2.pkl', 'rb') as f:
        cache = pickle.load(f)
    print(f"Done - {perf_counter() - start}!")
except IOError:
    print("No cache available")
    cache = {}


def cache_encode(passages, encoder):
    if isinstance(passages, str):
        passages = [passages]
    encoded = []
    for passage in passages:
        try:
            encoded.append(cache[passage])
        except KeyError:
            encoded.append(encoder(passage))
    return np.array(encoded)


class Model:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, passages: Union[str, Iterator[str]]):
        # encoded = cache_encode(passages, self.model.encode)
        encoded = self.model.encode(passages)
        return encoded
