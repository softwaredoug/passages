import pickle
import redis
import numpy as np
from time import perf_counter
from vector_cache import VectorCache


def load_encode_cache():
    try:
        print("Loading cache...")
        start = perf_counter()
        with open('.cache/all-mpnet-base-v2.pkl', 'rb') as f:
            cache = pickle.load(f)
        print(f"Done - {perf_counter() - start}!")
    except IOError:
        print("No cache available")
        cache = {}
    return cache


def save_to_redis(vector_cache, cache):
    idx = 0
    print(f"Saving {len(cache)}")
    for passage, vector in cache.items():
        vector_cache.set(passage, np.array(vector, dtype=np.float32))
        if (idx % 10000 == 0):
            print(f"Saved {idx}")
        idx += 1


if __name__ == "__main__":
    r = redis.Redis(host='localhost', port=6379)
    cache = load_encode_cache()
    save_to_redis(VectorCache(r, dtype=np.float32), cache)
