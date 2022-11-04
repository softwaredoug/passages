import redis
import json
import pickle
from time import perf_counter


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


def save_to_redis(r, cache):
    idx = 0
    print(f"Saving {len(cache)}")
    for passage, vector in cache.items():
        r.set(passage, json.dumps(vector))
        if (idx % 10000 == 0):
            print(f"Saved {idx}")
        idx += 1


if __name__ == "__main__":
    r = redis.Redis(host='localhost', port=6379)
    cache = load_encode_cache()
    save_to_redis(r, cache)
