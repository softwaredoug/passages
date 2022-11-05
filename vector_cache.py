import numpy as np
from typing import Union, Optional


class VectorCache:

    def __init__(self, r, dtype=np.float64, dims=768):
        self.r = r
        self.dtype = dtype
        self.dims = dims

    def set(self, key: str, arr: np.ndarray):
        """Store given Numpy array 'a' in Redis under key 'n'"""
        if len(arr.shape) > 1:
            raise ValueError("Only supports vectors")

        if arr.shape[0] != self.dims:
            raise ValueError(f"Only supports {self.dims} dimensions")

        if arr.dtype != self.dtype:
            msg = (f"Only type {self.dtype} supported" +
                   f"you passed arr of {arr.dtype}")
            raise ValueError(msg)
        encoded = arr.tobytes()

        # Store encoded data in Redis
        self.r.set(key, encoded)

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve Numpy array from Redis key 'n'"""
        encoded = self.r.get(key)
        if encoded is None:
            return None
        a = np.frombuffer(encoded, dtype=self.dtype)
        if a.shape[0] == self.dims:
            return a
        return None


class NullVectorCache:

    def __init__(self):
        pass

    def set(self, key, arr):
        pass

    def get(self, key):
        return None


vector_cache: Union[VectorCache, NullVectorCache] = NullVectorCache()

try:
    import redis
    r = redis.Redis(host='localhost', port=6379)
    vector_cache = VectorCache(r)
except ImportError:
    pass
