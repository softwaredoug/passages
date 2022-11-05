import numpy as np


class VectorCache:

    def __init__(self, r, dtype=np.float64):
        self.r = r
        self.dtype = dtype

    def set(self, key, arr):
        """Store given Numpy array 'a' in Redis under key 'n'"""
        if len(arr.shape) > 1:
            raise ValueError("Only supports vectors")

        if arr.dtype != self.dtype:
            msg = (f"Only type {self.dtype} supported" +
                   f"you passed arr of {arr.dtype}")
            raise ValueError(msg)
        encoded = arr.tobytes()

        # Store encoded data in Redis
        self.r.set(key, encoded)

    def get(self, key):
        """Retrieve Numpy array from Redis key 'n'"""
        encoded = self.r.get(key)
        a = np.frombuffer(encoded, dtype=self.dtype)
        return a
