from vector_cache import VectorCache
import numpy as np
import pytest


class RedisMock:

    def __init__(self):
        self.cache = {}

    def set(self, key, value):
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key)


@pytest.fixture
def redis_mock():
    return RedisMock()


def test_set_get_float64_default(redis_mock):
    arr = np.array([1.0, 2.0])
    vector_cache = VectorCache(redis_mock)
    vector_cache.set("foo", arr)
    assert (vector_cache.get("foo") == arr).all()


def test_set_get_int_types(redis_mock):
    arr = np.array([1, 2, 3])
    vector_cache = VectorCache(redis_mock, dtype=np.int64)
    vector_cache.set("foo", arr)
    assert (vector_cache.get("foo") == arr).all()


def test_raises_on_wrong_type(redis_mock):
    arr = np.array([1.0, 2.0, 3.0])
    vector_cache = VectorCache(redis_mock, dtype=np.int64)
    with pytest.raises(ValueError):
        vector_cache.set("foo", arr)


def test_none_on_no_vector(redis_mock):
    vector_cache = VectorCache(redis_mock, dtype=np.int64)
    foo = vector_cache.get("foo")
    assert foo is None
