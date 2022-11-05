from sim_field import SimField
from model import Model, CacheModel
import random
from time import perf_counter
import numpy as np
import pytest
from vector_cache import VectorCache


def test_corpus_quantized_matches_non_quantized():
    model = Model('all-mpnet-base-v2')
    corpus_quantized = SimField(model, quantize=True, cached=False)
    corpus_float = SimField(model, quantize=False, cached=False)

    corpus_quantized.index(passages={('doc1', 1): 'Mary had a little lamb.',
                                     ('doc1', 2): 'Tom owns a cat.',
                                     ('doc1', 3): 'Wow I love bananas!'})

    corpus_float.index(passages={('doc1', 1): 'Mary had a little lamb.',
                                 ('doc1', 2): 'Tom owns a cat.',
                                 ('doc1', 3): 'Wow I love bananas!'})

    top_n_q = corpus_quantized.search('What does Mary have?').reset_index()
    top_n_f = corpus_float.search('What does Mary have?').reset_index()

    assert top_n_q['doc_id'].to_list() == top_n_f['doc_id'].to_list()
    assert top_n_q['passage_id'].to_list() == top_n_f['passage_id'].to_list()


def test_corpus_update():
    model = Model('all-mpnet-base-v2')
    corpus = SimField(model, cached=False)

    corpus.index(passages={('doc1', 1): 'Mary had a little lamb.',
                           ('doc1', 2): 'Tom owns a cat.',
                           ('doc1', 3): 'Wow I love bananas!'})

    corpus.index(passages={('doc1', 1): 'Mary had a little ham.'})

    top_n = corpus.search('What does Mary have?')
    assert len(top_n) == 3


def test_corpus_upsert():
    model = Model('all-mpnet-base-v2')
    corpus = SimField(model, cached=False)

    corpus.index(passages={('doc1', 1): 'Mary had a little lamb.',
                           ('doc1', 2): 'Tom owns a cat.',
                           ('doc1', 3): 'Wow I love bananas!'})

    corpus.index(passages={('doc1', 1): 'Mary had a little ham.',
                           ('doc1', 4): 'And I love apples'})

    top_n = corpus.search('What does Mary have?')
    assert len(top_n) == 4


class BatchCheckingModel:

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, text):
        assert not isinstance(text, str)
        return np.random.random((len(text), 768))


def test_corpus_encodes_as_batch():
    model = BatchCheckingModel('dummy')
    corpus = SimField(model, cached=False)
    corpus.index(passages={("doc1", 1): 'Mary had a little lamb.',
                           ("doc1", 2): 'Tom owns a cat.',
                           ("doc1", 3): 'Wow I love bananas!'})


class DummyModel:

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts):
        if isinstance(texts, str):
            return np.random.random(1, 768)
        else:
            return np.random.random((len(texts), 768))


def test_skipping_updates_faster_than_new_entries():
    model = Model('all-mpnet-base-v2')
    corpus = SimField(model, cached=False)

    start = perf_counter()
    for idx in range(0, 100):

        d = 1
        p = 1

        corpus.index(passages={(f"doc{d}", p): 'Mary had a little lamb.',
                               (f"doc{d}", p): 'Tom owns a cat.',
                               (f"doc{d}", p): 'Wow I love bananas!'},
                     skip_updates=True)

    skip_time = perf_counter() - start

    model = Model('all-mpnet-base-v2')
    corpus = SimField(model, cached=False)

    start = perf_counter()
    for idx in range(0, 100):

        d = random.randint(0, 10000)
        p = random.randint(0, 10000)

        corpus.index(passages={(f"doc{d}", p): 'Mary had a little lamb.',
                               (f"doc{d}", p): 'Tom owns a cat.',
                               (f"doc{d}", p): 'Wow I love bananas!'})

        corpus.index(passages={(f"doc{d}", p): 'Mary had a little ham.',
                               (f"doc{d}", p): 'And I love apples'})

    with_updates_time = perf_counter() - start
    assert with_updates_time > (8 * skip_time)


@pytest.mark.skip(reason="Used just for benchmarking")
def test_corpus_upsert_perf_dominated_by_encoding():
    model = DummyModel('dummy')
    corpus = SimField(model, cached=False)

    start = perf_counter()
    for idx in range(0, 100):

        d = random.randint(0, 10000)
        p = random.randint(0, 10000)

        corpus.index(passages={(f"doc{d}", p): 'Mary had a little lamb.',
                               (f"doc{d}", p): 'Tom owns a cat.',
                               (f"doc{d}", p): 'Wow I love bananas!'})

        corpus.index(passages={(f"doc{d}", p): 'Mary had a little ham.',
                               (f"doc{d}", p): 'And I love apples'})

    dummy_time = perf_counter() - start

    start = perf_counter()
    model = Model('all-mpnet-base-v2')
    corpus = SimField(model, cached=False)

    start = perf_counter()
    for idx in range(0, 100):

        d = random.randint(0, 10000)
        p = random.randint(0, 10000)

        corpus.index(passages={(f"doc{d}", p): 'Mary had a little lamb.',
                               (f"doc{d}", p): 'Tom owns a cat.',
                               (f"doc{d}", p): 'Wow I love bananas!'})

        corpus.index(passages={(f"doc{d}", p): 'Mary had a little ham.',
                               (f"doc{d}", p): 'And I love apples'})

    actual_encoder_time = perf_counter() - start

    assert actual_encoder_time > (8 * dummy_time)


def test_encode_in_loop_slower_than_encode_batch():

    sentences = ["bar baz foo %d" % random.randint(0, 10000)
                 for i in range(0, 100)]
    start = perf_counter()
    model = Model('all-mpnet-base-v2')
    model.encode(sentences)
    batch_time = perf_counter() - start

    start = perf_counter()
    for sentence in sentences:
        model.encode(sentence)
    iter_time = perf_counter() - start
    print(batch_time, iter_time)
    assert batch_time < 3 * iter_time


def test_corpus_ignore_updates_mode():
    model = Model('all-mpnet-base-v2')
    corpus = SimField(model, cached=False)

    corpus.index(passages={('doc1', 1): 'Mary had a little lamb.',
                           ('doc1', 2): 'Tom owns a cat.',
                           ('doc1', 3): 'Wow I love bananas!'})

    corpus.index(passages={('doc1', 1): 'Mary had a little ham.'},
                 skip_updates=True)

    top_n = corpus.search('What does Mary have?')
    assert len(top_n) == 3


class RedisMock:

    def __init__(self, cache={}):
        self.cache = cache

    def set(self, key, value):
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key)


@pytest.fixture
def empty_redis_mock():
    return RedisMock()


@pytest.fixture
def filled_redis_mock():
    return RedisMock(cache={"foo": np.array([1234, 5678])})


def test_cache_encoder_same_when_no_cache(empty_redis_mock):
    model = Model('all-mpnet-base-v2')
    cache_model = CacheModel(model, VectorCache(empty_redis_mock,
                                                dtype=np.float32))

    assert (model.encode("foo") == cache_model.encode("foo")).all()
    assert (model.encode(["foo"]) == cache_model.encode(["foo"])).all()
    sentences = ["The cat ate a berry.", "The berry was yummy.",
                 "It fell from a tree.", "The tree was decidous."]
    np.testing.assert_allclose(model.encode(sentences),
                               cache_model.encode(sentences),
                               rtol=0.01, atol=0.01)


def test_cache_encoder_uses_cache(filled_redis_mock):
    model = Model('all-mpnet-base-v2')
    cache_model = CacheModel(model, VectorCache(filled_redis_mock,
                                                dtype=np.int64,
                                                dims=2))
    assert (cache_model.encode("foo") == np.array([1234, 5678])).all()
    assert (cache_model.encode(["foo"]) == np.array([[1234, 5678]])).all()
