from sim_field import SimField
from model import Model
import random
from time import perf_counter
import numpy as np


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


class DummyModel:

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, text):
        return np.random.random(768)


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
