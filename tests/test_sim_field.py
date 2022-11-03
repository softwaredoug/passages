from sim_field import SimField
from model import Model


def test_corpus_quantized_matches_non_quantized():
    model = Model('all-mpnet-base-v2')
    corpus_quantized = SimField(model, quantize=True)
    corpus_float = SimField(model, quantize=False)

    corpus_quantized.index(passages={('doc1', 1): 'Mary had a little lamb.',
                                     ('doc1', 2): 'Tom owns a cat.',
                                     ('doc1', 3): 'Wow I love bananas!'})

    corpus_float.index(passages={('doc1', 1): 'Mary had a little lamb.',
                                 ('doc1', 2): 'Tom owns a cat.',
                                 ('doc1', 3): 'Wow I love bananas!'})

    top_n_q = corpus_quantized.search('What does Mary have?')
    top_n_f = corpus_float.search('What does Mary have?')

    assert top_n_q['doc_id'].to_list() == top_n_f['doc_id'].to_list()
    assert top_n_q['passage_id'].to_list() == top_n_f['passage_id'].to_list()


def test_corpus_update():
    model = Model('all-mpnet-base-v2')
    corpus = SimField(model)

    corpus.index(passages={('doc1', 1): 'Mary had a little lamb.',
                           ('doc1', 2): 'Tom owns a cat.',
                           ('doc1', 3): 'Wow I love bananas!'})

    corpus.index(passages={('doc1', 1): 'Mary had a little ham.'})

    top_n = corpus.search('What does Mary have?')
    assert len(top_n) == 3


def test_corpus_upsert():
    model = Model('all-mpnet-base-v2')
    corpus = SimField(model)

    corpus.index(passages={('doc1', 1): 'Mary had a little lamb.',
                           ('doc1', 2): 'Tom owns a cat.',
                           ('doc1', 3): 'Wow I love bananas!'})

    corpus.index(passages={('doc1', 1): 'Mary had a little ham.',
                           ('doc1', 4): 'And I love apples'})

    top_n = corpus.search('What does Mary have?')
    assert len(top_n) == 4


def test_corpus_ignore_updates_mode():
    model = Model('all-mpnet-base-v2')
    corpus = SimField(model)

    corpus.index(passages={('doc1', 1): 'Mary had a little lamb.',
                           ('doc1', 2): 'Tom owns a cat.',
                           ('doc1', 3): 'Wow I love bananas!'})

    corpus.index(passages={('doc1', 1): 'Mary had a little ham.'},
                 skip_updates=True)

    top_n = corpus.search('What does Mary have?')
    assert len(top_n) == 3
