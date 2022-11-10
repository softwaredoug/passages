from typing import Optional, Mapping, Tuple, Union, List
from threading import Lock
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from time import perf_counter
from similarity import similarity
from model import Model, CacheModel
from vector_cache import VectorCache

Path(".cache").mkdir(parents=True, exist_ok=True)


id_type = Tuple[str, int]


def _empty_passages(dims) -> pd.DataFrame:
    columns: List[Union[str, int]] = ['doc_id', 'passage_id']
    columns.extend(list(range(dims)))
    passages = pd.DataFrame(columns=columns)
    passages = passages.set_index(['doc_id', 'passage_id'])
    return passages


def _passages_from_dict(passages: Mapping[id_type, str], dims) -> pd.DataFrame:
    new_passages = pd.DataFrame(passages.items(),
                                columns=['id', 'passage'])
    new_passages[['doc_id', 'passage_id']] =\
        pd.DataFrame(new_passages['id'].tolist(),
                     columns=['doc_id', 'passage_id'])
    new_passages = new_passages[['doc_id', 'passage_id', 'passage']]
    new_passages = new_passages.set_index(['doc_id', 'passage_id'])
    return new_passages


class SimField:
    """A field corresponding to vector data for passages, alongside metadata


    Internal Details
    ----------------

    > Storage at rest
    At rest we want to store a dataframe, where columns are just 0...N of the
    vector describing each row, with each row indexed by (docid, passage id).
    This makes it easy to search without performing copies (just a dot product)

    > Indexing a dataframe performantly
    While updating its faster to update at the object level (assigning objects
    like lists). Only at the end exploding that list into a more matrix

    """

    def __init__(self, model: Union[Model, CacheModel],
                 field_name: Optional[str] = None,
                 dims=768, cached=True, r=None):

        self.model = model

        if cached and r is not None:
            vector_cache = VectorCache(r, dtype=np.float32, dims=768)
            self.model = CacheModel(model, vector_cache)
        self.hits = 0
        self.misses = 0
        self.dims = dims
        self._column_index = slice(0, self.dims-1)
        self.last_cached = False
        if field_name is None:
            field_name = self.model.model_name + "_sim_field"
        self.field_name = field_name
        self.passages_lock = Lock()

        try:
            if cached:
                self.passages = pd.read_pickle(f".cache/{self.field_name}.pkl")
                print(f"Loaded {len(self.passages)} searchable passages")
            else:
                self.passages = _empty_passages(self.dims)
        except IOError:
            self.passages = _empty_passages(self.dims)

    def _encode_passages(self,
                         passages: pd.DataFrame,
                         recreate=True) -> pd.DataFrame:
        encoded = self._quantized_encoder_idx(passages['passage'])
        passages['passage'] = encoded.tolist()
        return passages

    def _explode_passages(self):
        set_passage_col = ~self.passages['passage'].isna()
        to_explode = self.passages.loc[set_passage_col]
        # The actual exploding is the bottleneck
        # for indexing (aside from encoding).
        exploded = to_explode['passage'].apply(pd.Series)
        self.passages = self.passages.drop(columns='passage')
        self.passages.loc[set_passage_col] = exploded

    def _explode_new_df(self):
        set_passage_col = ~self.passages['passage'].isna()
        to_explode = self.passages.loc[set_passage_col]
        exploded = pd.DataFrame(to_explode['passage'].tolist(),
                                index=to_explode.index)
        self.passages = self.passages.drop(columns='passage')
        self.passages.loc[set_passage_col] = exploded

    def insert(self, passages: Mapping[id_type, str]):
        """Insert new passages, ignore any that overlap with existing."""
        start = perf_counter()
        new_passages = _passages_from_dict(passages, self.dims)
        update_idxs = (
            new_passages.index.intersection(self.passages.index)
        )
        # All updates, we only insert, so ignore...
        if len(update_idxs) == len(new_passages):
            print(f"Ins: {perf_counter() - start}")
            return

        inserts = new_passages.loc[
            new_passages.index.difference(update_idxs)
        ]
        inserts = self._encode_passages(inserts)
        print(f"Enc: {perf_counter() - start}")
        self.passages_lock.acquire()
        self.passages = pd.concat([self.passages, inserts])
        self._explode_passages()
        self.passages_lock.release()
        assert len(self.passages.columns) == self.dims
        print(f"Ins: {perf_counter() - start}")

    def upsert(self, passages: Mapping[id_type, str], skip_updates=False):
        """Overwrite existing passages and insert new ones."""
        start = perf_counter()
        new_passages = _passages_from_dict(passages, self.dims)

        update_idxs = (
            new_passages.index.intersection(self.passages.index)
        )

        new_passages = self._encode_passages(new_passages)
        print(f"Enc: {perf_counter() - start}")
        inserts = new_passages.loc[
            new_passages.index.difference(update_idxs)
        ]

        self.passages_lock.acquire()
        self.passages.loc[update_idxs,
                          'passage'] = new_passages.loc[update_idxs]
        self.passages = pd.concat([self.passages, inserts])
        self._explode_passages()
        self.passages_lock.release()
        assert len(self.passages.columns) == self.dims
        print(f"Ups: {perf_counter() - start}")

    def stats(self) -> dict:
        return {
            "cache_size": len(self.passages),
            "is_cached": self.last_cached,
            "hits": self.hits,
            "misses": self.misses
        }

    def persist(self):
        self.passages_lock.acquire()
        cache = self.passages.copy()
        # if r:
        #    r.save()
        self.passages_lock.release()
        with open(f".cache/{self.field_name}.pkl", "wb") as f:
            pickle.dump(cache, f)

    def search(self, query: str) -> pd.DataFrame:
        start = perf_counter()
        top_n = similarity(query, self._quantized_encoder_query,
                           self.passages)
        print(f"Similarity - {perf_counter() - start}")
        return top_n

    def _quantized_encoder_idx(self, text):
        return self.model.encode(text)

    def _quantized_encoder_query(self, text):
        return self.model.encode(text)
