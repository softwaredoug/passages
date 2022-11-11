from typing import Optional, Mapping, Tuple
from threading import Lock
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from time import perf_counter
from similarity import exact_nearest_neighbors

Path(".cache").mkdir(parents=True, exist_ok=True)


id_type = Tuple[str, int]


class ArrayIndex:

    def __init__(self):
        self.index = {}
        self.reverse_index = {}

    def get_key(self, item):
        try:
            return self.index[item]
        except KeyError:
            self.index[item] = len(self.index)
            self.reverse_index[self.index[item]] = item
            return self.index[item]

    def index_for(self, idx):
        return self.reverse_index[idx]

    def __len__(self):
        return len(self.index)


def _passages_from_dict(passages: Mapping[id_type, str], dims) -> pd.DataFrame:
    new_passages = pd.DataFrame(passages.items(),
                                columns=['id', 'passage'])
    # new_passages[['doc_id', 'passage_id']] =\
    #    pd.DataFrame(new_passages['id'].tolist(),
    #                 columns=['doc_id', 'passage_id'])
    new_passages = new_passages[['id', 'passage']]
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

    def __init__(self, model,
                 field_name: Optional[str] = None,
                 dims=768, cached=True):

        self.model = model

        # if cached and r is not None:
        #    vector_cache = VectorCache(r, dtype=np.float32, dims=768)
        #    self.model = CacheModel(model, vector_cache)
        self.hits = 0
        self.misses = 0
        self.dims = dims
        # self._column_index = slice(0, self.dims-1)
        self.index = ArrayIndex()
        self.last_cached = False
        if field_name is None:
            field_name = self.model.model_name + "_sim_field"
        self.field_name = field_name
        self.passages_lock = Lock()
        self.passages = None

        try:
            if cached:
                self.passages = pd.read_pickle(f".cache/{self.field_name}.pkl")
                print(f"Loaded {len(self.passages)} searchable passages")
        except IOError:
            pass

    def _encode_passages(self,
                         passages: pd.DataFrame,
                         recreate=True) -> pd.DataFrame:
        encoded = self._quantized_encoder_idx(passages['passage'])
        return encoded

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
        self.upsert(passages, skip_updates=True)

    def upsert(self, passages: Mapping[id_type, str], skip_updates=False):
        """Overwrite existing passages and insert new ones."""
        start = perf_counter()
        orig_index_size = len(self.index)
        new_passages = _passages_from_dict(passages, self.dims)

        # FIX RACE ON INDEX KEYS
        new_passages['key'] = new_passages['id'].apply(self.index.get_key)
        if not skip_updates:
            new_passages['passage'] \
                = new_passages['passage'].apply(self._quantized_encoder_idx)

        updates = new_passages.loc[
            new_passages['key'] < orig_index_size, :
        ].reset_index(drop=True)
        inserts = new_passages.loc[
            new_passages['key'] >= orig_index_size, :
        ].reset_index(drop=True)

        if len(inserts) == 0 and skip_updates:
            return
        elif skip_updates:
            inserts['passage'] \
                = inserts['passage'].apply(self._quantized_encoder_idx)

        self.passages_lock.acquire()
        if self.passages is None:
            self.passages = np.array(inserts['passage'].tolist())
        elif len(inserts) > 0:
            # Assumes inserts will be placed in the right index
            insert_arrs = np.array(inserts['passage'].tolist())
            self.passages = np.vstack([self.passages,
                                       insert_arrs])

        if len(updates) > 0 and not skip_updates:
            updated_arrs = np.array(updates['passage'].tolist())
            self.passages[updates['key']] = updated_arrs
        self.passages_lock.release()
        print(f"UPS - {perf_counter() - start}")
        return

    def stats(self) -> dict:
        cache_size = 0
        if self.passages is not None:
            cache_size = len(self.passages)
        return {
            "cache_size": cache_size,
            "is_cached": self.last_cached,
            "hits": self.hits,
            "misses": self.misses
        }

    def is_empty(self):
        return self.passages.shape == (0,)

    def persist(self):
        self.passages_lock.acquire()
        cache = self.passages.copy()
        # if r:
        #    r.save()
        self.passages_lock.release()
        with open(f".cache/{self.field_name}.pkl", "wb") as f:
            pickle.dump(cache, f)

    def search(self, query: str) -> pd.DataFrame:
        if self.passages is None:
            return pd.DataFrame()
        start = perf_counter()
        top_n, scores = \
            exact_nearest_neighbors(self._quantized_encoder_query(query),
                                    self.passages)
        results = pd.DataFrame()
        results['key'] = top_n
        results['score'] = scores
        results['id'] = results['key'].apply(self.index.index_for)
        print(f"SCH - {perf_counter() - start}")
        return results

    def _quantized_encoder_idx(self, text):
        return self.model.encode(text)

    def _quantized_encoder_query(self, text):
        return self.model.encode(text)
