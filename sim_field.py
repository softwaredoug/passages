from typing import Optional, Mapping, Tuple
from threading import Lock
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from time import perf_counter
from copy import deepcopy
from similarity import exact_nearest_neighbors, keys, scores

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

    def copy(self):
        copy = ArrayIndex()
        copy.index = deepcopy(self.index)
        copy.reverse_index = deepcopy(self.reverse_index)
        return copy

    def keep_only(self, num_to_keep):
        copy = ArrayIndex()
        # This should be a dict expression
        for idx, val in self.index.items():
            if val < num_to_keep:
                copy.index[idx] = val
                copy.reverse_index[val] = idx
        assert len(copy) == num_to_keep
        return copy

    def __len__(self):
        return len(self.index)


def _passages_from_dict(passages: Mapping[id_type, str], dims) -> pd.DataFrame:
    new_passages = pd.DataFrame(passages.items(),
                                columns=['id', 'passage'])
    new_passages = new_passages[['id', 'passage']]
    return new_passages


class SimField:

    def __init__(self, model,
                 field_name: Optional[str] = None,
                 dims=768, cached=True):

        self.model = model

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
                self.passages = np.load(self.corpus_path)
                with open(self.index_path, 'rb') as f:
                    self.index = pickle.load(f)
                if len(self.passages) > len(self.index):
                    raise IOError("Invalid local cache files, skipping")
                else:
                    self.index = self.index.keep_only(len(self.passages))
                assert len(self.index) == len(self.passages)
                assert self.passages.dtype == np.half
                print(f"Loaded {len(self.passages)} searchable passages"
                      f" ({self.passages.nbytes} bytes)")
        except IOError as e:
            warnings.warn(f"Handling {e} - resetting corpus")
            pass

    def _encode_passages(self,
                         passages: pd.DataFrame) -> pd.DataFrame:
        passages['passage'] \
            = self._quantized_encoder_idx(passages['passage']).tolist()

        def half_flt(lst):
            return np.array(lst, dtype=np.half)

        passages['passage'] = passages['passage'].apply(half_flt)
        return passages

    def insert(self, passages: Mapping[id_type, str]):
        """Insert new passages, ignore any that overlap with existing."""
        self.upsert(passages, skip_updates=True)

    def upsert(self, passages: Mapping[id_type, str], skip_updates=False):
        """Overwrite existing passages and insert new ones."""
        start = perf_counter()
        orig_index_size = len(self.index)
        new_passages = _passages_from_dict(passages, self.dims)

        self.passages_lock.acquire()
        new_passages['key'] = new_passages['id'].apply(self.index.get_key)
        self.passages_lock.release()

        if not skip_updates:
            new_passages = self._encode_passages(new_passages)

        updates = new_passages.loc[
            new_passages['key'] < orig_index_size, :
        ].reset_index(drop=True)
        inserts = new_passages.loc[
            new_passages['key'] >= orig_index_size, :
        ].reset_index(drop=True)

        if len(inserts) == 0 and skip_updates:
            return
        elif skip_updates:
            inserts = self._encode_passages(inserts)

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
        nbytes = 0
        if self.passages is not None:
            cache_size = len(self.passages)
            nbytes = self.passages.nbytes
        return {
            "num_passages": cache_size,
            "bytes": nbytes,
            "is_cached": self.last_cached,
            "hits": self.hits,
            "misses": self.misses
        }

    def persist(self):
        self.passages_lock.acquire()
        # NOT ATOMIC
        # Not particularly safe to reload if either write fails
        np.save(self.corpus_path, self.passages)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        self.passages_lock.release()

    def search(self, query: str) -> pd.DataFrame:
        if self.passages is None:
            return pd.DataFrame()
        start = perf_counter()
        top_n = \
            exact_nearest_neighbors(self._quantized_encoder_query(query),
                                    self.passages)
        results = pd.DataFrame()
        results['key'] = keys(top_n)
        results['score'] = scores(top_n)
        results['id'] = results['key'].apply(self.index.index_for)
        print(f"SCH - {perf_counter() - start}")
        return results

    def _quantized_encoder_idx(self, text):
        return self.model.encode(text)

    def _quantized_encoder_query(self, text):
        return self.model.encode(text)

    @property
    def corpus_path(self):
        return f".cache/passages_{self.field_name}.npy"

    @property
    def index_path(self):
        return f".cache/index_{self.field_name}.pkl"
