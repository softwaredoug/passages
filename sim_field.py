from typing import Optional, Mapping, Tuple, Union
from threading import Lock
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from similarity import similarity, quantize
from model import Model, CacheModel
from vector_cache import VectorCache

Path(".cache").mkdir(parents=True, exist_ok=True)


id_type = Tuple[str, int]


def _empty_passages() -> pd.DataFrame:
    passages = pd.DataFrame(columns=['doc_id', 'passage_id',
                                     'passage'])
    passages = passages.set_index(['doc_id', 'passage_id'])
    return passages


def _passages_from_dict(passages: Mapping[id_type, str]) -> pd.DataFrame:
    new_passages = pd.DataFrame(passages.items(),
                                columns=['id', 'passage'])
    new_passages[['doc_id', 'passage_id']] =\
        pd.DataFrame(new_passages['id'].tolist(),
                     columns=['doc_id', 'passage_id'])
    new_passages = new_passages[['doc_id', 'passage_id', 'passage']]
    new_passages = new_passages.set_index(['doc_id', 'passage_id'])
    return new_passages


class SimField:
    """A field corresponding to vector data for passages, alongside metadata"""

    def __init__(self, model: Union[Model, CacheModel],
                 field_name: Optional[str] = None,
                 quantize=True, cached=True, r=None):

        self.model = model

        if cached and r is not None:
            vector_cache = VectorCache(r, dtype=np.float32, dims=768)
            self.model = CacheModel(model, vector_cache)
        self.hits = 0
        self.misses = 0
        self.quantize = quantize
        self.last_cached = False
        if field_name is None:
            field_name = self.model.model_name + "_sim_field"
        self.field_name = field_name
        self.passages_lock = Lock()

        try:
            if cached:
                self.passages = pd.read_pickle(f".cache/{self.field_name}.pkl")
            else:
                self.passages = _empty_passages()
        except IOError:
            self.passages = _empty_passages()

    def _encode_passages(self, passages: pd.DataFrame) -> pd.DataFrame:
        encoded = self._quantized_encoder(passages['passage'])
        passages['passage'] = encoded.tolist()
        passages['passage'] = passages['passage'].apply(self._as_uint8)
        return passages

    def insert(self, passages: Mapping[id_type, str]):
        """Insert new passages, ignore any that overlap with existing."""
        new_passages = _passages_from_dict(passages)
        update_idxs = (
            new_passages.index.intersection(self.passages.index)
        )
        # All updates, we only insert, so ignore...
        if len(update_idxs) == len(new_passages):
            return

        inserts = new_passages.loc[
            new_passages.index.difference(update_idxs)
        ]
        inserts = self._encode_passages(inserts)
        self.passages_lock.acquire()
        self.passages = pd.concat([self.passages, inserts])
        self.passages_lock.release()

    def upsert(self, passages: Mapping[id_type, str], skip_updates=False):
        """Overwrite existing passages and insert new ones."""
        new_passages = _passages_from_dict(passages)

        update_idxs = (
            new_passages.index.intersection(self.passages.index)
        )

        new_passages = self._encode_passages(new_passages)
        inserts = new_passages.loc[
            new_passages.index.difference(update_idxs)
        ]

        self.passages_lock.acquire()
        self.passages.loc[update_idxs,
                          'passage'] = new_passages.loc[update_idxs]
        self.passages = pd.concat([self.passages, inserts])
        self.passages_lock.release()

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
        return similarity(query, self._quantized_encoder,
                          self.passages, 'passage')

    def _quantized_encoder(self, text):
        if self.quantize:
            return quantize(self.model.encode(text))
        else:
            return self.model.encode(text)

    def _as_uint8(self, int_list):
        return np.array(int_list, dtype=np.uint8)
