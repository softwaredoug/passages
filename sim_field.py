from typing import Optional, Mapping, Tuple, Union
from similarity import similarity, quantize
from model import Model, CacheModel
from threading import Lock
import pandas as pd
import numpy as np
import pickle
from time import perf_counter
from pathlib import Path
Path(".cache").mkdir(parents=True, exist_ok=True)


id_type = Tuple[str, int]


def upsert(original, upserts):
    merged = original.merge(upserts,
                            left_index=True,
                            right_index=True,
                            how='outer',
                            indicator=True)
    merged = merged.rename(columns={'passage_x': 'passage'})
    # Drop 'both'
    # Keep passage_y for right only and both,
    # Keep passage_x for left only ,
    upserted = ((merged['_merge'] == 'right_only') |
                (merged['_merge'] == 'both'))
    merged.loc[upserted, 'passage'] = merged[upserted]['passage_y']
    merged = merged[['passage']]
    return merged


def remove_also_in(new, other):
    new = new.merge(other,
                    left_index=True,
                    right_index=True,
                    how='outer',
                    indicator=True)
    new = new[new['_merge'] == 'left_only']
    new = new.rename(columns={'passage_x': 'passage'})[['passage']]
    return new


def new_passages():
    passages = pd.DataFrame(columns=['doc_id', 'passage_id',
                                     'passage'])
    passages = passages.set_index(['doc_id', 'passage_id'])
    return passages


try:
    import redis
    import json
    r = redis.Redis(host='localhost', port=6379)
except ImportError:
    r = None

def get_from_redis(passage: str):
    vector = r.get(passage)
    if vector is not None:
        return np.array(json.loads(r.get(passage)))
    else:
        return None


class SimField:
    """A field corresponding to vector data for passages, alongside metadata"""

    def __init__(self, model: Union[Model, CacheModel],
                 field_name: Optional[str] = None,
                 quantize=True, cached=True):

        self.model = model

        if r is not None:
            self.model = CacheModel(model, get_from_redis)
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
                self.passages = new_passages()
        except IOError:
            self.passages = new_passages()

    def index(self, passages: Mapping[id_type, str], skip_updates=False):
        new_passages = pd.DataFrame(passages.items(),
                                    columns=['id', 'passage'])
        new_passages[['doc_id', 'passage_id']] =\
            pd.DataFrame(new_passages['id'].tolist(),
                         columns=['doc_id', 'passage_id'])
        new_passages = new_passages[['doc_id', 'passage_id', 'passage']]
        new_passages = new_passages.set_index(['doc_id', 'passage_id'])

        if skip_updates:
            new_passages = remove_also_in(new_passages, self.passages)

        encoded = self._quantized_encoder(new_passages['passage'])
        new_passages['passage'] = encoded.tolist()
        new_passages['passage'] = new_passages['passage'].apply(self._as_uint8)

        self.passages_lock.acquire()
        self.passages = upsert(self.passages, new_passages)
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
