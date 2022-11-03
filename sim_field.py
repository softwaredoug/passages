from typing import Optional, Mapping, Tuple
from similarity import similarity, quantize
from model import Model
from threading import Lock
import pandas as pd
import pickle
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


class SimField:
    """A field corresponding to vector data for passages, alongside metadata"""

    def __init__(self, model: Model,
                 sim_field_name: Optional[str] = None,
                 quantize=True):

        self.model = model
        self.hits = 0
        self.misses = 0
        self.quantize = quantize
        self.last_cached = False
        if sim_field_name is None:
            sim_field_name = self.model.model_name + "_sim_field"
        self.sim_field_name = sim_field_name
        self.sim_field_lock = Lock()

        try:
            self.sim_field = pd.read_pickle(f".cache/{self.sim_field_name}.pkl")
        except FileNotFoundError:
            self.sim_field = pd.DataFrame(columns=['doc_id', 'passage_id',
                                                   'passage'])
            self.sim_field = self.sim_field.set_index(['doc_id', 'passage_id'])
        except EOFError:
            self.sim_field = pd.DataFrame(columns=['doc_id', 'passage_id',
                                                   'passage'])
            self.sim_field = self.sim_field.set_index(['doc_id', 'passage_id'])

    def index(self, passages: Mapping[id_type, str], skip_updates=False):
        as_sim_field = pd.DataFrame(passages.items(),
                                    columns=['id', 'passage'])
        as_sim_field[['doc_id', 'passage_id']] =\
            pd.DataFrame(as_sim_field['id'].tolist(),
                         columns=['doc_id', 'passage_id'])
        as_sim_field = as_sim_field[['doc_id', 'passage_id', 'passage']]
        as_sim_field = as_sim_field.set_index(['doc_id', 'passage_id'])

        if skip_updates:
            as_sim_field = remove_also_in(as_sim_field, self.sim_field)

        as_sim_field['passage'] =\
            as_sim_field['passage'].apply(self._quantized_encoder)

        self.sim_field_lock.acquire()
        self.sim_field = upsert(self.sim_field, as_sim_field)
        self.sim_field_lock.release()

    def stats(self) -> dict:
        return {
            "cache_size": len(self.sim_field),
            "is_cached": self.last_cached,
            "hits": self.hits,
            "misses": self.misses
        }

    def persist(self):
        self.sim_field_lock.acquire()
        cache = self.sim_field.copy()
        self.sim_field_lock.release()
        with open(f".cache/{self.sim_field_name}.pkl", "wb") as f:
            pickle.dump(cache, f)

    def search(self, query: str) -> pd.DataFrame:
        return similarity(query, self._quantized_encoder,
                          self.sim_field, 'passage')

    def _quantized_encoder(self, text):
        if self.quantize:
            return quantize(self.model.encode(text))
        else:
            return self.model.encode(text)
