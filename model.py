from sentence_transformers import SentenceTransformer
from typing import Iterator, Union


class Model:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, passages: Union[str, Iterator[str]]):
        return self.model.encode(passages)
