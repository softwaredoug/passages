from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from time import perf_counter


app = Flask(__name__)

models = {}
cache = {}


class Model:

    def __init__(self, model_name):
        self.cache = {}
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, passage):
        try:
            encoded = self.cache[passage]
        except KeyError:
            encoded = self.model.encode(passage).tolist()
            if len(passage) < 100:
                self.cache[passage] = encoded
        return encoded


@app.route("/<model_name>")
def get_passage(model_name):
    start = perf_counter()
    args = request.args
    if model_name not in models:
        models[model_name] = Model(model_name)
    model = models[model_name]
    passage = args.get('q')
    encoded = model.encode(passage)
    resp = {
        "q": passage,
        "model": model_name,
        "encoded": encoded,
        "encode_time": perf_counter() - start
    }
    return jsonify(resp)
