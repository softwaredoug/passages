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
        self.hits = 0
        self.misses = 0
        self.last_cached = False

    def encode(self, passage):
        try:
            encoded = self.cache[passage]
            self.hits += 1
            self.last_cached = True
        except KeyError:
            encoded = self.model.encode(passage).tolist()
            self.last_cached = False
            self.misses += 1
            if len(passage) < 1000:
                self.cache[passage] = encoded
        return encoded

    def stats(self):
        return {
            "cache_size": len(self.cache),
            "is_cached": self.last_cached,
            "hits": self.hits,
            "misses": self.misses
        }



def get_stats():
    stats = {}
    for model_name, model in models.items():
        stats[model_name] = model.stats()
    return stats


@app.route("/stats")
def stats():
    resp = {}
    resp['stats'] = get_stats()
    if get_passage.num_requests % 100 == 0:
        print(resp['stats'])
    return jsonify(resp)


@app.route("/encode/<model_name>")
def get_passage(model_name):
    start = perf_counter()
    args = request.args
    if model_name not in models:
        models[model_name] = Model(model_name)
    model = models[model_name]
    passage = args.get('q')
    stats = args.get('stats')
    encoded = model.encode(passage)


    resp = {
        "q": passage,
        "model": model_name,
        "encoded": encoded,
        "encode_time": perf_counter() - start
    }

    if stats:
        resp['stats'] = get_stats()
    get_passage.num_requests += 1

    return jsonify(resp)


get_passage.num_requests = 0
