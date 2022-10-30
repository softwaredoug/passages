from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from time import perf_counter
import pickle
from copy import copy
from threading import Lock



from pathlib import Path
Path(".cache").mkdir(parents=True, exist_ok=True)


app = Flask(__name__)




class Model:

    def __init__(self, model_name):
        self.model_name = model_name

        try:
            with open(f".cache/{self.model_name}.pkl", "rb") as f:
                self.cache = pickle.load(f)
        except FileNotFoundError:
            self.cache = {}
        except EOFError:
            self.cache = {}

        self.model = SentenceTransformer(model_name)
        self.hits = 0
        self.misses = 0
        self.last_cached = False
        self.cache_lock = Lock()

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
                self.cache_lock.acquire()
                self.cache[passage] = encoded
                self.cache_lock.release()
        return encoded

    def stats(self):
        return {
            "cache_size": len(self.cache),
            "is_cached": self.last_cached,
            "hits": self.hits,
            "misses": self.misses
        }

    def persist(self):

        self.cache_lock.acquire()
        cache = copy(self.cache)
        self.cache_lock.release()

        with open(f".cache/{self.model_name}.pkl", "wb") as f:
            pickle.dump(cache, f)


models = {'all-mpnet-base-v2': Model('all-mpnet-base-v2')}


def get_stats():
    stats = {}
    for model_name, model in models.items():
        stats[model_name] = model.stats()
    return stats


@app.route("/persist")
def persist():
    stats = {}
    for _, model in models.items():
        model.persist()
    resp = {}
    resp['stats'] = get_stats()
    return jsonify(resp)


@app.route("/stats")
def stats():
    resp = {}
    resp['stats'] = get_stats()
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

    return jsonify(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
