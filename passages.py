from typing import Dict
from flask import Flask, request, jsonify
from sim_field import SimField
from model import Model, CacheModel, EncodedModel
from vector_cache import VectorCache
import json
import numpy as np
from redis import Redis


app = Flask(__name__)
fields: Dict[str, SimField] = {}


def get_stats():
    stats = {}
    for model_name, model in fields.items():
        stats[model_name] = model.stats()
    return stats


@app.route("/persist")
def persist():
    for _, field in fields.items():
        field.persist()
    resp = {}
    resp['stats'] = get_stats()
    return jsonify(resp)


@app.route("/stats")
def stats():
    resp = {}
    resp['stats'] = get_stats()
    return jsonify(resp)


# Available models
def load_fields():
    r = Redis('localhost', 6379)
    vector_cache = VectorCache(r, dtype=np.float32)
    for model_name in ['all-mpnet-base-v2']:
        field_name = model_name + "_field"
        model = Model(model_name)
        cached_model = CacheModel(model, vector_cache)
        encoded = EncodedModel(cached_model, dtype=np.half)
        fields[field_name] = SimField(encoded)


@app.route("/index/<model_name>", methods=["POST"])
def index(model_name):
    field_name = model_name + "_field"
    field = fields[field_name]

    lines = request.get_data().decode('utf-8').split('\n')
    to_index = {}
    for line in lines[:-1]:
        try:
            passage_obj = json.loads(line)
            doc_id = passage_obj['doc_id']
            passage_id = passage_obj['passage_id']
            passage = passage_obj['passage']
            to_index[(doc_id, passage_id)] = passage
        except Exception as e:
            print(e)

    field.insert(to_index)

    return "Created", 201


@app.route("/search/<model_name>")
def search(model_name):
    args = request.args
    field_name = model_name + "_field"
    field = fields[field_name]
    query = args.get('q')
    results = []
    top_n = field.search(query)
    for row in top_n.to_dict(orient='records'):
        results.append({'score': row['score'],
                        'doc_id': row['id'][0],
                        'passage_id': int(row['id'][1])})
    return jsonify(results)


if __name__ == "__main__":
    load_fields()
    app.run(host="0.0.0.0", port=5001, threaded=True)
