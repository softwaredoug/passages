from typing import Dict
from flask import Flask, request, jsonify
from sim_field import SimField
from model import Model


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


@app.route("/index/<model_name>")
def index(model_name):
    args = request.args
    field_name = model_name + "_field"
    if field_name not in fields:
        fields[field_name] = SimField(Model(model_name))
    field = fields[field_name]
    passage = args.get('q')
    passage_id = args.get('passage_id')
    doc_id = args.get('doc_id')
    field.index({(doc_id, passage_id): passage}, skip_updates=True)

    return "Created", 201


@app.route("/search/<model_name>")
def search(model_name):
    args = request.args
    field_name = model_name + "_field"
    if field_name not in fields:
        fields[field_name] = SimField(Model(model_name))
    field = fields[field_name]
    query = args.get('q')
    results = []
    top_n = field.search(query)
    for row in top_n.iterrows():
        results.append({'doc_id': row[0][0],
                        'passage_id': int(row[0][1])})
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
