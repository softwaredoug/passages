# passages
Flask server to compute and cache sentence transformer embeddings

```
flask --app passages run --port 5001 
```

Get embeddings

```
http://localhost:5000/encode/all-MiniLM-L6-v2?q=hello+world
```

Get cache stats

```
http://localhost:5000/stats/
```
