def fetch_json(path: str) -> dict:
    import json
    with open(path, 'r') as f:
        return json.load(f)