import json


def fetch_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def dump_json(path: str, data: dict) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
