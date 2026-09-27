import json
from config import maclar, SPLIT_PATH

train_keys = ["liverpool-realmadrid", "galatasaray-juventus"]
val_keys = ["fcsb-fenerbahce"]
test_keys = ["shkendija-samsunspor"]


all_matches = {
    "train_keys": train_keys,
    "val_keys": val_keys,
    "test_keys": test_keys
}


for key in train_keys + val_keys + test_keys:
    assert key in maclar, f"{key} maçı configte yok!"



with open(SPLIT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_matches, f, indent=2)