import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def repo_path(path):
    """Turn a path relative to the repository into an absolute path"""
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def load_yaml(path):
    """Read a yaml config file"""
    return yaml.safe_load(repo_path(path).read_text(encoding="utf-8"))


def save_json(path,obj):
    """Write an object to a json file and create the folder when needed"""
    path = repo_path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")


def load_json(path):
    """Read a json file"""
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def write_jsonl(path,rows):
    """Write a list of dictionaries as one json object per line"""
    path = repo_path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")


def read_jsonl(path):
    """Read a file with one json object per line"""
    with repo_path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def set_seed(seed):
    """Seed every random number generator that training uses"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
