import json
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import os
import re


def set_rand_seed(seed: int, rank: int = 0, force_deterministic: bool = False) -> None:
    """
    Set the random seed for torch and numpy.
    """
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    os.environ["PYTHONHASHSEED"] = str(seed + rank)
    if force_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def natural_sort_key(text):
    """
    A key for natural number sorting using re.split.
    Example: 'version_10.2' -> ['version_', 10, '.', 2, '']
    """
    text = str(text)
    parts = re.split(r"(\d+)", text)
    parts[1::2] = map(int, parts[1::2])
    return parts


def load_jsonl(filepath: str) -> list[dict]:
    """
    Read a JSONL file and return a list of JSON objects.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"The file '{filepath}' was not found.")

    json_list = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            try:
                json_obj = json.loads(line.strip())
                json_list.append(json_obj)
            except json.JSONDecodeError:
                print(f"[Error] Could not parse JSON on line {i}. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred on line {i}: {e}")
    return json_list


def save_jsonl(data: list[dict], filepath: str | Path) -> None:
    """
    Save a list of JSON objects to a JSONL file.

    Skips saving if data or filepath is None.
    """
    if data is None or filepath is None:
        return
    filepath = Path(filepath)
    if not filepath.suffix == ".jsonl":
        raise ValueError("The filepath must have a .jsonl extension.")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for item in data:
            f.write(f"{json.dumps(item)}\n")


def load_json(filepath: str) -> dict:
    """
    Load a JSON file and return its contents as a dictionary.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"The file '{filepath}' was not found.")

    with open(filepath, "r") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"An error occurred while reading the JSON file: {e}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}


def save_json(data: dict, filepath: str | Path) -> None:
    """
    Save a dictionary to a JSON file.

    Skips saving if data or filepath is None.
    """
    if data is None or filepath is None:
        return
    filepath = Path(filepath)
    if not filepath.suffix == ".json":
        raise ValueError("The filepath must have a .json extension.")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        try:
            json.dump(data, f, indent=4)
        except Exception as e:
            print(f"An error occurred while writing to the JSON file: {e}")


def load_parquet(filepath: str) -> pd.DataFrame:
    """
    Read a Parquet file and return a pandas DataFrame.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"The file '{filepath}' was not found.")
    return pd.read_parquet(filepath)


def save_parquet(data: pd.DataFrame | dict | list[dict], filepath: str | Path) -> None:
    """
    Save data to a Parquet file. Accepts a pandas DataFrame, a dictionary,
    or a list of dictionaries.
    """
    if data is None or filepath is None:
        return
    filepath = Path(filepath)
    if not filepath.suffix == ".parquet":
        raise ValueError("The filepath must have a .parquet extension.")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise TypeError(
            "Data must be a pandas DataFrame, a dictionary, or a list of " + "dictionaries."
        )

    df.to_parquet(filepath, index=False, engine="pyarrow")


def load_yaml(filepath: str, resolve: bool = True) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"The file '{filepath}' was not found.")

    yaml = OmegaConf.load(filepath)
    OmegaConf.set_struct(yaml, True)
    if resolve:
        OmegaConf.resolve(yaml)
    return yaml


def save_yaml(data: dict, filepath: str | Path) -> None:
    """
    Save a dictionary to a YAML file.

    Skips saving if data or filepath is None.
    """
    if data is None or filepath is None:
        return
    filepath = Path(filepath)
    if filepath.suffix not in [".yaml", ".yml"]:
        raise ValueError("The filepath must have a .yaml or .yml extension.")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    yaml = OmegaConf.create(data)
    OmegaConf.save(yaml, filepath)
