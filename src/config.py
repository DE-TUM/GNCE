from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class GNCEDataConfig:
    dataset_name: str
    query_type: str = "star"
    queries_path: str = ""
    embeddings_path: str = ""
    batch_size: int = 32
    split: Literal["train", "validation", "test", "train-val"] = "train-val"
    num_queries: int = -1


@dataclass
class GNCEConfig:
    data: List[GNCEDataConfig]
    output_dir: str = "output_gnce"
    epochs: int = 40
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cpu"  # "cpu" | "cuda" | "auto"
    train_ratio: float = 0.8  # used when split == "train-val"
    eval_every: int = 1
    save_every: int = 5
    random_embeddings: bool = False
    use_occurrence: bool = True
    max_occurrence: Optional[int] = None


def load_config(path: str | Path) -> GNCEConfig:
    """
    Load config from JSON (default) or YAML if PyYAML is available.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in [".json"]:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    elif suffix in [".yaml", ".yml"]:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install it or use a .json config."
            ) from e
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config extension: {suffix} (use .json or .yaml/.yml)")

    data_cfgs = [GNCEDataConfig(**dc) for dc in raw["data"]]
    raw = {**raw, "data": data_cfgs}
    return GNCEConfig(**raw)


