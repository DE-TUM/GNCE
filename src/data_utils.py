from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Subset

from src.config import GNCEConfig, GNCEDataConfig
from src.datasets import GNCEGraphMeta, GNCEQueries
from src.data_loader import DataComponents, MultigraphLoader


def _subset_first_n(ds, n: int):
    if n is None or n < 0:
        return ds
    n = min(n, len(ds))
    return Subset(ds, list(range(n)))


def _train_val_split(ds, train_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    n = len(ds)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_n = int(n * train_ratio)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]
    return Subset(ds, train_idx), Subset(ds, val_idx)


def prepare_gnce_data(config: GNCEConfig) -> DataComponents:
    train_meta: List[GNCEGraphMeta] = []
    train_sets: List[torch.utils.data.Dataset] = []
    train_bs: List[int] = []

    val_meta: List[GNCEGraphMeta] = []
    val_sets: List[torch.utils.data.Dataset] = []
    val_bs: List[int] = []

    test_meta: List[GNCEGraphMeta] = []
    test_sets: List[torch.utils.data.Dataset] = []
    test_bs: List[int] = []

    for dc in config.data:
        assert isinstance(dc, GNCEDataConfig)

        queries_path = Path(dc.queries_path).expanduser().resolve()
        embeddings_path = Path(dc.embeddings_path).expanduser().resolve()
        if not queries_path.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        meta = GNCEGraphMeta(
            dataset_name=dc.dataset_name,
            embeddings_path=str(embeddings_path),
            queries_path=str(queries_path),
            query_type=dc.query_type,
        )

        ds = GNCEQueries(
            queries_path=queries_path,
            embeddings_path=embeddings_path,
            dataset_name=dc.dataset_name,
            query_type=dc.query_type,
            root=Path(__file__).parent / "gnce_data" / dc.dataset_name,
            random_embeddings=config.random_embeddings,
            use_occurrence=config.use_occurrence,
            max_occurrence=config.max_occurrence,
        )

        ds = _subset_first_n(ds, dc.num_queries)

        if dc.split == "train":
            train_meta.append(meta)
            train_sets.append(ds)
            train_bs.append(dc.batch_size)

        elif dc.split == "validation":
            val_meta.append(meta)
            val_sets.append(ds)
            val_bs.append(dc.batch_size)

        elif dc.split == "test":
            test_meta.append(meta)
            test_sets.append(ds)
            test_bs.append(dc.batch_size)

        elif dc.split == "train-val":
            train_ds, val_ds = _train_val_split(ds, train_ratio=config.train_ratio, seed=config.seed)
            train_meta.append(meta)
            train_sets.append(train_ds)
            train_bs.append(dc.batch_size)
            val_meta.append(meta)
            val_sets.append(val_ds)
            val_bs.append(dc.batch_size)
        else:
            raise ValueError(f"Unknown split: {dc.split}")

    if not train_sets:
        raise ValueError("No training datasets configured (need at least one with split=train or split=train-val).")

    train_loader = MultigraphLoader(data=train_meta, queries=train_sets, batch_size=train_bs, shuffle=True)
    validation_loader = MultigraphLoader(data=val_meta or train_meta, queries=val_sets or train_sets, batch_size=val_bs or train_bs, shuffle=False)
    test_loader = MultigraphLoader(data=test_meta or val_meta or train_meta, queries=test_sets or val_sets or train_sets, batch_size=test_bs or val_bs or train_bs, shuffle=False)

    return DataComponents(train_loader=train_loader, validation_loader=validation_loader, test_loader=test_loader)


