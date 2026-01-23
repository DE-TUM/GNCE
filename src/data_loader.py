from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Union

import torch
from torch_geometric.loader import DataLoader


@dataclass
class DataComponents:
    train_loader: object
    validation_loader: object
    test_loader: object


class MultigraphLoader:
    """
    Simple multi-dataset loader: yields batches from multiple PyG DataLoaders.

    Each iterator step returns `(meta, batch)` where `meta` is the corresponding
    dataset's metadata object (or anything the caller provided in `data=`).
    """

    def __init__(
        self,
        data: Union[object, Sequence[object]],
        queries: Union[torch.utils.data.Dataset, Sequence[torch.utils.data.Dataset]],
        batch_size: Union[int, Sequence[int]] = 1,
        fair_chance: bool = True,
        shuffle: bool = False,
        **kwargs,
    ):
        if not isinstance(data, (list, tuple)):
            data = [data]
        if not isinstance(queries, (list, tuple)):
            queries = [queries]
        if not isinstance(batch_size, (list, tuple)):
            batch_size = [batch_size] * len(data)

        assert len(data) == len(queries) == len(batch_size), "All input lists must have the same length."

        self.data = list(data)
        self.queries = list(queries)
        self.batch_size = list(batch_size)
        self.fair_chance = fair_chance
        self.shuffle = shuffle

        self.queries_loaders: List[DataLoader] = []
        for i in range(len(self.data)):
            loader = DataLoader(
                dataset=self.queries[i],
                batch_size=self.batch_size[i],
                shuffle=self.shuffle,
                **kwargs,
            )
            self.queries_loaders.append(loader)

        self.queries_iters: List[object] = []
        self.empty: List[bool] = []
        self.length = sum(len(ql) for ql in self.queries_loaders)

    def _get_random_loader_idx(self) -> int:
        if not self.fair_chance:
            weights = torch.ones(len(self.queries_iters), dtype=torch.float32)
            idx = int(torch.multinomial(weights, 1).item())
        else:
            idx = random.randint(0, len(self.queries_iters) - 1)

        if self.empty[idx]:
            return self._get_random_loader_idx()
        return idx

    def __iter__(self):
        self.queries_iters = [iter(loader) for loader in self.queries_loaders]
        self.empty = [False] * len(self.queries_loaders)
        return self

    def __next__(self):
        if all(self.empty):
            raise StopIteration

        idx = self._get_random_loader_idx()
        try:
            batch = next(self.queries_iters[idx])
        except StopIteration:
            self.empty[idx] = True
            if all(self.empty):
                raise StopIteration
            return self.__next__()

        return self.data[idx], batch

    def __len__(self):
        return self.length


