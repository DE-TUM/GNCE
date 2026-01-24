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
        weighted: bool = True,
        shuffle: bool = False,
        **kwargs,
    ):
        """
        Initializes the MultigraphLoader.

        Args:
            data (Union[NTDatasets, List[NTDatasets]]): A single dataset or a list of datasets.
            queries (Union[NTQueries, List[NTQueries]]): A single query set or a list of query sets.
            batch_size (Union[int, List[int]], optional): Batch size for each dataset.
                If an int, it is applied to all datasets. Defaults to 1.
            weighted (bool, optional): If True, samples datasets based on the number of
                remaining queries. If False, samples uniformly. Defaults to True.
            shuffle (bool, optional): Whether to shuffle the query loaders. Defaults to False.
            **kwargs: Additional arguments passed to the loader (unused).
        """
        if not isinstance(data, (list, tuple)):
            data = [data]
        if not isinstance(queries, (list, tuple)):
            queries = [queries]
        if not isinstance(batch_size, (list, tuple)):
            batch_size = [batch_size] * len(data)

        assert len(data) == len(queries) == len(batch_size), (
            "All input lists must have the same length."
        )

        self.data = data
        self.queries = queries
        self.batch_size = batch_size
        self.weighted = weighted
        self.shuffle = shuffle

        self.queries_loaders = []
        for i in range(len(self.data)):
            loader = DataLoader(
                dataset=self.queries[i],
                batch_size=self.batch_size[i],
                shuffle=self.shuffle,
            )
            self.queries_loaders.append(loader)

        self.queries_iters = []
        self.remaining_counts = []
        self.length = sum([len(ql) for ql in self.queries_loaders])

    def _get_random_loader(self):
        """
        Selects a random loader index from the active loaders.

        Returns:
            int: The index of the selected loader, or None if no loaders are active.
        """
        active_loaders = [
            i for i, count in enumerate(self.remaining_counts) if count > 0
        ]
        if active_loaders == []:
            return None

        if self.weighted:
            active_weights = [self.remaining_counts[i] for i in active_loaders]
            weights = torch.tensor(active_weights, dtype=torch.float32)

            # use torch.multinomial with uniform weights to choose a loader index
            rel_idx = int(torch.multinomial(weights, 1).item())
            return active_loaders[rel_idx]
        else:
            return random.choice(active_loaders)

    def __next__(self):
        """
        Retrieves the next batch from the sampled loader.

        Returns:
            Tuple[NTDatasets, Data]: A tuple containing the dataset and the query batch.

        Raises:
            StopIteration: When all loaders are exhausted.
        """
        while True:
            idx = self._get_random_loader()

            if idx is None:
                raise StopIteration

            try:
                batch = next(self.queries_iters[idx])
                self.remaining_counts[idx] -= 1
                return self.data[idx], batch
            except StopIteration:
                # remove exhausted loader and continue
                self.remaining_counts[idx] = 0
                if sum(self.remaining_counts) == 0:
                    raise StopIteration

    def __iter__(self):
        """
        Initializes the iterators for each query loader.

        Returns:
            MultigraphLoader: The loader instance itself.
        """
        self.queries_iters = [iter(loader) for loader in self.queries_loaders]
        self.remaining_counts = [len(loader) for loader in self.queries_loaders]
        return self

    def __len__(self):
        """
        Returns the total number of batches across all loaders.

        Returns:
            int: The total length of the loader.
        """
        return self.length
