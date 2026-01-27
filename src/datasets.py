from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset

from src.utils import get_query_graph_data_new


@dataclass(frozen=True)
class GNCEGraphMeta:
    dataset_name: str
    embeddings_path: str
    queries_path: str
    query_type: str


def _short_hash(*parts: str, n: int = 10) -> str:
    h = hashlib.md5(("|".join(parts)).encode("utf-8")).hexdigest()
    return h[:n]


def _make_undirected_with_direction(data: Data) -> Data:
    """Create reverse edges and set the last edge_attr dimension to -1 for reverse edges."""
    if getattr(data, "edge_attr", None) is None:
        raise ValueError("Expected homogeneous Data with edge_attr containing direction as last dim.")

    edge_index = data.edge_index
    edge_attr = data.edge_attr

    rev_edge_index = edge_index.flip(0)
    rev_edge_attr = edge_attr.clone()
    rev_edge_attr[:, -1] = -1

    data.edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
    data.edge_attr = torch.cat([edge_attr, rev_edge_attr], dim=0)

    if getattr(data, "edge_type", None) is not None:
        data.edge_type = torch.cat([data.edge_type, data.edge_type.clone()], dim=0)

    return data


class GNCEQueries(InMemoryDataset):
    """
    GNCE query dataset that combines static node/edge embeddings and occurrences into the  query graph.

    It mirrors the FICE `NTQueries` concept, but uses GNCE's embedding JSON format:
    `{iri: {"embedding": [...], "occurence": int}}` for entities and relations.
    """

    def __init__(
        self,
        queries_path: str | Path,
        embeddings_path: str | Path,
        dataset_name: str,
        query_type: str,
        root: Optional[str | Path] = None,
        transform=None,
        pre_transform=None,
        random_embeddings: bool = False,
        use_occurrence: bool = True,
        max_occurrence: Optional[int] = None,
    ):
        self.queries_src = str(Path(queries_path).expanduser().resolve())
        self.embeddings_src = str(Path(embeddings_path).expanduser().resolve())
        self.dataset_name = dataset_name
        self.query_type = query_type
        self.random_embeddings = random_embeddings
        self.use_occurrence = use_occurrence
        self.max_occurrence = max_occurrence

        if root is None:
            root = Path(__file__).parent / "gnce_data" / dataset_name
        else:
            root = Path(root)

        # Make dataset instance unique per (queries_path, embeddings_path, knobs)
        self._instance_id = _short_hash(
            Path(self.queries_src).name,
            Path(self.embeddings_src).name,
            f"re={int(self.random_embeddings)}",
            f"occ={int(self.use_occurrence)}",
            f"maxocc={self.max_occurrence}",
        )

        super().__init__(str(root), transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        qname = Path(self.queries_src).name
        ename = Path(self.embeddings_src).name
        return [f"{self._instance_id}_{qname}", f"{self._instance_id}_{ename}"]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"gnce_queries_{self.query_type}_{self._instance_id}.pt"]

    def download(self):
        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        dst_queries = raw_dir / self.raw_file_names[0]
        dst_emb = raw_dir / self.raw_file_names[1]

        if not dst_queries.exists():
            shutil.copyfile(self.queries_src, dst_queries)
        if not dst_emb.exists():
            shutil.copyfile(self.embeddings_src, dst_emb)

    def process(self):
        raw_queries_path = Path(self.raw_paths[0])
        raw_embeddings_path = Path(self.raw_paths[1])

        with raw_embeddings_path.open("r", encoding="utf-8") as f:
            embeddings: Dict[str, Dict[str, Any]] = json.load(f)

        with raw_queries_path.open("r", encoding="utf-8") as f:
            queries: List[Dict[str, Any]] = json.load(f)

        pyg_graphs: List[Data] = []

        for q in queries:
            # Build hetero query graph with embeddings + occurrence
            hetero = get_query_graph_data_new(
                q,
                embeddings,
                device="cpu",
                random_embeddings=self.random_embeddings,
                use_occurrence=self.use_occurrence,
                max_occurrence=self.max_occurrence,
            )

            # Debug: print edge_attr dimensions for each edge type
            for edge_type in hetero.edge_types:
                ea = hetero[edge_type].edge_attr
                print(f"Edge type {edge_type}: edge_attr shape = {ea.shape}")

            # Convert to homogeneous and make undirected with direction attribute
            g = hetero.to_homogeneous()
            g = _make_undirected_with_direction(g)

            y = q.get("y", None)
            if y is None:
                raise KeyError("Query object missing key 'y'.")
            g.y = torch.tensor(float(y), dtype=torch.float32)
            g.num_triples = int(len(q.get("triples", [])))

            # Metadata for logging/analysis
            g.dataset_name = self.dataset_name
            g.type = self.query_type
            g.query_set_name = self.dataset_name

            pyg_graphs.append(g)

        torch.save(self.collate(pyg_graphs), self.processed_paths[0])


