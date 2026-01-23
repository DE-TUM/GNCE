# GNCE - Cardinality Estimation over Knowledge Graphs with Embeddings and Graph Neural Networks

This repository contains the implementation of GNCE, a method to predict the cardinality
of conjunctive queries over Knowledge Graphs. It is based on Graph Neural Networks
and Knowledge Graph Embeddings and is implemented in PyTorch. 

Paper: https://dl.acm.org/doi/abs/10.1145/3639299

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Comparing to related Approaches](#comparing-to-related-approaches)
- [License](#license)

## Installation

### Requirements

You need to have Python 3.10 installed. We recommend using [uv](https://docs.astral.sh/uv/) for package management.

### Setup

Install the project and its dependencies using uv:

```sh
uv sync
```

This will create a virtual environment and install all dependencies specified in `pyproject.toml`.

To activate the virtual environment:

```sh
source .venv/bin/activate
```

## Usage

### Data

We expect you to have a Knowledge Graph in the `.nt` or `.ttl` format, as well as
it served over a SPARQL endpoint.

We further expect you to have a file containing the queries you want to predict in
the following format:

```json
{"x": ["http://example.org/entity1", ...], "y": 4, 
"query": ["SELECT * WHERE..."], 
"triples": [["http://example.org/entity1", "http://example.org/predicate1", "http://example.org/entity2"], ...]}
```

Here, `x` is the list of entities that are part of the query, `y` is the cardinality of the query,
`query` is the SPARQL query, and `triples` is the list of triples that are part of the query.

#### Example Data

The used datasets, queries and results from the paper can be found under the following link:
https://nx36303.your-storageshare.de/s/wMbJJ2JLnkXcSE6

### Embedding Generation

The first step is to generate RDF2Vec embeddings for the entities occurring in your Knowledge Graph.

1. **Create a configuration file** in `configs/` (see `configs/embedding_config_example.yml` for reference):

```yaml
dataset_name: "my_dataset"

# Path to the KG file (.nt or .ttl)
kg_file: "data/my_dataset/graph/graph.nt"

# SPARQL endpoint URL (if remote=true) or path to KG file
sparql_endpoint: "http://127.0.0.1:8890/sparql/"

# Whether to use a remote SPARQL endpoint or the KG file via rdflib
remote: true

# Base directory for saving embeddings output
output_dir: "data/"

# Use all entities from KG file (true) or extract from query files (false)
use_all_entities: true

# Query files to extract entities from (used when use_all_entities is false)
query_files:
  - "data/my_dataset/star/queries.json"
  - "data/my_dataset/path/queries.json"
```

2. **Run the embedding generation script**:

```sh
uv run python scripts/generate_embeddings.py --config configs/embedding_config.yml
```

The embeddings will be saved to `{output_dir}/{dataset_name}/{dataset_name}_embeddings.json`.
This file contains a JSON object mapping each entity to its embedding vector and occurrence count:

```json
{
  "http://example.org/entity1": {
    "embedding": [0.1, 0.2, ...],
    "occurence": 42
  },
  ...
}
```

### Training

Next, you can train the GNN model to predict cardinalities.

1. **Create a training configuration file** in `configs/` (see `configs/gnce_example.yaml` for reference):

```yaml
data:
  - dataset_name: "my_dataset-star"
    query_type: "star"
    queries_path: "data/my_dataset/star/queries.json"
    embeddings_path: "data/my_dataset/my_dataset_embeddings.json"
    batch_size: 32
    split: "train-val"       # Use for training and validation
    num_queries: -1          # -1 for all queries

  - dataset_name: "my_dataset-path"
    query_type: "path"
    queries_path: "data/my_dataset/path/queries.json"
    embeddings_path: "data/my_dataset/my_dataset_embeddings.json"
    batch_size: 32
    split: "test"            # Use only for testing
    num_queries: -1

output_dir: "output_gnce/"
epochs: 40
learning_rate: 0.0001
weight_decay: 0.0
seed: 42
device: "auto"              # "cpu" | "cuda" | "auto"
train_ratio: 0.8            # Train/validation split ratio
eval_every: 1
save_every: 5

random_embeddings: false    # Use random embeddings instead of RDF2Vec
use_occurrence: true        # Include entity occurrence counts as features
max_occurrence: null        # Max occurrence value for normalization (null = auto)
```

2. **Run the training script**:

```sh
uv run python scripts/train.py
```

By default, this uses `configs/gnce_local.yaml`. Edit `scripts/train.py` to change the config path.

#### Training Outputs

Training results are saved to `{output_dir}/run_{timestamp}/` with the following structure:

```
output_gnce/run_2026-01-23_12-00-00/
├── config.json                    # Snapshot of training configuration
├── training_progress.json         # Training history (loss, MAE, q-error per epoch)
├── training_progress_by_dataset.json  # Per-dataset training metrics
├── logs/
│   └── train.log                  # Training logs
├── checkpoints/
│   ├── best_model.pt              # Best model (lowest validation loss)
│   ├── checkpoint_epoch_N.pt      # Periodic checkpoints
│   └── final_model.pt             # Final model after training
├── plots/
│   ├── loss.png                   # Train/val loss curves
│   ├── mae.png                    # Train/val MAE curves
│   ├── q_error.png                # Train/val q-error curves
│   └── by_dataset/                # Per-dataset training curves
├── scatter/
│   ├── epoch_0001.png             # Prediction scatter plots per epoch
│   └── by_dataset/                # Per-dataset scatter plots
├── val_predictions/
│   └── epoch_N/                   # Validation predictions per epoch
│       ├── preds.npy              # Predicted cardinalities
│       ├── gts.npy                # Ground truth cardinalities
│       ├── sizes.npy              # Query sizes (number of triples)
│       └── dataset_names.json     # Dataset name per prediction
└── test_predictions/              # Final test set predictions
    ├── preds.npy
    ├── gts.npy
    ├── sizes.npy
    └── dataset_names.json
```

### Comparing to related Approaches

The repository includes code from LMKG, and functionality to connect to code from 
LSS and GCARE. For that the run_LMKG and run_GCARE functions in run_experiments as well as the code in LSS
can be used. Make sure to install the GCARE (https://github.com/yspark-dblab/gcare)
and LSS (https://github.com/Kangfei/LSS) code as instructed there.


## License

This project is licensed under the AGPL-3.0 license - see the LICENSE file for details.
