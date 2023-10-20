# GNCE - Cardinality Estimation over Knowledge Graphs with Embeddings and Graph Neural Networks

This repository contains the implementation of GNCE, a method to predict the cardinality
of conjunctive queries over Knowledge Graphs. It is based on Graph Neural Networks
and Knowledge Graph Embeddings and is implemented in PyTorch. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Comparing to related Approaches](#comparing-to-related-approaches)
- [License](#license)

## Installation

### Requirements

You need to have Python 3 installed. The code was tested with Python 3.8.10

### Setup

You can install the required packages with the following command:

```sh
$ pip install -r requirements.txt
```
You might adapt the version of the torch and torch-geometric packages to your needs,
based on your CUDA version.

Note that we have included an adapted version of the pyrdf2vec code from
https://github.com/IBCNServices/pyRDF2Vec that saves the random walks to disk for 
scaling to large graphs on systems with limited RAM.
## Usage

### Data
We except you to have a Knowledge Graph in the .ttl or .nt format, as well as
it served over a SPARQL endpoint. <br>
We further expect you to have a file containing the queries you want to predict in
the following format:
```
{"x": ["http://example.org/entity1", ...], "y":4, 
"query": ["SELECT * WHERE..."], 
"triples": [["http://example.org/entity1", "http://example.org/predicate1", "http://example.org/entity2"], ...]}
```
}
<br>
Here, "x" is the list of entities that are part of the query, "y" is the cardinality of the query,
"query" is the SPARQL query, and "triples" is the list of triples that are part of the query.<br>


#### Example Data
The used datasets, queries and results from the paper can boe found under the following link:
https://nx36303.your-storageshare.de/s/wMbJJ2JLnkXcSE6

### Embedding Generation
The first step is to generate embeddings for the entities occurring in the given queries.
For that, the file `embeddings_generator.py` is used. In the corresponding
main function, set the value of QUERY_FILE_PATH to the path of the file containing the queries
in the above format, set KG_FILE_PATH to the path of the Knowledge Graph file(.ttl or .nt), 
and set KG_ENDPOINT to the SPARQL endpoint of the Knowledge Graph. KG_NAME is the name you 
assign to your dataset. <br>
The resulting will be saved relative to the file,
under Datasets/KG_NAME/statistics. One file will be saved per entity containing the embeddings
as well as the occurrence of the entity. <br>

Make sure that the folder structure exists like:
```
/Datasets
    /KG_NAME
        /graph
            graph.nt
        /Results
        /query_type
            query_file.json
        /statistics
            /entity1.json
            /entity2.json
            ...
```

### Training

Next, you can train the GNN model to predict the cardinalities. 
For that, the file `run_experiments.py` is used.
Here, set the dataset as well as query_type to the same values as in the previous step. The model will be trained on 80% of the given queries and evaluated on the remaining 20%. 
The best model will be saved under ```model.pth```. Furthermore,
the predictions, true cardinalities, query sizes and prediction times of the test set
will be saved under
```Datasets/KG_NAME/Results``` as 4 separate nummpy arrays as well as in a json file.


### Comparing to related Approaches
The repository includes code from LMKG, and functionality to connect to code from 
LSS and GCARE. For that the run_LMKG and run_GCARE functions in run_experiments as well as the code in LSS
can be used. Make sure to install the GCARE (https://github.com/yspark-dblab/gcare)
and LSS (https://github.com/Kangfei/LSS) code as instructed there.


## License

This project is licensed under the AGPL-3.0 license - see the LICENSE file for details.
