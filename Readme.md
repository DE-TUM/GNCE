# SPARQL cardinality estimation using Graph Neural Networks and Knowledge Graph Embeddings

## Introduction
This repository contains the code alongside the paper "SPARQL cardinality estimation using Graph Neural Networks and Knowledge Graph Embeddings".

It can be used to train and create a machine learning model that predicts cardinalities of BGP queries over rdf graphs.

## Requirements

The following requirements need to be met in order to use the repository:

- Virtuoso Conductor needs to be installed and running
- python 3.8.x installed
- python packages
  - sklearn 1.1.2
  - requests 2.28.1
  - torch 1.12.1
  - torch geometric 2.1.0
  - pyrdf2vec 0.2.3
  - numpy 1.23
  - matplotlib
  - pyodbc 4.0.34
  - tqdm
  - rdflib 6.1.1
  - SPARQLWrapper 2.0.0
  - pandas

## Workflow

The initial assumption is that you have given a rdf graph in the .ttl or .nt format.

For that graph, we first calculate statistics as well as embeddings for each entity.
After that, random test queries are generated and performed against the graph to acquire the
true cardinality.
Lastly, a model is trained on the queries to estimate the cardinalities.

In the following is described how to perform those steps.

### Embeddings and Statistics Generation

The function `get_embeddings` in `embeddings_generator.py` is used to calculate occurences for each entity as well as 
numerical embeddings using RDF2VEC.
The functions receives a `dataset_name`, which is used to store the statistics and rdf2vec model like "dataset_name_embeddings.json"
The second argument is the path to the rdf file of the KG. The arguments `remote` and `sparql_endpoint` are optional.
`remote` is a flag to determine whether an in memory access to the rdf file should be used(False) or whether the rdf graph
is served over a SpARQL Endpoint(True). In the latter case `sparql_endpoint` can be used to set the correct url. Lastly, he parameter
`entities` can be used to provide a list of entities for whom entities should be calculated. If not set, all found entities
in the graph will be used.

The function will first contact the rdf graph to get the total number of triples each entity participated in(occurence).
Next, it will calculate embeddings for each given entity. Lastly, the occurences will be saved together with the embeddings
as a dict to a json file of the form
{"entity_1":{"embedding":[0.8,...], "occurence":21},...}

### Query Generation

To produce a dataset of test and train queries, the function `get_testdata` in `query_generator.py` can be used. The
first argument `graphfile` is again the path to the .ttl file, `dataset_name` is the name of the graph or dataset and used 
to store the generated queries in a file `dataset_name_test_data.json`. `n_triples` denotes how many triples are present
in the query. Currently, the generated queries are only star shaped queries. `n_queries` denotes how many queries should
be generated. Finally, `endpoint_url` is the url where the SPARQL Endpoint serving the rdf graph is hosted.
The function generates and saves the queries in a json dict of the form
[{"x": ["entity_1,...], "y": 232, "query": "SELECT * ..", "triples": [["entity_3, ?p1, "entity_14"], [...],..]}]. x is a
list of all occuring bounded entities in the query grapgh, y is the true cardinality of the query performed over the rdf
graph, query is a string of the raw SPARQL query, and triples is a list of lists containing each subject, predicate and
object of the triples in the query.

### Model Training

Currently, training of 2 different kinds of models is supported. The first kind is a very simple model while the second
makes use of a Graph Neural Network.

### Simple model

In the simple model, the true cardinality is estimated as

<img src="simple_model_probability.svg">

Here, the product goes over all N entities in the query graph(irrelevant of their order). o_i is the occurence of 
entity i, |G| is the number of triples in the graph, e_i is the embedding of entity i and f is an in general arbitrary
function. In our case, f is realized as k-nearest neighbor regression.
The simple model can be trained using `cardinality_estimator.train_simple_model` function. The function receives a trainset
and a testset of queries. The cardinality for each query in the trainset is estimated using the above product rule. Afterwards,
a k-nearest neighbor model is fitted to the predicted cardinalities and the true cardinalities. Finally, this model is
evaluated on the given testset in terms of mean absolute error and mean q-error.

### GNN based model 

The simple model has a few shortcomings. First, it does ignure the graph structure completely. Thus, different queries 
with the same entities will be mapped to the same cardinality. Further, it has no learnable parameters in calculating
the conditional probabilities and just assumes that the embeddings are enough to estimate the cardinality.
To eliviate the problem, a GNN is used instead of the above. The cardinality is then estimated as C = GNN(query).
The GNN is implemented using pytorch geometric. To train the GNN, the function `cardinality_estimator.train_GNN` can be used.
The function again recevies a train and a testset of queries. A GNN architecture is trained on the trainset and evaluated
on the testset, again in terms of MAE and QError. 


