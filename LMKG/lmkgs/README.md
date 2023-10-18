# LMKG-S  

LMKG-S is a supervised neural network cardinality estimator for knowledge graphs.


## Quick start

Set up a conda environment with dependencies installed:

```bash
# Install Python environment
conda env create -f environment.yml
conda activate lmkg-s
# Run commands below inside this directory.
```

## Datasets and Queries
To run the model for the SWDF dataset, download everythig from the folder **lmkg-s_data/** from the [dropbox link](https://www.dropbox.com/sh/709rxcpyl631kyk/AAAfbEXXnjQvacrgOPGbg87oa?dl=0) and place inside the folder **final_datasets**.
The queries for the swdf star 2 and chain 2 datasets are located in the same folder and start with the prefix _eval__.  

For LUBM and YAGO, you can download the datasets from the [dropbox link](https://www.dropbox.com/sh/709rxcpyl631kyk/AAAfbEXXnjQvacrgOPGbg87oa?dl=0) and create training and evaluation data of preferred size. The python file is already set up for all the different datasets, you just need to specify the dataset name as a parameter when running the model, i.e., --dataset=swdf.

## Running experiments
The training params are:
```bash
python lmkgs.py --training --query-type=["combined","star","chain","complex"] --query-join=<int>
--batch-size=<int> --epochs=<int> --dataset="swdf" --query-join=<int> --layers=<int>
--neurons=<int> --decay
```
The query type corresponds to the model grouping strategy. If combined (default option) is chosen,
 a single model for star and chain queries is created (in this scenario of size 2). If complex is chosen a single model
 is trained on a set of complex queries.

The eval params are:
```bash
python lmkgs.py --eval --query-type=["combined","star","chain","complex"] --query-join=<int>
--dataset="swdf" --query-join=<int> --layers=<int>
--neurons=<int> --decay --test-mode=["all","star","chain",]
```
The test mode corresponds to the queries over which the end statistics are evaluated.
A combined model can answer both star and chain queries, but to test on them separately the test mode has to be specified.
Otherwise, the test is conducted on both star and chain queries. 
For complex queries the test mode selects 100 queries from each complex query file.

The training for a model for star and chain queries of size 2 (query-size grouping), can be trained in the following way:
```bash
python lmkgs.py --training --query-type="combined" --query-join=2
```

Evaluating the same model can be done using the following command:
```bash
python lmkgs.py --eval --query-type="combined" --query-join=2
```

For training and evaluating a separate model for star 2 or chain 2 use the following commands:
```bash
#train
python lmkgs.py --training --query-type="star" --query-join=2
python lmkgs.py --training --query-type="chain" --query-join=2

#eval
python lmkgs.py --eval --query-type="star" --query-join=2
python lmkgs.py --eval --query-type="chain" --query-join=2
```

## Metrics & Monitoring 

The key metrics to track are
* Cardinality estimation accuracy (Q-error)
    * Separated per ranges, medians, query type and query size
* Training time (seconds)
* Evaluation time 


## Custom Data Processor

The current project works with a specific format of queries.
For using LMKG-S for other query forms, the following steps need to be performed.
1. Reader of the custom queries.
2. Conversion from queries to a QueryGraph instance. 

    The QueryGraph instance requires 
    * n - number of nodes in the knowledge graph 
    * e - number of edges in the knowledge graph
    * d - number of nodes in the query
    * b - number of edges in the query
    
    The QueryGraph receives triples as input where a variable starts with "?". A join is denoted with the usage of the same variable.
    The result of the create_graph method in the QueryGraph instance gives the matrices for the specific query, which need to be concatenated and forwarded to the model.
    
3. Adding the reader to the lmkgs main method. 


<sup>The data_processor_release.py file contains the original star and chain reader. Since the extension of LMKG-S for complex queries, the complex_reader.py and query_graph.py are used. The results for star and chain queries remain the same in using either of the methods. The usage of the latter is advised.</sup>


If you compare with this code, or you use it in your research, please cite:

```
@inproceedings{DBLP:conf/edbt/DavitkovaG022,
  author    = {Angjela Davitkova and
               Damjan Gjurovski and
               Sebastian Michel},
  title     = {{LMKG:} Learned Models for Cardinality Estimation in Knowledge Graphs},
  booktitle = {{EDBT}},
  pages     = {2:169--2:182},
  publisher = {OpenProceedings.org},
  year      = {2022}
}
```   
