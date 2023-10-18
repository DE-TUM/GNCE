# LMKG-U

LMKG-U is an unsupervised autoregressive model for cardinality estimation in knowledge graphs. 


## Quick start

Set up a conda environment with depedencies installed:

```bash
# Install Python environment
conda env create -f environment.yml
conda activate lmkg-u
# Run commands below inside this directory.
```

## Datasets and Queries
To run the model for the SWDF dataset, download everythig from the folder **/lmkg-u_data/** from the [dropbox link](https://www.dropbox.com/sh/709rxcpyl631kyk/AAAfbEXXnjQvacrgOPGbg87oa?dl=0) and place inside the folder **datasets/**.
The queries for the swdf star 2 and chain 2  datasets are located in the **queries/** folder.  

For LUBM and YAGO, you can download the datasets from the **datsets_lmkg/** folder from the [dropbox link](https://www.dropbox.com/sh/709rxcpyl631kyk/AAAfbEXXnjQvacrgOPGbg87oa?dl=0) and create training and evaluation data of preferred size. To be able to read the dataset you need to add a corresponding method in the `datasets.py` file by following the code of the already created methods.

## Running experiments
Once you have downloaded the dataset, to train the model for the sdwf star 2 (chain 2) dataset you should run the following command:

```bash
# Train on swdf_star_2
python train_model.py --dataset=swdf_star_2 --epochs=10 --bs=1024  --residual --layers=4 --fc-hiddens=512 --direct-io --input-encoding=binary --output-encoding=one_hot --column-masking --warmups=8000

# Train on swdf_chain_2
python train_model.py --dataset=swdf_chain_2 --epochs=10 --bs=1024  --residual --layers=4 --fc-hiddens=512 --direct-io --input-encoding=binary --output-encoding=one_hot --column-masking --warmups=8000
```

For the `input-encoding` and `output-encoding`, we have performed the experiments by choosing either:
  - `--input-encoding=binary` and `--output-encoding=one_hot`, or 
  - `--input-encoding=embed` and `--output-encoding=embed`

which was always combined with our term compression by setting the parameter `do_compresion=True` in the respective dataset method. Additionally,
the compression threshold as well as the number of subterms the term should be split into can also be set in the dataset methods in the file `datasets.py`. 

The parameter `--column-masking` is required for queries involving variables. See the paper for more details.

Once trained, the models will be stored under the **models/** directory. 

To evaluate the created model for the swdf star 2 (chain 2) dataset you can run the following commands:

```bash
# Eval swdf_star_2
python eval_model.py --dataset=swdf_star_2 --glob='swdf_star_2-*.pt' --residual --layers=4 --fc-hiddens=512 --direct-io --input-encoding=binary --output-encoding=one_hot --column-masking --query-type='star'

# Eval swdf_chain_2
python eval_model.py --dataset=swdf_chain_2 --glob='swdf_chain_2-*.pt' --residual --layers=4 --fc-hiddens=512 --direct-io --input-encoding=binary --output-encoding=one_hot --column-masking --query-type='chain'
```
 

## Metrics & Monitoring 

The key metrics to track are
* Cardinality estimation accuracy (Q-error)
    * Separated per ranges, medians, query type and query size
* Training time (seconds)
* Evaluation time 


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
