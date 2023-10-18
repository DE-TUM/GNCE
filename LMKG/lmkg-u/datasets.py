"""
    Title: Deep Unsupervised Cardinality Estimation Source Code
    Author:  Amog Kamsetty, Chenggang Wu, Eric Liang, Zongheng Yang
    Date: 2020
    Availability: https://github.com/naru-project/naru

    Source Code used as is or modified from the above mentioned source
"""

"""Dataset registrations."""
import common

def LoadStar2(filename='swdf_star_2.csv'):
    csv_file = 'datasets/{}'.format(filename)
    cols = ['predicate', 'object', 'predicate1', 'object1']

    return common.CsvTable('triples', csv_file, cols, do_compression=True, num_submterms=2, comp_threshold=1000, type_casts={})

def LoadChain2(filename='swdf_chain_2.csv'):
    csv_file = 'datasets/{}'.format(filename)
    cols = ['subject', 'predicate', 'object', 'predicate1', 'object1']

    return common.CsvTable('triples', csv_file, cols, do_compression=True, num_submterms=2, comp_threshold=1000, type_casts={})
