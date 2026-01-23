import os.path

import torch
from torch_geometric.data import Data, HeteroData, Dataset
import torch_geometric.transforms as T

import numpy as np
import json
import random
import torch_geometric
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected
from typing import Union
from torch import Tensor
import copy
import inspect


class ToUndirectedCustom(BaseTransform):
    r"""
    Custom ToUndirected transform that does not merge the reverse edges,
    but changes the last dimension of the edge attributes to -1


    Converts a homogeneous or heterogeneous graph to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge
    :math:`(i,j) \in \mathcal{E}` (functional name: :obj:`to_undirected`).
    In heterogeneous graphs, will add "reverse" connections for *all* existing
    edge types.

    Args:
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)
        merge (bool, optional): If set to :obj:`False`, will create reverse
            edge types for connections pointing to the same source and target
            node type.
            If set to :obj:`True`, reverse edges will be merged into the
            original relation.
            This option only has effects in
            :class:`~torch_geometric.data.HeteroData` graph data.
            (default: :obj:`True`)
    """
    def __init__(self, reduce: str = "add", merge: bool = True):
        self.reduce = reduce
        self.merge = merge

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            nnz = store.edge_index.size(1)

            if isinstance(data, HeteroData) and (store.is_bipartite()
                                                 or not self.merge):
                src, rel, dst = store._key

                # Just reverse the connectivity and add edge attributes:
                row, col = store.edge_index
                rev_edge_index = torch.stack([col, row], dim=0)

                inv_store = data[dst, f'rev_{rel}', src]
                inv_store.edge_index = rev_edge_index
                for key, value in store.items():
                    if key == 'edge_index':
                        continue
                    if isinstance(value, Tensor) and value.size(0) == nnz:
                        inv_value = copy.deepcopy(value)
                        inv_value[0, -1] = -1
                        inv_store[key] = inv_value
                        #value[0, -1] = -1
                        #print(value.shape)
                        #rint(value)

            else:
                keys, values = [], []
                for key, value in store.items():
                    if key == 'edge_index':
                        continue

                    if store.is_edge_attr(key):
                        keys.append(key)
                        values.append(value)

                store.edge_index, values = to_undirected(
                    store.edge_index, values, reduce=self.reduce)

                for key, value in zip(keys, values):
                    store[key] = value

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class StatisticsLoader():
    """
    This class is used to load the statistics of the dataset
    It wraps the folder containing the statistics in a dict like object
    """
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path
    def __getitem__(self, item):
        try:
            with open(os.path.join(self.statistics_path, item.replace("/", "|") + ".json")) as f:
                return json.load(f)

        except FileNotFoundError:
            print("Cant find embedding for", item)
            raise
            # Return a random embedding
            statistic_dict = {"embedding": [random.uniform(0, 1) for i in range(100)], "occurence": 0}
            return statistic_dict




def get_query_graph_data_new(query_graph, statistics, device, unknown_entity='false', n_atoms: int = None, random_embeddings=False, use_occurrence=True, max_occurrence=None):
    """
    This function is used to get the graph data object from a query graph
    :param query_graph: Dict representing the query graph of the form {"triples": [triple1, triple2, ...], "y": cardinality,
    "query": String of sparql query, "x": List of occurring entities}
    :param statistics: Dict or dict like loader for the embeddings of entities
    :param device: cpu or cuda
    :param unknown entity: Determines if an entity receives its embedding('false'), randomly, embedding
    or random vector('random') or always a random embedding('true')
    :param random_embeddings: If True, use binary vector representations instead of embeddings for entities and predicates
    :param use_occurrence: If True, use actual occurrence values; if False, use placeholder value of 1
    :param max_occurrence: Maximum occurrence value for normalization. If None, uses 16018 (backward compatibility).
                          Occurrence values are normalized using log normalization: log(occurrence+1)/log(max_occurrence+1)
    :return: Graph data object
    """
    import math

    data = HeteroData()
    data = data.to(device)
    node_mapping = {}
    n_edge_variables = 0 # todo remove, for testing
    n_entities = 0 # todo remove, only for testing
    n_edge_variables_random = random.randint(0, 5) # todo remove, for testing
    n_entities_random = random.randint(0, 5) # todo remove, only for testing
    
    # Set normalization factor for occurrence values
    occurrence_norm_factor = max_occurrence if max_occurrence is not None else 16018
    node_embeddings = []
    # Embeddings for variables in edges or nodes
    # edge has an additional dimension to indicate the direction of the edge
    variable_embedding_edge = np.ones(102)
    variable_embedding = np.ones(101)
    np.random.seed(0)
    unknown_entity_embedding = list(np.random.rand(100))
    # Reset the seed

    # Random number indicating if to use embeddings or not
    rand_num = random.random()


    np.random.seed(None)

    # Set to count the total unique atoms in the query
    atom_set = set()

    feature_vector = None

    USE_EMBEDDING = True
    USE_OCCURRENCE = True

    # Whether to shuffle triples and start with a random integer for variable denoting
    shuffled =True

    triple_list = query_graph["triples"]

    # Todo remove, for testing effect of variable enumeration
    if shuffled:
        random.shuffle(triple_list)
    for triple in query_graph["triples"]:

        atom_set.update(triple)
        try:
            s = triple[0].replace("<", "").replace(">", "")
            o = triple[2].replace("<", "").replace(">", "")
        except:
            print(query_graph)
            raise
        # If predicate is a variable, add the variable embedding to the edge attributes
        if "?" in triple[1]:
            p = "237"
            try:
                v = variable_embedding_edge
                if shuffled:
                    v[-1] = n_edge_variables_random #todo
                else:
                    v[-1] = n_edge_variables
                n_edge_variables_random +=1 #todo
                n_edge_variables += 1
                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([v])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                # If the edge does not exist yet, create it
                v = variable_embedding_edge
                if shuffled:
                    v[-1] = n_edge_variables_random #todo
                else:
                    v[-1] = n_edge_variables #todo

                n_edge_variables_random += 1 #todo
                n_edge_variables += 1
                data["entity", p, "entity"].edge_attr = torch.tensor([v])

        else:
            try:
                #p = int(triple[1].replace("<http://example.com/", "").replace(">", ""))
                p = triple[1].replace("<http://example.com/", "").replace(">", "")

            except:
                print(triple)
                raise
            try:
                # Get the embedding of the predicate
                if USE_EMBEDDING and not random_embeddings:
                    feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                else:
                    idx = int(triple[1].replace(">", "").split("/")[-1])
                    idx_bin = bin(idx)[2:].zfill(100)
                    feature_vector = [float(i) for i in idx_bin]
                # Add the occurence of the predicate to the embedding
                if USE_OCCURRENCE and use_occurrence:
                    feature_vector.append(math.log(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))
                else:
                    feature_vector.append(1)
                # Add a dimension for the direction of the edge
                feature_vector.append(1)

                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([feature_vector])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                # Case if edge set does not exist yet
                if USE_EMBEDDING and not random_embeddings:
                    feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                else:
                    idx = int(triple[1].replace(">", "").split("/")[-1])
                    idx_bin = bin(idx)[2:].zfill(100)
                    feature_vector = [float(i) for i in idx_bin]
                if USE_OCCURRENCE and use_occurrence:
                    feature_vector.append(math.log(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))
                else:
                    feature_vector.append(1)
                feature_vector.append(1)

                data["entity", p, "entity"].edge_attr = torch.tensor([feature_vector])

        # Adding the embeddings of s and o
        if not s in node_mapping.keys():
            node_mapping[s] = n_entities
            n_entities += 1
            if "?" in s:
                emb = variable_embedding
                # indicate index of variable in query

                if shuffled:
                    emb[0] = n_entities_random #todo
                else:
                    emb[0] = node_mapping[s]  # todo
                n_entities_random +=1 #todo
                node_embeddings.append(emb)
            else:
                if random_embeddings:
                    # Use binary vector representation
                    idx = int(s.replace(">", "").split("/")[-1])
                    idx_bin = bin(idx)[2:].zfill(100)
                    feature_vector = [float(i) for i in idx_bin]
                    # Append occurrence information or placeholder
                    if use_occurrence:
                        feature_vector.append(math.log(statistics[s]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))
                    else:
                        feature_vector.append(1)
                elif unknown_entity == 'false':
                    feature_vector = statistics[s]["embedding"].copy()
                    if use_occurrence:
                        feature_vector.append(math.log(statistics[s]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))
                    else:
                        feature_vector.append(1)
                elif unknown_entity == 'true':
                    feature_vector = unknown_entity_embedding.copy()
                    feature_vector.append(1)
                elif unknown_entity == 'random':
                    # Generate a random float between 0 and 1

                    # 70% chance of being True
                    if rand_num < 0.7:
                        feature_vector = statistics[s]["embedding"].copy()
                        if use_occurrence:
                            feature_vector.append(math.log(statistics[s]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))
                        else:
                            feature_vector.append(1)
                    else:
                        feature_vector = unknown_entity_embedding.copy()
                        feature_vector.append(1)
                else:
                    idx = int(s.replace(">", "").split("/")[-1])
                    idx_bin = bin(idx)[2:].zfill(100)
                    feature_vector = [float(i) for i in idx_bin]

                node_embeddings.append(feature_vector)
        if not o in node_mapping.keys():
            node_mapping[o] = n_entities
            n_entities += 1
            if "?" in o:
                emb = variable_embedding
                if shuffled:
                    emb[0] = n_entities_random #todo
                else:
                    emb[0] = node_mapping[o]  # todo
                n_entities_random +=1 #todo
                node_embeddings.append(emb)
            else:
                if random_embeddings:
                    # Use binary vector representation
                    idx = int(o.replace(">", "").split("/")[-1])
                    idx_bin = bin(idx)[2:].zfill(100)
                    feature_vector = [float(i) for i in idx_bin]
                    # Append occurrence information or placeholder
                    if use_occurrence:
                        feature_vector.append(math.log(statistics[o]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))
                    else:
                        feature_vector.append(1)
                elif unknown_entity == 'false':
                    feature_vector = statistics[o]["embedding"].copy()
                    if use_occurrence:
                        feature_vector.append(math.log(statistics[o]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))
                    else:
                        feature_vector.append(1)
                elif unknown_entity == 'true':
                    feature_vector = unknown_entity_embedding.copy()
                    feature_vector.append(1)
                elif unknown_entity == 'random':
                    # Generate a random float between 0 and 1
                    rand_num = random.random()
                    # 70% chance of being True
                    if rand_num < 0.7:
                        feature_vector = statistics[o]["embedding"].copy()
                        if use_occurrence:
                            feature_vector.append(math.log(statistics[o]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))
                        else:
                            feature_vector.append(1)
                    else:
                        feature_vector = unknown_entity_embedding.copy()
                        feature_vector.append(1)
                else:
                    idx = int(o.replace(">", "").split("/")[-1])
                    idx_bin = bin(idx)[2:].zfill(100)
                    feature_vector = [float(i) for i in idx_bin]

                node_embeddings.append(feature_vector)


        # Finally, add the edge to the graph
        try:
            tp = torch.cat(
                (data['entity', p, 'entity'].edge_index, torch.tensor([[node_mapping[s]], [node_mapping[o]]])), dim=1)
            data["entity", p, "entity"].edge_index = tp

        except:
            tp = torch.tensor([[node_mapping[s]], [node_mapping[o]]])
            data["entity", p, "entity"].edge_index = tp

    data["entity"].x = torch.tensor(node_embeddings)

    if n_atoms is not None:
        n_atoms += len(atom_set)
        return data, n_atoms
    else:
        return data

def get_query_graph_data(query_graph, statistics, device):
    data = HeteroData()
    data = data.to(device)
    node_mapping = {}
    n_entities = 0
    node_embeddings = []
    variable_embedding = np.ones(101)
    feature_vector = None
    for triple in query_graph["triples"]:

        s = triple[0].replace("<", "").replace(">", "")
        o = triple[2].replace("<", "").replace(">", "")

        # If predicate is a variable
        if "?" in triple[1]:
            p = 237
            try:
                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([variable_embedding])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                data["entity", p, "entity"].edge_attr = torch.tensor([variable_embedding])

        else:
            try:
                p = int(triple[1].replace("<http://ex.org/", "").replace(">", ""))
            except:
                print(triple)
                raise
            try:
                feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                feature_vector.append(math.log(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))

                ea = torch.cat((data["entity", p, "entity"].edge_attr, torch.tensor([feature_vector])), dim=0)
                data["entity", p, "entity"].edge_attr = ea
            except:
                feature_vector = statistics[triple[1].replace("<", "").replace(">", "")]["embedding"].copy()
                feature_vector.append(math.log(statistics[triple[1].replace("<", "").replace(">", "")]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))
                data["entity", p, "entity"].edge_attr = torch.tensor([feature_vector])

        # Adding the embeddings of s and o to
        if not s in node_mapping.keys():
            node_mapping[s] = n_entities
            n_entities += 1
            if "?" in s:
                emb = variable_embedding
                emb[0] = node_mapping[s]
                node_embeddings.append(emb)
            else:
                feature_vector = statistics[s]["embedding"].copy()
                feature_vector.append(math.log(statistics[s]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))
                node_embeddings.append(feature_vector)
        if not o in node_mapping.keys():
            node_mapping[o] = n_entities
            n_entities += 1
            if "?" in o:
                emb = variable_embedding
                emb[0] = node_mapping[s]
                node_embeddings.append(variable_embedding)
            else:
                feature_vector = statistics[o]["embedding"].copy()
                feature_vector.append(math.log(statistics[o]["occurence"] + 1) / math.log(occurrence_norm_factor + 1))
                node_embeddings.append(feature_vector)
        try:
            tp = torch.cat(
                (data['entity', p, 'entity'].edge_index, torch.tensor([[node_mapping[s]], [node_mapping[o]]])), dim=1)
            data["entity", p, "entity"].edge_index = tp

        except:
            tp = torch.tensor([[node_mapping[s]], [node_mapping[o]]])
            data["entity", p, "entity"].edge_index = tp

    data["entity"].x = torch.tensor(node_embeddings)
    return data

