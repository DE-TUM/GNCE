import argparse
import random

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
import json
import rdflib
from tqdm import tqdm
import os
import sys
import yaml

def uri_to_id(uri):
    return uri.split('/')[-1]


uri_query = """
        PREFIX org: <https://w3id.org/scholarlydata/organisation/>

        SELECT *
        WHERE{{
        {{<{}> ?p ?o.}}
        UNION
        {{?s <{}> ?o.}}
        UNION
        {{?s ?p <{}>.}}

        }}
        LIMIT 30000000000
        """

literal_query = """
        PREFIX org: <https://w3id.org/scholarlydata/organisation/>

        SELECT *
        WHERE{{
        {{?s ?p '{}'.}}

        }}
        LIMIT 3000000000000
        """


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_embeddings(dataset_name, kg_file, entities=None, remote=True, sparql_endpoint="http://127.0.0.1:8902/sparql/", output_dir=None):
    """
    This function calculates simple occurences as well as rdf2vec embeddings for a given rdf graph
    Saves the embeddings and occurrences in one json file per entity
    :param dataset_name: The name of the kg, used to save the statistics
    :param kg_file: path to the .ttl file of the kg
    :param entities: list of entities for which to calculate embeddings
    :param remote: flag whether a remote sparql endpoint will be used or the .ttl file in memory
    :param sparql_endpoint: url of the remote sparql endpoint, if used or path to the rdf file
    :param output_dir: base directory for saving embeddings output
    :return: None
    """
    GRAPH = KG(sparql_endpoint, skip_verify=True)

    if remote:
        entities = entities
    else:
        if entities is not None:
            entities = entities
        else:
            train_entities = [entity.name for entity in list(GRAPH._entities)]
            test_entities = [entity.name for entity in list(GRAPH._vertices)]
            entities = set(train_entities + test_entities)

        #Create the RDF2vec model
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=10, vector_size=100),
        walkers=[RandomWalker(4, max_walks=5, with_reverse=True, n_jobs=12, md5_bytes=None)],
        verbose=2
    )


    # Count Occurences of nodes
    occurrences = {}


    print("Calculating Occurences")

    if True:  # occurrence_from_file

        with open(kg_file, "r") as file:
            for line in tqdm(file):
                line = line.strip().split(" ")  # Assuming the elements are separated by a space
                s = line[0].replace("<", "").replace(">", "")
                p = line[1].replace("<", "").replace(">", "")
                o = line[2].replace("<", "").replace(">", "")

                # Using dict.get() method to eliminate try-except blocks
                occurrences[s] = occurrences.get(s, 0) + 1
                occurrences[p] = occurrences.get(p, 0) + 1
                occurrences[o] = occurrences.get(o, 0) + 1

    else:
        raise NotImplementedError


    #Saving Occurences
    with open(dataset_name + "_ocurrences.json", "w") as fp:
        json.dump(occurrences, fp)

    # Filter entities to only those present in the KG
    valid_entities = {v.name for v in GRAPH._entities}
    original_count = len(entities)
    invalid_entities = [e for e in entities if e not in valid_entities]
    entities = [e for e in entities if e in valid_entities]
    print(f"Filtered {original_count} entities down to {len(entities)} existing in the KG.")

    # Generate the embeddings using fit_transform (more efficient than fit + transform)
    print("Starting to fit and transform model")
    embeddings_result = transformer.fit_transform(GRAPH, entities)
    print("Finished fitting model")

    # Access the underlying Word2Vec model to get predicate embeddings
    word2vec_model = transformer.embedder._model

    # Treat entities that are not in the KG as predicates (as discovered from the queries)
    predicates = invalid_entities
    print("Found ", len(predicates), " invalid entities / predicates")

    # Extract embeddings for predicates that actually exist in the vocabulary
    predicate_embeddings = {
        pred: word2vec_model.wv[pred]
        for pred in predicates
        if pred in word2vec_model.wv
    }
    print(f"Found embeddings for {len(predicate_embeddings)} predicates out of {len(predicates)} invalid entities")
    missing_predicates = [pred for pred in predicates if pred not in word2vec_model.wv]
    if missing_predicates:
        print("No embedding found for the following entities (invalid entities):")
        for missing in missing_predicates:
            print(f" - {missing}")

    # Build entity embeddings dictionary
    entity_embeddings = dict(zip(entities, embeddings_result[0]))

    # Combine entity and predicate embeddings into a single dictionary
    all_embeddings = {**entity_embeddings, **predicate_embeddings}



    # Save combined embeddings (entities + predicates) as JSON with occurrences
    print("Saving combined embeddings as JSON with occurrences")
    all_embeddings_with_occurrences = {}
    for entity, embedding in all_embeddings.items():
        all_embeddings_with_occurrences[entity] = {
            "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
            "occurence": occurrences.get(entity, 0)
        }
    
    # Use output_dir from config if provided
    if output_dir:
        embeddings_json_path = os.path.join(output_dir, dataset_name, f"{dataset_name}_embeddings.json")
    else:
        embeddings_json_path = f"{dataset_name}_embeddings.json"
    
    with open(embeddings_json_path, "w") as f:
        json.dump(all_embeddings_with_occurrences, f)
    print(f"Saved {len(all_embeddings_with_occurrences)} embeddings to {embeddings_json_path}")




    return


def get_entities_from_kg(kg_file):
    """Get ALL atoms from the KG file (subjects, predicates, objects)."""
    entities = set()
    
    with open(kg_file, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                s = parts[0].replace("<", "").replace(">", "")
                p = parts[1].replace("<", "").replace(">", "")
                o = parts[2].replace("<", "").replace(">", "")
                entities.add(s)
                entities.add(p)
                entities.add(o)
    
    return list(entities)


def get_entities_from_queries(query_files):
    """Extract entities from query JSON files."""
    entities = []
    
    for file_path in query_files:
        with open(file_path, 'r') as f:
            queries = json.load(f)
            for query in queries:
                entities += query.get('x', [])
                if 'triples' in query:
                    for triple in query['triples']:
                        subj = triple[0].replace("<", "").replace(">", "")
                        pred = triple[1].replace("<", "").replace(">", "")
                        obj = triple[2].replace("<", "").replace(">", "")
                        if not subj.startswith("?"):
                            entities.append(subj)
                        if not pred.startswith("?"):  # Skip variables like ?p1
                            entities.append(pred)
                        if not obj.startswith("?"):  # Skip variables
                            entities.append(obj)
    
    return list(set(entities))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate RDF2Vec embeddings')
    parser.add_argument('--config', type=str, default='configs/embedding_config.yml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    dataset_name = config['dataset_name']
    kg_file = config['kg_file']
    sparql_endpoint = config['sparql_endpoint']
    remote = config.get('remote', True)
    output_dir = config.get('output_dir', None)
    use_all_entities = config.get('use_all_entities', True)
    query_files = config.get('query_files', [])

    if use_all_entities:
        entities = get_entities_from_kg(kg_file)
        print(f'Using {len(entities)} entities from KG file for RDF2Vec')
    else:
        entities = get_entities_from_queries(query_files)
        print(f'Using {len(entities)} entities from query files for RDF2Vec')

    print('Starting...')
    get_embeddings(
        dataset_name=dataset_name,
        kg_file=kg_file,
        remote=remote,
        entities=entities,
        sparql_endpoint=sparql_endpoint,
        output_dir=output_dir
    )
