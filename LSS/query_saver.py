import json
import random
from tqdm import tqdm
import os


def remove_files_in_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Loop through each file and remove it
    for file in files:
        file_path = os.path.join(folder_path, file)

        # Check if it's a file (not a folder)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            print(f"Skipping {file_path}, not a file")


def query_to_gcare(query, query_idx, id_to_id_mapping, id_to_id_mapping_predicate, dataset, card):
    # Delete existing query:


    vertices = set()
    vertex_labels = {}
    # Get unique vertices
    for tp in query:
        vertices.add(tp[0])
        vertices.add(tp[2])
        if dataset == 'yago' or 'yago_inductive':
            rdf_type_uri = '<http://example.com/13000179>'
        elif dataset == 'wikidata':
            rdf_type_uri = '<http://www.wikidata.org/prop/direct/P31>'
        elif dataset == 'swdf' or 'swdf_inductive':
            rdf_type_uri = '<http://ex.org/03>'
        elif dataset == 'lubm':
            rdf_type_uri = '<http://example.org/1>'
        else:
            raise AssertionError("rdf type uri missing !")

        if (tp[1] == rdf_type_uri) and ('?' not in tp[2]):

            if tp[0] in vertex_labels:
                vertex_labels[tp[0]] += [tp[2]]
            else:
                vertex_labels[tp[0]] = [tp[2]]

    # Creating Vertex Dict
    vertex_dict = {}
    vid = 0
    for vertex in vertices:
        # dvid = vertex.split("/")[-1].replace(">", "") if not "?" in vertex else -1
        try:
            dvid = id_to_id_mapping[vertex] if not "?" in vertex else -1
        except KeyError:
            dvid = 13000000 #yago
        if vertex in vertex_labels:
            # print('In vertex')
            try:
                labels = [id_to_id_mapping[v] for v in vertex_labels[vertex]]
            except KeyError:
                labels = [-1]
        else:
            labels = [-1]
        vertex_dict[vertex] = [vid] + labels + [dvid]
        vid += 1

    # Creating Edge List
    edge_list = []
    for tp in query:
        # edge_label = tp[1].split("/")[-1].replace(">", "") if not "?" in tp[1] else -1
        edge_label = id_to_id_mapping_predicate[tp[1]] if not "?" in tp[1] else -1

        edge_list.append([vertex_dict[tp[0]][0], vertex_dict[tp[2]][0], edge_label])

    # Writing the Query File
    with open(f"/home/tim/LSS/data/queryset_homo/{dataset}/paths_1/{dataset}_{str(query_idx)}.txt", "w") as f:
        f.write("t # s " + str(query_idx))
        f.write("\n")
        for v in vertex_dict:
            label_str = ''
            for l in vertex_dict[v][1:-1]:
                label_str += str(l)
                label_str += ' '
            f.write("v " + str(vertex_dict[v][0]) + " " + label_str + str(vertex_dict[v][2]))
            f.write("\n")
        for e in edge_list:
            f.write("e " + str(e[0]) + " " + str(e[1]) + " " + str(e[2]))
            f.write("\n")
    # Saving cardinality
    with open(f"/home/tim/LSS/data/true_homo/{dataset}/paths_1/{dataset}_{str(query_idx)}.txt", "w") as f:
        f.write(str(card))

def save_all_queries(dataset, query_type, query_filename):

    print('Removing existing queries..')
    remove_files_in_folder(f"/home/tim/LSS/data/queryset_homo/{dataset}/paths_1/")
    remove_files_in_folder(f"/home/tim/LSS/data/true_homo/{dataset}/paths_1/")


    print("Loading Data and Mappings..")

    if isinstance(query_filename, str):
        with open(f"/home/tim/Datasets/{dataset}/{query_type}/{query_filename}") as f:
            data = json.load(f)
    else:
        data = []
        for file in query_filename:
            with open(f"/home/tim/Datasets/{dataset}/{query_type}/{file}") as f:
                data += json.load(f)

    with open(f"/home/tim/Datasets/{dataset}/id_to_id_mapping.json", "r") as f:
        id_to_id_mapping = json.load(f)
    with open(f"/home/tim/Datasets/{dataset}/id_to_id_mapping_predicate.json", "r") as f:
        id_to_id_mapping_predicate = json.load(f)

    i = 0
    print("Generating Query Files..")
    for query in tqdm(data):
        query_to_gcare(query["triples"], i, id_to_id_mapping=id_to_id_mapping,
                       id_to_id_mapping_predicate=id_to_id_mapping_predicate, dataset=dataset, card=query["y"])
        i += 1


if __name__ == "__main__":
    dataset = 'wikidata'

    with open(f"/home/tim/Datasets/{dataset}/star/Joined_Queries.json") as f:
        data = json.load(f)

    with open(f"/home/tim/Datasets/{dataset}/id_to_id_mapping.json", "r") as f:
        id_to_id_mapping = json.load(f)
    with open(f"/home/tim/Datasets/{dataset}/id_to_id_mapping_predicate.json", "r") as f:
        id_to_id_mapping_predicate = json.load(f)

    query = data[0]
    print(query)
    query_to_gcare(query["triples"], 0, id_to_id_mapping=id_to_id_mapping,
                   id_to_id_mapping_predicate= id_to_id_mapping_predicate, dataset=dataset, card = query["y"])

