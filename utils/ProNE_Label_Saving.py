import json
import numpy as np
from tqdm import tqdm
import random
import os
import shutil



#First, investigate the Labels of the Dataset

dataset = "yago"

# Paths for loading the data and saving the results
loading_path = "/work/schwatkm"
embedding_path = "/work/schwatkm/"
embedding_save_path = "/work/schwatkm/Embeddings/"
mapping_save_path = "/work/schwatkm/Embeddings/"

file = open(loading_path + "/gcare/data/dataset/" + dataset + "/" + dataset + ".txt", "r")
lines = file.readlines()
file.close()

# Loading mapping of entity and predicate ids to separate ids:
with open(loading_path + "/cardinality_estimator/Datasets/" + dataset + "/id_to_id_mapping.json", "r") as f:
    id_to_id_mapping = json.load(f)

with open(loading_path + "/cardinality_estimator/Datasets/" + dataset + "/id_to_id_mapping_predicate.json", "r") as f:
    id_to_id_mapping_predicate = json.load(f)

id_to_id_mapping_inv = {v: k for k, v in id_to_id_mapping.items()}


# Loading the labels of the dataset:
i = 0
labels_data = []
for line in lines:
    i += 1
    line = line.split()
    if "v" in line:
        labels_data += line[2:]
    if "e" in line:
        break
labels_data = set(labels_data)
print(labels_data)
labels_data_raw = {l: id_to_id_mapping_inv[int(l)] for l in list(labels_data)[1:] }
print(labels_data_raw)


# Loading the prone embeddings:
file = open(embedding_path + "spectral.emb")
lines = file.readlines()
file.close()

# Cleaning the embeddings:
for i in tqdm(range(1, len(lines))):2
    lines[i] = lines[i].split("\n")[0].split(" ")
    lines[i] = [float(el) for el in lines[i]]

lines = np.asarray(lines[1:])

labels_data = [int(el) for el in labels_data]
lines = lines[np.isin(lines[:, 0], labels_data)]

# Save the embeddings for the labels:
np.save(embedding_save_path + "/" + dataset + ".emb.npy", lines[:, 1:])

# Save the mapping of the labels to the ids:
with open(mapping_save_path + "/" + dataset + "_mapping.txt", "w") as f:
    for idd in lines[:, 0]:
        f.write(str(int(idd)))
        f.write("\n")


