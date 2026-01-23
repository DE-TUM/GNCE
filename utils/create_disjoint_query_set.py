import json
import random
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

dataset = 'swdf'
query_type = 'star'

with open(f"Datasets/{dataset}/{query_type}/Joined_Queries.json") as f:
    data = json.load(f)

# This hyperparameter denotes the probability of applying a query to the trainset. It very
# sensibly influences the final splits, need to adjustit !
# Default Value: 0.002 (swdf), 0.0005 (yago star)
training_st_prob = 0.002

# Testing with respect to invariance of triple permutation
# Attention ! THis implementation only checks for overlapping Stars !
train_data = []
test_data = []

train_entities = set()
test_entities = set()

random.shuffle(data)

for query in tqdm(data):
    if len(train_data)==0:
        train_data.append(query)
        train_entities.update([a[2] for a in query["triples"] if not "?" in a[2]])
        continue
    if len(train_data) < training_st_prob * (len(data)):
        train_data.append(query)
        train_entities.update([a[2] for a in query["triples"] if not "?" in a[2]]) 
        continue

    # Getting all objects of query:
    entity_list = [a[2] for a in query["triples"] if not "?" in a[2]] 
    if any(entity in train_entities for entity in entity_list):
        if all(entity not in test_entities for entity in entity_list):
            train_entities.update(entity_list)
            train_data.append(query)
    else:
        test_data.append(query)
        test_entities.update(entity_list)


for e in tqdm(test_entities):
    if e in train_entities:
        print('Error: Overlapping Entities !')

print('Summary of disjoint split: ')
print(f'Number of queries in training set: {len(train_data)}')
print(f'Number of queries in test set: {len(test_data)}')


ls = []
for t in train_data:
    ls.append(len(t["triples"]))
ls_test = []
for t in test_data:
    ls_test.append(len(t["triples"]))

plt.hist(ls)

plt.hist(ls_test)

with open(f"Datasets/{dataset}/{query_type}/disjoint_train.json", "w") as f:
    json.dump(train_data, f)
with open(f"Datasets/{dataset}/{query_type}/disjoint_test.json", "w") as f:
    json.dump(test_data, f)