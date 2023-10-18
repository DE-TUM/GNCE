import json
import random
from tqdm import tqdm
import copy


def query_remove_duplicates(data):
    # Testing with respect to invariance of triple permutation and ignoring distinction between variables

    print("Overall Length of data: " , len(data))

    new_data = []

    query_hashs = []

    same_queries = 0
    same_queries_check = 0
    n_queries = 0
    i = 0
    for query in tqdm(data):
        n_queries += 1
        i += 1
        if new_data == []:
            new_data.append(query)
            query_hash = copy.deepcopy(query["triples"])
            for l in query_hash:
                for i in range(len(l)):
                    l[i] = l[i][:-1] if len(l[i]) == 3 else l[i]
            query_hashs.append(hash(frozenset([tuple(l) for l in query_hash])))

        hash_ = copy.deepcopy(query["triples"])
        for l in hash_:
            for i in range(len(l)):
                l[i] = l[i][:-1] if len(l[i]) == 3 else l[i]
        hash_ = hash(frozenset([tuple(l) for l in hash_]))
        if hash_ in query_hashs:
            same_queries += 1
        else:
            query_hashs.append(hash_)
            new_data.append(query)

    print("Found and removed ", same_queries, " duplicates !")
    print("Resulting Length of data: ", len(new_data))


    return new_data


if __name__ == "__main__":
    #Load Data
    data = []
    with open("/home/tim/cardinality_estimator/Datasets/swdf/star/swdf_test_data2.json") as f:
        test_data = json.load(f)
        data += test_data
    with open("/home/tim/cardinality_estimator/Datasets/swdf/star/swdf_test_data3.json") as f:
        test_data = json.load(f)
        data += test_data
    with open("/home/tim/cardinality_estimator/Datasets/swdf/star/swdf_test_data5.json") as f:
        test_data = json.load(f)
        data += test_data
    with open("/home/tim/cardinality_estimator/Datasets/swdf/star/swdf_test_data8.json") as f:
        test_data = json.load(f)
        data += test_data
    new_data = query_remove_duplicates(data)

    #Saving Cleaned Data to file
    with open("/home/tim/cardinality_estimator/Datasets/swdf/star/Joined_Queries.json", "w") as f:
        json.dump(new_data, f)

