import os
import subprocess
import time
from datetime import datetime
import numpy as np
import json
from .transform_query import query_to_gcare
import random
from pathlib import Path
from tqdm import tqdm

def run_gcare(dataset, query_type, eval_folder, query_filename, inductive):
    methods = ['wj', 'cset', 'jsub', 'impr']
    #methods = ['impr']

    # Loading Queries and IDX transform mappings
    print(f'Loading Queries from: ')
    print(f"/home/tim/Datasets/{dataset}/{query_type}/{query_filename}")
    with open(f"/home/tim/Datasets/{dataset}/{query_type}/{query_filename}") as f:
        data = json.load(f)
    # Same split as other methods
    if not inductive == 'full':
        random.Random(4).shuffle(data)
        data = data[int(0.8 * len(data)):]
    else:
        with open(f"/home/tim/Datasets/{dataset}/{query_type}/{query_filename}") as f:
            data = json.load(f)
        pass

    with open(f"/home/tim/Datasets/{dataset}/id_to_id_mapping.json", "r") as f:
        id_to_id_mapping = json.load(f)
    with open(f"/home/tim/Datasets/{dataset}/id_to_id_mapping_predicate.json", "r") as f:
        id_to_id_mapping_predicate = json.load(f)

    for method in methods:
        print(f"---- Starting {method} ----")
        preds = []
        gts = []
        sizes = []
        result_data = []
        pred_times = []
        Path(f"/home/tim/Datasets/{dataset}/Results/{eval_folder}/{method}").mkdir(parents=True, exist_ok=True)


        # Predicting top n queries of the testset
        for query in tqdm(data[:]):
            query_to_gcare(query["triples"], 0, id_to_id_mapping=id_to_id_mapping,
                           id_to_id_mapping_predicate=id_to_id_mapping_predicate, dataset=dataset, card=query["y"])
            try:
                y_pred, pred_time = predict(method=method, dataset=dataset)
            except subprocess.CalledProcessError as e:
                print(e)
                continue
            preds.append(y_pred)
            gts.append(query["y"])
            pred_times.append(pred_time)
            sizes.append(len(query["triples"]))
            query["y_pred"] = y_pred
            query["exec_time_total"] = pred_time
            result_data.append(query)
        np.save(os.path.join(f"/home/tim/Datasets/{dataset}/Results/{eval_folder}/{method}/preds.npy"), preds)
        np.save(os.path.join(f"/home/tim/Datasets/{dataset}/Results/{eval_folder}/{method}/gts.npy"), gts)
        np.save(os.path.join(f"/home/tim/Datasets/{dataset}/Results/{eval_folder}/{method}/sizes.npy"), sizes)
        np.save(os.path.join(f"/home/tim/Datasets/{dataset}/Results/{eval_folder}/{method}/pred_times.npy"), pred_times)


        with open(f"/home/tim/Datasets/{dataset}/Results/{eval_folder}/{method}/results.json", 'w') as file:
            json.dump(result_data, file, indent=4)

def predict(method:str, dataset: str):
    pass
    os.chdir("/home/tim/gcare/scripts")
    seed = 0
    now = 'current'
    result_dir = f'/home/tim/gcare/data/result/accuracy/{now}'
    p = 0.03
    repeat = 1 if method in ['cset', 'bsk', 'sumrdf'] else 30
    if method == 'bsk':
        os.environ['GCARE_BSK_BUDGET'] = '4096'


    command = f'./run-exp.sh {method} {dataset} {p} {seed} {repeat} {result_dir}'
    result = subprocess.run(f'{command} | grep -A 2 "Average Time"',
                           check=True, shell=True, stdout=subprocess.PIPE)
    result = result.stdout.decode()

    # command = f'./run-exp.sh {method} {dataset} {p} {seed} {repeat} {result_dir}'
    # process = subprocess.Popen(f'{command} | grep -A 2 "Average Time"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    # # Wait for the subprocess to finish
    # stdout, stderr = process.communicate()
    # # Optionally, you can check the return code to ensure the process completed successfully
    # if process.returncode != 0:
    #     print(f'Subprocess failed with error: {stderr.decode()}')
    # # Ensure the process is cleaned up
    # process.terminate()  # This sends SIGTERM to the process, if it is still running
    # process.wait()  # This waits for the process to terminate
    # result = stdout.decode()



    lines = result.strip().split('\n')
    predicted_cardinalities = np.asarray(lines[2].split(' ')[1:], dtype=float)
    prediction_time = float(lines[0].split(" ")[4])
    pred_cardinality = np.mean(predicted_cardinalities)

    # Clear the environment variable if it was set
    if 'GCARE_BSK_BUDGET' in os.environ:
        del os.environ['GCARE_BSK_BUDGET']

    return pred_cardinality, prediction_time


if __name__ == "__main__":
    predict('wj', 'yago')

