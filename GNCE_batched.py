import json
import random
from models import GINmodel, TripleModelAdapt
import torch
from tqdm import tqdm
import numpy as np
from utils import get_query_graph_data, StatisticsLoader, get_query_graph_data_new, ToUndirectedCustom
import os
from torch_geometric.data import Data, DataLoader
from time import time
from torch_geometric.nn import GINEConv
import torch.nn.functional as F
import matplotlib.pyplot as plt


def q_error(pred, gt):
    gt_exp = torch.exp(gt)
    pred_exp = torch.exp(pred.squeeze())
    return torch.max(gt_exp / pred_exp, pred_exp / gt_exp)

if __name__ == "__main__":
    dataset = 'swdf'
    query_type = 'star'

    with open(f"/home/tim/Datasets/{dataset}/{query_type}/Joined_Queries.json") as f:
        data = json.load(f)

    random.Random(4).shuffle(data)
    train_data = data[:int(0.8 * len(data))][:]
    test_data = data[int(0.8 * len(data)):][:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'


    model = TripleModelAdapt().to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    criterion = torch.nn.MSELoss()
    batch_size = 1

    n_graphs= 1000

    statistics = StatisticsLoader(os.path.join("/home/tim/Datasets", dataset, "statistics"))

    # Creating datasets
    # Preparing train set
    graphs = []

    # How many atoms are in total in the queries:
    n_atoms = 0

    for datapoint in tqdm(train_data[:n_graphs]):
        # Get graph representation of query
        data, n_atoms = get_query_graph_data_new(datapoint, statistics, device, unknown_entity='false', n_atoms=n_atoms)


        # Transform graph to undirected representation, with feature indicating edge direction
        data = ToUndirectedCustom(merge=False)(data)
        data = data.to_homogeneous()
        y = np.log(datapoint["y"])
        #y = np.array(datapoint["y"]).astype(np.float64)

        y = torch.tensor(y)
        data.y = y
        graphs.append(data)

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    # Preparing test set
    test_graphs = []
    for datapoint in tqdm(test_data[:n_graphs]):
        data, n_atoms = get_query_graph_data_new(datapoint, statistics, device, unknown_entity='false', n_atoms=n_atoms)
        data = ToUndirectedCustom(merge=False)(data)
        data = data.to_homogeneous()
        y = np.log(datapoint["y"])
        #y = np.array(datapoint["y"]).astype(np.float64)

        y = torch.tensor(y)
        data.y = y
        test_graphs.append(data)

    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=True)

    train_loss = 0
    train_q_error = 0
    num_batches = 0

    start_time = time()
    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            batch_q_error = q_error(out, batch.y)
            train_q_error += torch.sum(batch_q_error).item()  # Sum q-errors in the batch
            num_batches += len(batch.y)
        #print(f'Epoch {epoch+1}, Loss: {train_loss.item()}')
        avg_train_loss = train_loss / num_batches
        avg_train_q_error = train_q_error / num_batches
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Avg Train Q-Error: {avg_train_q_error}')

        model.eval()
        test_loss = 0
        test_q_error = 0
        num_batches = 0
        preds = []
        gts = []
        with torch.no_grad():
            for test_batch in test_loader:
                test_batch = test_batch.to(device)
                out = model(test_batch)
                test_loss += criterion(out, test_batch.y).item()

                try:
                    batch_q_error = q_error(out, test_batch.y)
                except:
                    print('prapa')
                test_q_error += torch.sum(batch_q_error).item()  # Sum q-errors in the batch
                num_batches += len(test_batch.y)

                if batch_size ==1:
                    preds.append(out.squeeze().tolist())
                else:
                    preds += out.squeeze().tolist()

                gts += test_batch.y.tolist()

        plt.plot(np.exp(gts), np.exp(preds), "x")
        plt.xlim(0, 10000)
        plt.ylim(0, 10000)
        plt.show()

        avg_test_loss = test_loss / num_batches
        avg_test_q_error = test_q_error / num_batches
        print(f'Epoch {epoch+1}, Test Loss: {avg_test_loss}, Avg Test Q-Error: {avg_test_q_error}')
        print('\n')


    total_time = time() - start_time
    print('Total Training Time: ', total_time)
    print('Number of atoms: ' , n_atoms)
    print('Time per Atom in ms: ', total_time/n_atoms * 1000)
