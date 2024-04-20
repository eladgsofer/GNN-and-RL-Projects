import torch
import torch.nn.functional as F
import torch_geometric.nn as torch_geo_nn
from torch_geometric.utils import to_dense_adj
from dataset import delieveries_dataset
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import tqdm
import matplotlib.pyplot as plt
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils import calculate_deleiveries, show_graph_with_labels, from_adjacency_tolist


# if __name__ == '__main__':
#     num_of_nodes = 20
#     data_size = 1000
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     D = delieveries_dataset(num_of_nodes=20, dataset_size=1000, edge_percentage=0.2)
#     D.generate_dataset(device=device)
#     loader = DataLoader(D.dataset, batch_size=32)
#
#
#     print("success")
#

class GCN(torch.nn.Module):
    def __init__(self, x_feature_num, batch_size):
        super().__init__()
        # Feature number is equal to node number |F| = |V|
        self.nodes_num = x_feature_num
        self.batch_size = batch_size
        self.conv1 = torch_geo_nn.GCNConv(x_feature_num, 16).double()
        self.conv2 = torch_geo_nn.GCNConv(16, 4).double()
        self.conv3 = torch_geo_nn.GCNConv(4, 1).double()

        self.out = torch_geo_nn.Linear(self.nodes_num*batch_size, self.batch_size).double()  # Changed output size to 1

    def forward(self, x, A, A_weights):
        x = self.conv1(x, A, edge_weight=A_weights)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, A, edge_weight=A_weights)
        x = F.relu(x)
        x = self.conv3(x, A, edge_weight=A_weights)
        x = F.relu(x)
        x = torch.flatten(x)
        x = self.out(x)
        return x  # Removed log_softmax


if __name__ == '__main__':
    num_of_nodes = 20
    data_size = 1000
    batch_size = 1
    epochs = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = delieveries_dataset(num_of_nodes=num_of_nodes, dataset_size=data_size, edge_percentage=0.2)
    D.generate_dataset(device=device)
    loader = DataLoader(D.dataset, batch_size=batch_size)

    # in our case, feature number is equal to the node number => |F| = |V|
    GNN_model = GCN(x_feature_num=num_of_nodes, batch_size=batch_size).to(device)
    optimizer = torch.optim.Adam(GNN_model.parameters(), lr=0.01, weight_decay=5e-4)

    GNN_model.train()
    loss_history = []
    pbar = tqdm.tqdm(total=data_size)
    for epoch in range(epochs):
        epoch_total_loss = 0
        for batch in loader:
            x, A, A_weights, y = batch.x, batch.edge_index, batch.weights, batch.y
            optimizer.zero_grad()
            y_hat = GNN_model(x, A, A_weights=A_weights)
            loss = F.mse_loss(y_hat, y)
            loss.backward()
            epoch_total_loss += loss.item()
            optimizer.step()

        epoch_loss = epoch_total_loss / epochs
        pbar.set_description(f'Epoch {epoch:02d}, Loss {epoch_loss}')
        loss_history.append(epoch_loss)

    plt.plot(loss_history)
    plt.title("GNN Loss graph")
    plt.yscale('log')
    plt.xlabel("#Epoch")
    plt.ylabel("Loss")
    plt.show()
    loader = DataLoader(D.dataset, batch_size=1)

    alpha = 0.2
    GNN_model.eval()
    opt_avg_cost = 0
    for sample in loader:
        x, A, A_weights, y = sample.x, sample.edge_index, sample.weights, sample.y

        # A_optimizer = torch.optim.Adam(A_weights, lr=0.05)
        for i in range(200):
            A_weights.requires_grad = True

            cost = GNN_model(x, A, A_weights)
            grad = torch.autograd.grad(cost, A_weights)[0]
            A_weights = A_weights + alpha*grad[0]
            A_weights = A_weights.detach()

            print(cost.item())
        opt_avg_cost += cost.item()

    print("####### {0} #######".format(opt_avg_cost))

