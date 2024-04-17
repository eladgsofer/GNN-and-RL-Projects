import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataset import delieveries_dataset

class GCN(torch.nn.Module):
    def __init__(self, x_feature_num):
        super().__init__()
        self.conv1 = GCNConv(x_feature_num, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)










if __name__ == '__main__':
    num_of_nodes = 20
    data_size = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = delieveries_dataset(num_of_nodes=20, dataset_size=1000, edge_percentage=0.2)
    D.generate_dataset(device=device)
    loader = DataLoader(D.dataset, batch_size=32)

    model = GCN(num_of_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()

    pbar = tqdm(total=data_size)
    for epoch in range(200):
        avg_loss = 0
        pbar.set_description(f'Epoch {epoch:02d}')

        for batch in loader:
            x, A, y = batch
            optimizer.zero_grad()
            y_hat = model(x, A)
            loss = F.l1_loss(y_hat, y)
            loss.backward()
            avg_loss+=loss.item()
            optimizer.step()

        print("loss: {0}".format(avg_loss/data_size))

    print("success")
