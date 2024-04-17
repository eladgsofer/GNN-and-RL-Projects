import os
import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

target_dataset = 'ogbn-arxiv'
# This will download the ogbn-arxiv to the 'networks' folder
dataset = PygNodePropPredDataset(name=target_dataset, root='networks')

data = dataset[0]

split_idx = dataset.get_idx_split()

train_idx = split_idx['train']
valid_idx = split_idx['valid']
test_idx = split_idx['test']

train_loader = NeighborLoader(data, input_nodes=train_idx,
                              shuffle=True, num_workers=1,
                              batch_size=1024, num_neighbors=[30] * 2)
total_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
                              batch_size=4096, shuffle=False,
                              num_workers=1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.layers = Sequential('x, edge_index', [
            (GCNConv(in_channels, hidden_channels), 'x, edge_index -> x'),
            ReLU(inplace=True),
            torch.nn.BatchNorm1d(hidden_channels),
            (GCNConv(hidden_channels, hidden_channels), 'x, edge_index -> x'),
            ReLU(inplace=True),
            torch.nn.BatchNorm1d(hidden_channels),
            Linear(hidden_channels, out_channels),
        ])

    def forward(self, x, edge_index):
        if len(self.layers) > 1:
            looper = self.layers[:-1]
        else:
            looper = self.layers

        for i, layer in enumerate(looper):
            x = layer(x, edge_index)
            try:
                x = self.layers_bn[i](x)
            except Exception as e:
                abs(1)
            finally:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        if len(self.layers) > 1:
            x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=-1), torch.var(x)

    def inference(self, total_loader, device):
        xs = []
        var_ = []
        for batch in total_loader:
            out, var = self.forward(batch.x.to(device), batch.edge_index.to(device))
            out = out[:batch.batch_size]
            xs.append(out.cpu())
            var_.append(var.item())

        out_all = torch.cat(xs, dim=0)

        return out_all, var_


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(data.x.shape[1], 256, dataset.num_classes, n_layers=2)
model.to(device)
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=7)


def test(model, device):
    evaluator = Evaluator(name=target_dataset)
    model.eval()
    out, var = model.inference(total_loader, device)
    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({'y_true': y_true[split_idx['train']], 'y_pred': y_pred[split_idx['train']], })['acc']
    val_acc = evaluator.eval({'y_true': y_true[split_idx['valid']], 'y_pred': y_pred[split_idx['valid']], })['acc']
    test_acc = evaluator.eval({'y_true': y_true[split_idx['test']], 'y_pred': y_pred[split_idx['test']], })['acc']
    return train_acc, val_acc, test_acc, torch.mean(torch.Tensor(var))


for epoch in range(1, epochs):
    model.train()
    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = total_correct = 0
    for batch in train_loader:
        batch_size = batch.batch_size
        optimizer.zero_grad()
        out, _ = model(batch.x.to(device), batch.edge_index.to(device))
        out = out[:batch_size]
        batch_y = batch.y[:batch_size].to(device)
        batch_y = torch.reshape(batch_y, (-1,))
        loss = F.nll_loss(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
        pbar.update(batch.batch_size)
    pbar.close()
    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)
    train_acc, val_acc, test_acc, var = test(model, device)

    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Var: {var:.4f}')
