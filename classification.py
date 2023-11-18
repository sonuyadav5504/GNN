import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# Load data from CSV files
node_features = torch.from_numpy(np.loadtxt('node_features.csv', delimiter=',', dtype=np.float64))
edges = torch.from_numpy(np.loadtxt('edges.csv', delimiter=',', dtype=int))
edge_features = torch.from_numpy(np.loadtxt('edge_features.csv', delimiter=',', dtype=np.float64))
graph_labels = torch.from_numpy(np.loadtxt('graph_labels.csv', dtype=int))

data_list = []
for idx in range(len(graph_labels)):
    start_node = sum(num_nodes[:idx])
    end_node = start_node + num_nodes[idx]

    start_edge = sum(num_edges[:idx])
    end_edge = start_edge + num_edges[idx]

    data = Data(
        x=node_features[start_node:end_node],
        edge_index=edges[start_edge:end_edge].t().contiguous(),
        edge_attr=edge_features[start_edge:end_edge],
        y=graph_labels[idx]
    )
    data_list.append(data)

# DataLoader for classification
loader = DataLoader(data_list, batch_size=32, shuffle=True)

# Define your GNN model architecture for classification
class MyGNNClassification(torch.nn.Module):
    def __init__(self):
        super(MyGNNClassification, self).__init__()
        self.conv1 = GCNConv(9, 64)
        self.conv2 = GCNConv(64, 32)
        self.out_layer = torch.nn.Linear(32, 1)  # Output for binary classification

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch=None)
        x = torch.sigmoid(self.out_layer(x)).squeeze(1)  # Sigmoid for binary classification
        return x

# Instantiate the model
model = MyGNNClassification()

# Train your classification model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()  # Binary Cross Entropy loss for classification

# Example usage of the DataLoader
for data in loader:
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output, data.y.float())  # Convert data.y to float for BCELoss
    loss.backward()
    optimizer.step()
