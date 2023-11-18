import torch
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader

# Load data from CSV files
node_features = torch.from_numpy(np.loadtxt('node_features.csv', delimiter=',', dtype=np.float64))
edges = torch.from_numpy(np.loadtxt('edges.csv', delimiter=',', dtype=int))
edge_features = torch.from_numpy(np.loadtxt('edge_features.csv', delimiter=',', dtype=np.float64))
num_nodes = torch.from_numpy(np.loadtxt('num_nodes.csv', dtype=int))
num_edges = torch.from_numpy(np.loadtxt('num_edges.csv', dtype=int))
graph_labels = torch.from_numpy(np.loadtxt('graph_labels.csv', dtype=int, converters=float))
graph_labels = torch.from_numpy(np.loadtxt('graph_labels.csv').astype(np.int64))

# def preprocess_data(node_features, edge_features):
#     # Normalize node features
#     node_features = (node_features - node_features.mean(dim=0)) / node_features.std(dim=0)

#     # Normalize edge attributes
#     edge_features = (edge_features - edge_features.mean(dim=0)) / edge_features.std(dim=0)
    
#     return node_features, edge_features

# # Preprocess node and edge features
# node_features, edge_features = preprocess_data(node_features, edge_features)

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
loader = DataLoader(data_list, batch_size=32, shuffle=True)

# # Example usage of the DataLoader
# for data in loader:
#     print(data)
