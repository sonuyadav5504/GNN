import torch
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader

# Load train data from CSV files
node_features = torch.from_numpy(np.loadtxt('node_features.csv', delimiter=',', dtype=np.float64))
edges = torch.from_numpy(np.loadtxt('edges.csv', delimiter=',', dtype=int))
edge_features = torch.from_numpy(np.loadtxt('edge_features.csv', delimiter=',', dtype=np.float64))
num_nodes = torch.from_numpy(np.loadtxt('num_nodes.csv', dtype=int))
num_edges = torch.from_numpy(np.loadtxt('num_edges.csv', dtype=int))
graph_labels = torch.from_numpy(np.loadtxt('graph_labels.csv', dtype=int, converters=float))
graph_labels = torch.from_numpy(np.loadtxt('graph_labels.csv').astype(np.int64))

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

# Load test data from CSV files
node_features1 = torch.from_numpy(np.loadtxt('node_features1.csv', delimiter=',', dtype=np.float64))
edges1 = torch.from_numpy(np.loadtxt('edges1.csv', delimiter=',', dtype=int))
edge_features1 = torch.from_numpy(np.loadtxt('edge_features1.csv', delimiter=',', dtype=np.float64))
num_nodes1 = torch.from_numpy(np.loadtxt('num_nodes1.csv', dtype=int))
num_edges1 = torch.from_numpy(np.loadtxt('num_edges1.csv', dtype=int))
graph_labels1 = torch.from_numpy(np.loadtxt('graph_labels1.csv', dtype=int, converters=float))
graph_labels1 = torch.from_numpy(np.loadtxt('graph_labels1.csv').astype(np.int64))

data_list1 = []
for idx in range(len(graph_labels1)):
    start_node = sum(num_nodes1[:idx])
    end_node = start_node + num_nodes1[idx]

    start_edge = sum(num_edges1[:idx])
    end_edge = start_edge + num_edges1[idx]

    data = Data(
        x=node_features1[start_node:end_node],
        edge_index=edges1[start_edge:end_edge].t().contiguous(),
        edge_attr=edge_features1[start_edge:end_edge],
        y=graph_labels1[idx]
    )
    data_list1.append(data)  # Use data_list1 here instead of data_list
test_loader = DataLoader(data_list1, batch_size=32, shuffle=True)
