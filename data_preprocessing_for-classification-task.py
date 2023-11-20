import os
import gzip
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

train_folder = 'train'

# Load data from gzip files
def load_data(file_name):
    with gzip.open(os.path.join(train_folder, file_name), 'rb') as f:
        return torch.from_numpy(np.loadtxt(f, delimiter=',', dtype=np.float64))

node_features_c = load_data('node_features.csv.gz')
edges_c = load_data('edges.csv.gz').long()  # Ensure edges are of type long/int
edge_features_c = load_data('edge_features.csv.gz')
num_nodes_c = load_data('num_nodes.csv.gz')  # Ensure num_nodes is of type long/int
num_edges_c = load_data('num_edges.csv.gz') # Ensure num_edges is of type long/int
graph_labels_c = load_data('graph_labels.csv.gz')  # Ensure graph_labels is of type long/int

# handling nan values present in the graph_labels dataset
default_label = 0
graph_labels_c = np.nan_to_num(graph_labels_c, nan=default_label)

data_list_c = []
for idx in range(len(graph_labels_c)):
    start_node = int(sum(num_nodes_c[:idx]))
    end_node = int(start_node + num_nodes_c[idx])

    start_edge = int(sum(num_edges_c[:idx]))
    end_edge = int(start_edge + num_edges_c[idx])

    data = Data(
        x=node_features_c[start_node:end_node],
        edge_index=edges_c[start_edge:end_edge].t().contiguous(),
        edge_attr=edge_features_c[start_edge:end_edge],
        y=graph_labels_c[idx]
    )
    data_list_c.append(data)

loader_c = DataLoader(data_list_c, batch_size=32, shuffle=True)
