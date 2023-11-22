import os
import gzip
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

train_folder = 'train2'

# Load data from gzip files
def load_data(file_name):
    with gzip.open(os.path.join(train_folder, file_name), 'rb') as f:
        return torch.from_numpy(np.loadtxt(f, delimiter=',', dtype=np.float64))

node_features_c = load_data('node_features.csv.gz')
edges_c = load_data('edges.csv.gz').long()  
edge_features_c = load_data('edge_features.csv.gz')
num_nodes_c = load_data('num_nodes.csv.gz')  
num_edges_c = load_data('num_edges.csv.gz')
graph_labels_c = load_data('graph_labels.csv.gz')  

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


test_folder = 'valid2'
# Load test data from gzip files
def load_data(file_name):
    with gzip.open(os.path.join(train_folder, file_name), 'rb') as f:
        return torch.from_numpy(np.loadtxt(f, delimiter=',', dtype=np.float64))

node_features_cv = load_data('node_features.csv.gz')
edges_cv = load_data('edges.csv.gz').long()  
edge_features_cv = load_data('edge_features.csv.gz')
num_nodes_cv = load_data('num_nodes.csv.gz')  
num_edges_cv = load_data('num_edges.csv.gz') 
graph_labels_cv = load_data('graph_labels.csv.gz')  

# handling nan values present in the graph_labels dataset
default_labelv = 0
graph_labels_cv = np.nan_to_num(graph_labels_cv, nan=default_labelv)

data_list_cv = []
for idx in range(len(graph_labels_cv)):
    start_node = int(sum(num_nodes_cv[:idx]))
    end_node = int(start_node + num_nodes_cv[idx])

    start_edge = int(sum(num_edges_cv[:idx]))
    end_edge = int(start_edge + num_edges_cv[idx])

    data = Data(
        x=node_features_cv[start_node:end_node],
        edge_index=edges_cv[start_edge:end_edge].t().contiguous(),
        edge_attr=edge_features_cv[start_edge:end_edge],
        y=graph_labels_c[idx]
    )
    data_list_cv.append(data)

test_loader_c = DataLoader(data_list_cv, batch_size=32, shuffle=True)
