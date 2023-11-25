import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import os
import gzip
import numpy as np
from scipy.stats import mode

# Load data from gzip files
def load_data(folder, file_name):
    with gzip.open(os.path.join(folder, file_name), 'rb') as f:
        return torch.from_numpy(np.loadtxt(f, delimiter=',', dtype=np.float64))

def process_data(folder, node_features, edges, edge_features, num_nodes, num_edges, graph_labels):
    default_label = 0
    graph_labels = np.nan_to_num(graph_labels, nan=default_label)
    mode_value = mode(graph_labels[~np.isnan(graph_labels)], axis=0)[0][0]
    graph_labels[np.isnan(graph_labels)] = mode_value

    data_list = []
    for idx in range(len(graph_labels)):
        start_node = int(sum(num_nodes[:idx]))
        end_node = int(start_node + num_nodes[idx])

        start_edge = int(sum(num_edges[:idx]))
        end_edge = int(start_edge + num_edges[idx])

        data = Data(
            x=node_features[start_node:end_node],
            edge_index=edges[start_edge:end_edge].t().contiguous(),
            edge_attr=edge_features[start_edge:end_edge],
            y=graph_labels[idx]
        )
        data_list.append(data)

    return DataLoader(data_list, batch_size=32, shuffle=False)

test_folder = 'valid_r'

node_features_cv = load_data(test_folder, 'node_features.csv.gz')
edges_cv = load_data(test_folder, 'edges.csv.gz').long()
edge_features_cv = load_data(test_folder, 'edge_features.csv.gz')
num_nodes_cv = load_data(test_folder, 'num_nodes.csv.gz')
num_edges_cv = load_data(test_folder, 'num_edges.csv.gz')
graph_labels_cv = load_data(test_folder, 'graph_labels.csv.gz')

test_loader_r = process_data(test_folder, node_features_cv, edges_cv, edge_features_cv, num_nodes_cv, num_edges_cv, graph_labels_cv)

for data in test_loader_c:
    num_features = data.x.shape[1]
    break

# Define your tocsv function
def tocsv(y_arr, *, task):
    r"""Writes the numpy array to a csv file.
    params:
        y_arr: np.ndarray. A vector of all the predictions. Classes for
        classification and the regression value predicted for regression.

        task: str. Must be either of "classification" or "regression".
        Must be a keyword argument.
    Outputs a file named "y_classification.csv" or "y_regression.csv" in
    the directory it is called from. Must only be run once. In case outputs
    are generated from batches, only call this output on all the predictions
    from all the batches collected in a single numpy array. This means it'll
    only be called once.

    This code ensures this by checking if the file already exists, and does
    not overwrite the csv files. It just raises an error.
    """
    assert task in ["classification", "regression"], f"task must be either 'classification' or 'regression'. Found: {task}"
    assert isinstance(y_arr, np.ndarray), f"y_arr must be a numpy array, found: {type(y_arr)}"
    assert len(y_arr.squeeze().shape) == 1, f"y_arr must be a vector. Shape found: {y_arr.shape}"

    file_name = f"y_{task}.csv"

    if os.path.isfile(file_name):
        raise FileExistsError(f"File already exists: {file_name}. Ensure you're not calling this function multiple times.")

    df = pd.DataFrame(y_arr)
    df.to_csv(file_name, index=False, header=False)

    print(f"Predictions saved to {file_name}")

# Load the saved model
loaded_model = MyGIN(num_features)  # Assuming MyGIN is defined in your code
loaded_model.load_state_dict(torch.load('trained_model_regression.pth'))
loaded_model.eval()

# Predictions using the loaded model
predicted_values = []
ground_truth = []

with torch.no_grad():
    for data in test_loader_r:
        edge_index = data.edge_index.to(torch.int64)
        out = loaded_model(data.x.float(), edge_index, data.batch)
        predicted_values.extend(out.numpy())
        ground_truth.extend(data.y.numpy())

# Calculate RMSE
predicted_values = np.array(predicted_values)
ground_truth = np.array(ground_truth)
rmse = np.sqrt(np.mean(np.square(predicted_values - ground_truth)))

# Save predictions for validation set to a CSV file
tocsv(predicted_values, task="regression")
print(f'Validation RMSE Loss using loaded model: {rmse:.4f}')
