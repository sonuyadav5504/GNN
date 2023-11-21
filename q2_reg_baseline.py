import torch
import numpy as np
import gzip
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

test_folder = 'valid'

# Load test data from gzip files
def load_data(file_name):
    with gzip.open(os.path.join(test_folder, file_name), 'rb') as f:
        return torch.from_numpy(np.loadtxt(f, delimiter=',', dtype=np.float64))

node_features_rv = load_data('node_features.csv.gz')
edges_rv = load_data('edges.csv.gz').long()
edge_features_rv = load_data('edge_features.csv.gz')
num_nodes_rv = load_data('num_nodes.csv.gz').long()
num_edges_rv = load_data('num_edges.csv.gz').long()
graph_labels_rv = load_data('graph_labels.csv.gz').float()

# Handling NaN values in the graph_labels dataset
default_labelv = 0
graph_labels_rv = torch.nan_to_num(graph_labels_rv, nan=default_labelv)

# Random prediction model
def random_prediction(labels):
    min_label = labels.min().item()
    max_label = labels.max().item()
    num_samples = labels.size(0)
    random_preds = torch.rand(num_samples) * (max_label - min_label) + min_label
    return random_preds

random_preds = random_prediction(graph_labels_rv)

# Function to calculate RMSE
def calculate_rmse(predictions, labels):
    return torch.sqrt(torch.mean((predictions - labels) ** 2))

num_samplesv = len(graph_labels_rv)
val_labels = graph_labels_rv
val_preds = random_prediction(val_labels)
val_rmse = calculate_rmse(val_preds, val_labels)

print(f"Validation RMSE: {val_rmse}")

val_sizes = [int(i * 0.01 * num_samplesv) for i in range(1, 101)]

val_errors = []

for size in val_sizes:
    val_sample = val_labels[:size]
    val_pred_sample = val_preds[:size]
    val_rmse = calculate_rmse(val_pred_sample, val_sample)
    val_errors.append(val_rmse.item())

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(val_sizes, val_errors, label='Validation RMSE')
plt.title('Learning Curve')
plt.xlabel('Validation Set Size')
plt.ylabel('RMSE')
plt.legend()
plt.show()
