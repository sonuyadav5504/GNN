import torch
import numpy as np
import gzip
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

test_folder = 'valid2'

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

# Random prediction model for binary classification
def random_prediction(labels):
    num_samples = labels.size(0)
    random_preds = torch.rand(num_samples)
    return random_preds

random_preds = random_prediction(graph_labels_rv)

# Function to calculate ROC-AUC for binary classification
def calculate_roc_auc(predictions, labels):
    return roc_auc_score(labels.numpy(), predictions.numpy())

num_samplesv = len(graph_labels_rv)
val_labels = graph_labels_rv
val_preds = random_prediction(val_labels)
val_roc_auc = calculate_roc_auc(val_preds, val_labels)

print(f"Validation ROC-AUC: {val_roc_auc}")

val_sizes = [int(i * 0.01 * num_samplesv) for i in range(1, 101)]

val_roc_aucs = []

for size in val_sizes:
    val_sample = val_labels[:size]
    val_pred_sample = val_preds[:size]
    val_roc_auc = calculate_roc_auc(val_pred_sample, val_sample)
    val_roc_aucs.append(val_roc_auc)

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(val_sizes, val_roc_aucs, label='Validation ROC-AUC')
plt.title('Learning Curve (ROC-AUC)')
plt.xlabel('Validation Set Size')
plt.ylabel('ROC-AUC')
plt.legend()
plt.show()
