import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.metrics import roc_auc_score
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
    mode_value = mode(graph_labels[~np.isnan(graph_labels)], axis=0)[0][0]  # Explicitly specify the axis
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

train_folder = 'train_c'
test_folder = 'valid_c'

node_features_c = load_data(train_folder, 'node_features.csv.gz')
edges_c = load_data(train_folder, 'edges.csv.gz').long()
edge_features_c = load_data(train_folder, 'edge_features.csv.gz')
num_nodes_c = load_data(train_folder, 'num_nodes.csv.gz')
num_edges_c = load_data(train_folder, 'num_edges.csv.gz')
graph_labels_c = load_data(train_folder, 'graph_labels.csv.gz')

loader_c = process_data(train_folder, node_features_c, edges_c, edge_features_c, num_nodes_c, num_edges_c, graph_labels_c)

node_features_cv = load_data(test_folder, 'node_features.csv.gz')
edges_cv = load_data(test_folder, 'edges.csv.gz').long()
edge_features_cv = load_data(test_folder, 'edge_features.csv.gz')
num_nodes_cv = load_data(test_folder, 'num_nodes.csv.gz')
num_edges_cv = load_data(test_folder, 'num_edges.csv.gz')
graph_labels_cv = load_data(test_folder, 'graph_labels.csv.gz')

test_loader_c = process_data(test_folder, node_features_cv, edges_cv, edge_features_cv, num_nodes_cv, num_edges_cv, graph_labels_cv)


class ROCAUC(torch.nn.Module):
    def __init__(self):
        super(ROCAUC, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred_sigmoid = torch.sigmoid(y_pred)
        return roc_auc_score(y_true.cpu().detach().numpy(), y_pred_sigmoid.cpu().detach().numpy())

# Define your GNN model architecture for binary classification using GIN
class MyGIN(torch.nn.Module):
    def __init__(self, num_features):
        super(MyGIN, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(num_features, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU()
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        ))
        self.conv3 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU()
        ))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.mlp(x).squeeze(1)
        return x

# Define your GNN model
for data in loader_c:
    num_features = data.x.shape[1]
    break

model = MyGIN(num_features)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()
criterion_roc_auc = ROCAUC()

# Training loop
train_losses = []
train_roc_auc_list = []
valid_losses = []
valid_roc_auc_list = []
best_loss = float('inf')
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    y_true_train = []
    y_pred_train = []
    
    for data in loader_c:
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index, data.batch)
        loss = criterion(out.squeeze(), data.y.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        y_pred_train.extend(torch.sigmoid(out).cpu().detach().numpy().flatten())
        y_true_train.extend(data.y.cpu().detach().numpy().flatten())

    train_roc_auc = roc_auc_score(y_true_train, y_pred_train)
    train_roc_auc_list.append(train_roc_auc)
    train_loss = epoch_loss / len(loader_c)
    train_losses.append(train_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train ROC-AUC: {train_roc_auc:.4f}')

    model.eval()

    # Validation loop
    y_true_valid = []
    y_pred_valid = []
    with torch.no_grad():
        for data in test_loader_c:
            out = model(data.x.float(), data.edge_index, data.batch)
            loss = criterion(out.squeeze(), data.y.float())
            valid_losses.append(loss.item())

            y_pred_valid.extend(torch.sigmoid(out).cpu().detach().numpy().flatten())
            y_true_valid.extend(data.y.cpu().detach().numpy().flatten())

        valid_roc_auc = roc_auc_score(y_true_valid, y_pred_valid)
        valid_roc_auc_list.append(valid_roc_auc)

    print(f'Epoch [{epoch+1}/{num_epochs}], Validation ROC-AUC: {valid_roc_auc:.4f}')

# Saving the trained model
torch.save(model.state_dict(), 'trained_model_classification.pth')

# Plot learning curves
plt.plot(train_roc_auc_list, label='Training ROC-AUC')
plt.plot(valid_roc_auc_list, label='Validation ROC-AUC')
plt.xlabel('Epochs')
plt.ylabel('ROC-AUC')
plt.legend()
plt.show()

plt.plot(valid_roc_auc_list, label='Validation ROC-AUC')
plt.xlabel('Epochs')
plt.ylabel('ROC-AUC')
plt.legend()
plt.show()
