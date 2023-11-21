import torch
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import roc_auc_score

loader_valid=loader_c

class ROCAUC(torch.nn.Module):
    def __init__(self):
        super(ROCAUC, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred_sigmoid = torch.sigmoid(y_pred)
        return roc_auc_score(y_true.cpu().detach().numpy(), y_pred_sigmoid.cpu().detach().numpy())

# Define your GNN model architecture for binary classification
class MyGNN(torch.nn.Module):
    def __init__(self, num_features):
        super(MyGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 8)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.conv2 = GCNConv(8, 16)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.dropout1 = torch.nn.Dropout(0.4)
        self.conv3 = GCNConv(16, 32)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.conv4 = GCNConv(32, 64)
        self.bn4 = torch.nn.BatchNorm1d(64)
        self.out_layer = torch.nn.Linear(64, 32)  # Output layer for binary classification

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = torch.relu(x)
        x = global_mean_pool(x, batch=None)
        x = self.out_layer(x).squeeze(1)  # Adjust the shape for binary classification
        return x

# Initialize your GNN model
num_features =  9  # Replace with the appropriate number of features
model = MyGNN(num_features)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy loss for binary classification
criterion_roc_auc = ROCAUC()  # Custom ROC-AUC metric

# Training loop
train_losses = []
train_roc_auc_list = []
valid_losses = []
valid_roc_auc_list = []
best_loss = float('inf')
num_epochs = 5  # Set your desired number of epochs

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    y_true_train = []
    y_pred_train = []
    
    for idx, data in enumerate(loader_c):
        if idx == total_batches - 1:  # Skip the last batch
            break
            
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index)
        loss = criterion(out.squeeze(), data.y.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        y_pred_train.extend(torch.sigmoid(out).cpu().detach().numpy().flatten())
        y_true_train.extend(data.y.cpu().detach().numpy().flatten())

    train_roc_auc = criterion_roc_auc(torch.tensor(y_pred_train), torch.tensor(y_true_train))
    train_roc_auc_list.append(train_roc_auc)

    # Validation loop
    model.eval()
    valid_loss = 0.0
    y_true_valid = []
    y_pred_valid = []
    with torch.no_grad():
        for idx, data in enumerate(loader_valid):
            if idx == total_batches - 1:  # Skip the last batch
                break
            out = model(data.x.float(), data.edge_index)
            loss = criterion(out.squeeze(), data.y.float())
            valid_loss += loss.item()
            
            y_pred_valid.extend(torch.sigmoid(out).cpu().detach().numpy().flatten())
            y_true_valid.extend(data.y.cpu().detach().numpy().flatten())
            
    valid_roc_auc = criterion_roc_auc(torch.tensor(y_pred_valid), torch.tensor(y_true_valid))
    valid_roc_auc_list.append(valid_roc_auc)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train ROC-AUC: {train_roc_auc:.4f},Validation ROC-AUC: {valid_roc_auc:.4f}')

    model.train()  # Set the model back to train mode
    
plt.plot(train_roc_auc_list, label='Training ROC-AUC')
plt.plot(valid_roc_auc_list, label='Validation ROC-AUC')
plt.xlabel('Epochs')
plt.ylabel('ROC-AUC')
plt.legend()
plt.show()
