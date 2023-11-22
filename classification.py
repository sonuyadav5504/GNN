import torch
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.metrics import roc_auc_score

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
num_epochs = 100

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

    train_roc_auc = criterion_roc_auc(torch.tensor(y_pred_train), torch.tensor(y_true_train))
    train_roc_auc_list.append(train_roc_auc)

    model.eval()
    valid_loss = 0.0
    y_true_valid = []
    y_pred_valid = []
    with torch.no_grad():
        for data in test_loader_c:
            out = model(data.x.float(), data.edge_index, data.batch)
            loss = criterion(out.squeeze(), data.y.float())
            valid_loss += loss.item()
            
            y_pred_valid.extend(torch.sigmoid(out).cpu().detach().numpy().flatten())
            y_true_valid.extend(data.y.cpu().detach().numpy().flatten())
            
    valid_roc_auc = criterion_roc_auc(torch.tensor(y_pred_valid), torch.tensor(y_true_valid))
    valid_roc_auc_list.append(valid_roc_auc)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train ROC-AUC: {train_roc_auc:.4f}, Validation ROC-AUC: {valid_roc_auc:.4f}')

    model.train()

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
