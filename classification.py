import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
from sklearn.metrics import roc_auc_score

# Define your GNN model architecture for binary classification
class MyGNN(torch.nn.Module):
    def __init__(self):
        super(MyGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.conv2 = GCNConv(16, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.dropout = torch.nn.Dropout(0.5)
        self.conv3 = GCNConv(64, 64)  # Adjust the output size to match the target size
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.out_layer = torch.nn.Linear(64, 32)  # Output layer for binary classification

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)
        x = global_max_pool(x, batch=None)
        x = self.dropout(x)
        x = self.out_layer(x)  # Adjust the shape for binary classification
        return x

# Initialize your GNN model
model = MyGNN()  

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy loss for binary classification


# Training loop
train_losses = []
valid_losses = []
best_loss = float('inf')
num_epochs = 100  # Set your desired number of epochs
total_batches = len(loader_c)
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for idx, data in enumerate(loader_c):
        if idx == total_batches - 1:  # Skip the last batch
            break
            
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index)
        loss = criterion(out.squeeze(), data.y.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= (total_batches - 1)  # Adjust for skipping the last batch
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')

# Validation loop
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for idx, data in enumerate(loader_c):
        if idx == total_batches - 1:  # Skip the last batch
            break
             # Assuming loader_valid contains validation data
        out = model(data.x.float(), data.edge_index)
        pred = torch.sigmoid(out).cpu().numpy().flatten()  # Sigmoid for binary predictions
        y_pred.extend(pred)
        y_true.extend(data.y.cpu().numpy().flatten())

# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_true, y_pred)
print(f"Validation ROC-AUC: {roc_auc}")
