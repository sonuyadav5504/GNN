import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
import matplotlib.pyplot as plt

for data in loader:
    num_features = data.x.shape[1]  # Get the number of features from 'x'
    break

# Define your GNN model architecture for binary classification
class MyGNN(torch.nn.Module):
    def __init__(self):
        super(MyGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 8)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.conv2 = GCNConv(8, 16)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.dropout1 = torch.nn.Dropout(0.4)
        self.conv3 = GCNConv(16, 16)
        self.bn3 = torch.nn.BatchNorm1d(16)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.conv4 = GCNConv(16, 8)
        self.bn4 = torch.nn.BatchNorm1d(8)
        self.out_layer = torch.nn.Linear(8, 1)  # Output layer for binary classification

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
        x = global_max_pool(x, batch=None)
        x = self.out_layer(x).squeeze(1)  # Adjust the shape for binary classification
        return x

# Instantiate your model
model = MyGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

num_epochs = 20
train_losses = []
valid_losses = []
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index)
        target = data.y.view(-1, 1).float()  # Reshape target to match the output shape
        loss = criterion(out, target)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    train_losses.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for data in test_loader:
            out = model(data.x.float(), data.edge_index)
            target = data.y.view(-1, 1).float()  # Reshape target to match the output shape
            loss = criterion(out, target)
            
            val_loss += loss.item()
        val_loss /= len(test_loader)
        valid_losses.append(val_loss)
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Valid Loss: {val_loss:.4f}')

# Plotting the learning curves
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the learning curves
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
