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

# Assuming graph_labels are binary (0 or 1)
# graph_labels_c = graph_labels_c.float()  # Convert labels to float for BCEWithLogitsLoss

# Training loop
train_losses = []
valid_losses = []
best_loss = float('inf')
num_epochs = 30  # Set your desired number of epochs
l=len(loader_c)
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for data in loader_c:
        if i>len(loader_c):
            optimizer.zero_grad()
            out = model(data.x.float(), data.edge_index)
            print("out ",out.squeeze().size())
            print("data ",data.y.size())
            loss = criterion(out.squeeze(), data.y.float())  # Use data.y for graph labels
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    epoch_loss /= len(loader_c)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')
# # Evaluation
# model.eval()
# with torch.no_grad():
#     predictions = model(data.x.float(), data.edge_index)
#     roc_auc = roc_auc_score(graph_labels.numpy(), torch.sigmoid(predictions.squeeze()).numpy())
#     print(f'ROC-AUC: {roc_auc:.4f}')
