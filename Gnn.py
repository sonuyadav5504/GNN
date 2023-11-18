import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import matplotlib.pyplot as plt

num_epochs = 25  # Define the number of epochs

# Define your GNN model architecture for regression
class MyGNN(torch.nn.Module):
    def __init__(self):
        super(MyGNN, self).__init__()
        # Define your message passing layers, e.g., using GCNConv, GATConv, etc.
        self.conv1 = GCNConv(9, 64)
        self.conv2 = GCNConv(64, 32)
        # Define your output layer for regression
        self.out_layer = torch.nn.Linear(32, 1)  # Output a single regression value

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # Global pooling to aggregate information from all nodes
        x = global_mean_pool(x, batch=None)
        # Apply the output layer for regression
        x = self.out_layer(x).squeeze(1)  # Squeeze to get rid of extra dimension
        return x

# Instantiate your model
model = MyGNN()
# Train your regression model
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = torch.nn.MSELoss()  # Mean Squared Error loss for regression

for epoch in range(num_epochs):
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index)
        loss = criterion(out, data.y.float())  # Ensure targets are float for regression
        loss.backward()
        optimizer.step()

    # Optionally validate or test your model after each epoch
    # ...

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Optionally, save your trained model
# torch.save(model.state_dict(), 'gnn_regression_model.pth')
