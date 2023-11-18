import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import matplotlib.pyplot as plt

train_losses = []
num_epochs = 100  # Define the number of epochs

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
criterion = torch.nn.MSELoss()  # Mean Squared Error loss for regression

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

min_loss = float('inf')  # Initialize with a large value
patience = 4  # Define the number of epochs to wait for loss improvement
counter = 0  # Counter to track epochs without improvement

for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_predictions = []
    epoch_targets = []

    for data in loader:
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index)
        loss = criterion(out, data.y.float())  # Ensure targets are float for regression
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Store predictions and targets for later visualization
        epoch_predictions.extend(out.detach().numpy())
        epoch_targets.extend(data.y.numpy())
        
    scheduler.step()
    epoch_loss /= len(data_list)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')
    
    # Stopping criterion: Check for loss improvement
    if epoch_loss < min_loss:
        min_loss = epoch_loss
        counter = 0  # Reset counter since there's an improvement
    else:
        counter += 1  # Increment counter if no improvement
    
    if counter >= patience:
        print(f'Early stopping at epoch {epoch+1} as no improvement observed for {patience} epochs.')
        break
        
# After training, add testing code
test_losses = []
model.eval()  # Set the model to evaluation mode

for data in test_loader:
    with torch.no_grad():
        out = model(data.x.float(), data.edge_index)
        test_loss = criterion(out, data.y.float())
        test_losses.append(test_loss.item())

# Calculate average test loss
avg_test_loss = sum(test_losses) / len(test_losses)
print(f'Average Test Loss: {avg_test_loss:.4f}')
        
