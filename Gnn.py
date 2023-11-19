import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt

# Define your GNN model architecture for regression
class MyGNN(torch.nn.Module):
    def __init__(self):
        super(MyGNN, self).__init__()
        self.conv1 = GCNConv(9, 64)
        self.conv2 = GCNConv(64, 32)
        self.out_layer = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch=None)
        x = self.out_layer(x).squeeze(1)
        return x

# Instantiate your model
model = MyGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
criterion = torch.nn.MSELoss()

num_epochs = 30
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index)
        loss = criterion(out, data.y.float())
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
            loss = criterion(out, data.y.float())
            val_loss += loss.item()
        val_loss /= len(test_loader)
        valid_losses.append(val_loss)

    scheduler.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Valid Loss: {val_loss:.4f}')


# visualizing the varerage train and test loss    
avg_train_loss = sum(train_losses) / len(train_losses)
print(f'Average Train Loss: {avg_train_loss:.4f}')

avg_test_loss = sum(valid_losses) / len(valid_losses)
print(f'Average Test Loss: {avg_test_loss:.4f}')

# Plotting the learning curves
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
