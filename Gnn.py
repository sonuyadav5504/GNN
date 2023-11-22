import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import matplotlib.pyplot as plt

for data in loader:
    num_features = data.x.shape[1]  
    break

class MyGINRegression(torch.nn.Module):
    def __init__(self):
        super(MyGINRegression, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(num_features, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5) 
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)  
        ))
#         self.conv3 = GINConv(torch.nn.Sequential(
#             torch.nn.Linear(64, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 128),
#             torch.nn.BatchNorm1d(128),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.5) 
#         ))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
#         x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.mlp(x).squeeze(1)
        return x

model = MyGINRegression()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

num_epochs = 100
train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for data in loader:
        optimizer.zero_grad()
        edge_index = data.edge_index.to(torch.int64)
        out = model(data.x.float(), edge_index, data.batch)
        out = out.view(-1, 1)
        target = data.y.view(-1, 1).float()
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
            edge_index = data.edge_index.to(torch.int64)
            out = model(data.x.float(), edge_index, data.batch)
            out = out.view(-1, 1)
            target = data.y.view(-1, 1).float()
            loss = criterion(out, target)
            
            val_loss += loss.item()
        val_loss /= len(loader)
        valid_losses.append(val_loss)
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Valid Loss: {val_loss:.4f}')

plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
