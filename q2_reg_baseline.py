import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have 'loader' and 'test_loader' for training and test data

for data in loader:
    num_targets = 1  # For regression, there is usually one target dimension
    break

num_epochs = 100
train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    random_predictions = np.random.rand(len(loader.dataset), num_targets)
    target_values = np.array([data.y.numpy() for data in loader.dataset])
    
    train_loss = np.mean(np.square(random_predictions - target_values))
    train_losses.append(train_loss)

    random_val_predictions = np.random.rand(len(test_loader.dataset), num_targets)
    val_target_values = np.array([data.y.numpy() for data in test_loader.dataset])
    
    val_loss = np.mean(np.square(random_val_predictions - val_target_values))
    valid_losses.append(val_loss)
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

# Plotting the learning curves
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
