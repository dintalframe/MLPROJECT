import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from tqdm import tqdm
import os
from sklearn.model_selection import ParameterGrid
import shap
import numpy as np

# Load dataset from CSV file
df = pd.read_csv('/mnt/c/Users/PVRL-01/OneDrive - University of North Carolina at Charlotte/Old_Files/Backup 09162020/Documents/Donald Intal/mlproj/output/compiled_features_cleaned.csv')
#testing
# Feature matrix (X) and target vector (y)
X = df[['Rsh_(ohm-cm2)', 'Rs_(ohm-cm2)', 'n_at_1_sun', 'n_at_ 1/10_suns', 'Jo2_(A/cm2)']].values
y = df['Efficiency_(percent)'].values

# Split dataset into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert to PyTorch tensors and create DataLoader
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

train_dataset = data_utils.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = data_utils.TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 128
num_workers = 4

train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 300

# Create model, define loss function and optimizer
model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Create output directory if it doesn't exist
output_dir = 'plotoutputNN'
os.makedirs(output_dir, exist_ok=True)

# Lists to store metrics
train_losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device, non_blocking=True)
        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch.view(-1, 1))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * X_batch.size(0)
    
    # Average training loss for the epoch
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

# Plot Training Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
loss_plot_path = os.path.join(output_dir, 'training_loss_plot.png')
plt.savefig(loss_plot_path)
plt.show()
print("Training loss plot saved to:", loss_plot_path)
