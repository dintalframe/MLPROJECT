import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from tqdm import tqdm
import os
from sklearn.model_selection import ParameterGrid

# Load dataset from CSV file
df = pd.read_csv('/mnt/c/Users/dinta/OneDrive/Desktop/mlproj/output/compiled_features_cleaned.csv')

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

batch_size = 128  # Increased batch size to utilize GPU memory
num_workers = 4  # Set number of workers for data loading

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
hidden_size = 64  # Increased hidden layer size
output_size = 1
learning_rate = 0.001  # Reduced learning rate for stability with larger batch size
num_epochs = 300

# Create model, define loss function and optimizer
model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Create output directory if it doesn't exist
output_dir = 'plotoutputNN'
os.makedirs(output_dir, exist_ok=True)

# Lists to store loss and R² score for each epoch
train_losses = []
val_losses = []
train_r2_scores = []
val_r2_scores = []

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

    # Calculate training R² score
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_tensor.to(device))
        y_train_mean = torch.mean(y_train_tensor.to(device))
        ss_total_train = torch.sum((y_train_tensor.to(device) - y_train_mean) ** 2)
        ss_residual_train = torch.sum((y_train_tensor.to(device) - y_train_pred) ** 2)
        train_r2_score = 1 - (ss_residual_train / ss_total_train)
        train_r2_scores.append(train_r2_score.item())

    # Validation pass
    epoch_val_loss = 0.0
    val_predictions = []
    y_vals = []
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device, non_blocking=True)
        with torch.no_grad():
            val_pred = model(X_batch)
            loss = criterion(val_pred, y_batch.view(-1, 1))
            epoch_val_loss += loss.item() * X_batch.size(0)
            val_predictions.append(val_pred)
            y_vals.append(y_batch)
    
    # Average validation loss for the epoch
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    # Calculate validation R² score
    val_predictions = torch.cat(val_predictions)
    y_vals = torch.cat(y_vals)
    y_val_mean = torch.mean(y_vals)
    ss_total_val = torch.sum((y_vals - y_val_mean) ** 2)
    ss_residual_val = torch.sum((y_vals - val_predictions) ** 2)
    val_r2_score = 1 - (ss_residual_val / ss_total_val)
    val_r2_scores.append(val_r2_score.item())

    # Print epoch results
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Training R² Score: {train_r2_score.item():.4f}, Validation R² Score: {val_r2_score.item():.4f}, Training Accuracy: {train_r2_score.item() * 100:.2f}%, Validation Accuracy: {val_r2_score.item() * 100:.2f}%")

# Plot Training and Validation Loss over Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='b')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()

# Save the training and validation loss plot
loss_plot_path = os.path.join(output_dir, 'nn_training_validation_loss.png')
plt.savefig(loss_plot_path)
print("Training and validation loss plot saved to:", loss_plot_path)

# Plot Training and Validation R² Score over Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_r2_scores, label='Training R² Score', color='g')
plt.plot(range(1, num_epochs + 1), val_r2_scores, label='Validation R² Score', color='orange')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('Training and Validation R² Score Over Epochs')
plt.legend()

# Save the training and validation R² score plot
r2_plot_path = os.path.join(output_dir, 'nn_training_validation_r2.png')
plt.savefig(r2_plot_path)
print("Training and validation R² score plot saved to:", r2_plot_path)

# Plot Training and Validation Accuracy over Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), [r2 * 100 for r2 in train_r2_scores], linestyle='-', label='Training Accuracy (%)', color='b')
plt.plot(range(1, num_epochs + 1), [r2 * 100 for r2 in val_r2_scores], linestyle='-', label='Validation Accuracy (%)', color='r')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()

# Save the training and validation accuracy plot
accuracy_plot_path = os.path.join(output_dir, 'nn_training_validation_accuracy.png')
plt.savefig(accuracy_plot_path)
print("Training and validation accuracy plot saved to:", accuracy_plot_path)

# Output final metrics
final_train_r2_score = train_r2_scores[-1]
final_val_r2_score = val_r2_scores[-1]
final_train_accuracy = final_train_r2_score * 100
final_val_accuracy = final_val_r2_score * 100
final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]

print(f"Final Training R² Score: {final_train_r2_score:.4f}")
print(f"Final Validation R² Score: {final_val_r2_score:.4f}")
print(f"Final Training Accuracy: {final_train_accuracy:.2f}%")
print(f"Final Validation Accuracy: {final_val_accuracy:.2f}%")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

# Save final metrics to CSV file
metrics_output_path = os.path.join(output_dir, 'nn_final_metrics.csv')
metrics_df = pd.DataFrame({
    'Metric': ["Training R² Score", "Validation R² Score", "Training Accuracy (%)", "Validation Accuracy (%)", "Training Loss", "Validation Loss"],
    'Value': [final_train_r2_score, final_val_r2_score, final_train_accuracy, final_val_accuracy, final_train_loss, final_val_loss]
})
metrics_df.to_csv(metrics_output_path, index=False)
print("Final metrics saved to:", metrics_output_path)
print("Training and validation R² score and accuracy plot saved to:", r2_plot_path)