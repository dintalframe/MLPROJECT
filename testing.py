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
df = pd.read_csv('/mnt/c/Users/PVRL-01/Desktop/Github/MLPROJECT/output/compiled_features_cleaned.csv')
##dfdkjfkdjfd
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
print(f"Using device: {device}")  # Debug statement

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
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.01
num_epochs = 100

# Create model, define loss function and optimizer
model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Create output directory if it doesn't exist
output_dir = 'plotoutputNN'
os.makedirs(output_dir, exist_ok=True)

# Lists to store metrics
train_losses = []
train_r2_scores = []
val_r2_scores = []

# Training loop
for epoch in tqdm(range(num_epochs), desc="Training Progress"):
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

    # Calculate R² score for training
    with torch.no_grad():
        model.eval()
        train_preds = model(X_train_tensor.to(device))
        train_r2 = 1 - (torch.sum((y_train_tensor.to(device) - train_preds) ** 2) / torch.sum((y_train_tensor.to(device) - torch.mean(y_train_tensor.to(device))) ** 2))
        train_r2_scores.append(train_r2.item())

    # Calculate R² score for validation
    with torch.no_grad():
        val_preds = model(X_val_tensor.to(device))
        val_r2 = 1 - (torch.sum((y_val_tensor.to(device) - val_preds) ** 2) / torch.sum((y_val_tensor.to(device) - torch.mean(y_val_tensor.to(device))) ** 2))
        val_r2_scores.append(val_r2.item())

# Save the final training loss to CSV and print to console
final_training_loss = {'Metric': ['Final Training Loss'], 'Value': [train_losses[-1]]}
final_loss_df = pd.DataFrame(final_training_loss)
final_loss_csv_path = os.path.join(output_dir, 'final_training_loss.csv')
final_loss_df.to_csv(final_loss_csv_path, index=False)
print(f"Final Training Loss saved to: {final_loss_csv_path}")
print("Final Training Loss:", train_losses[-1])

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', linewidth=7)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel('Epochs', fontsize=16, fontweight='bold')
plt.ylabel('Loss', fontsize=16, fontweight='bold')
plt.title('Training Loss Over Epochs', fontsize=16, fontweight='bold')
plt.legend(prop={'weight': 'bold', 'size': 16})
loss_plot_path = os.path.join(output_dir, 'training_loss_plot.png')
plt.savefig(loss_plot_path)
print("Training loss plot saved to:", loss_plot_path)

# Plot Training and Validation Accuracy (R² Score)
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_r2_scores, label='Training R² Score', color='green',linewidth=7)
plt.plot(range(1, num_epochs + 1), val_r2_scores, label='Validation R² Score', color='orange', linewidth=4)
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=16, fontweight='bold')
plt.ylabel('R² Score', fontsize=16, fontweight='bold')
plt.title('Training and Validation R² Score Over Epochs', fontsize=16, fontweight='bold')
plt.legend(prop={'weight': 'bold', 'size': 16})
r2_plot_path = os.path.join(output_dir, 'r2_score_plot.png')
plt.savefig(r2_plot_path)
print("Training and validation R² score plot saved to:", r2_plot_path)

# Save final R² scores to CSV and print to console
final_r2_scores = {
    'Metric': ['Training R²', 'Validation R²'],
    'R² Score': [train_r2_scores[-1], val_r2_scores[-1]]
}
final_r2_df = pd.DataFrame(final_r2_scores)
r2_csv_path = os.path.join(output_dir, 'final_r2_scores.csv')
final_r2_df.to_csv(r2_csv_path, index=False)
print("Final R² scores saved to:", r2_csv_path)
print(final_r2_df)

# Feature Importance Using SHAP
print(f"Running SHAP on device: {device}")  # Debug statement
explainer = shap.DeepExplainer(model, X_train_tensor.to(device))  # Use the entire training set
shap_values = []
print("Starting SHAP computation...")
for i in tqdm(range(X_val_tensor.size(0)), desc="SHAP Progress"):
    single_shap_values = explainer.shap_values(X_val_tensor[i].unsqueeze(0).to(device))[0]
    shap_values.append(single_shap_values)
shap_values = np.squeeze(np.array(shap_values), axis=-1)  # Remove the last dimension
print("SHAP computation completed.")  # Compute SHAP values for all validation samples

# Debugging lengths
print(f"Flattened SHAP values shape: {shap_values.shape}")
print(f"X_val_scaled shape: {X_val_scaled.shape}")

# Compute mean absolute SHAP values for feature importance
shap_feature_importance = np.abs(shap_values).mean(axis=0)  # Mean SHAP values across all samples

# Debugging feature importance
print(f"SHAP feature importance length: {len(shap_feature_importance)}")

# Save SHAP feature importance to CSV
shap_importance_df = pd.DataFrame({
    'Feature': ['Rsh_(ohm-cm2)', 'Rs_(ohm-cm2)', 'n_at_1_sun', 'n_at_ 1/10_suns', 'Jo2_(A/cm2)'],
    'SHAP Importance': shap_feature_importance
})
shap_importance_csv_path = os.path.join(output_dir, 'shap_feature_importance.csv')
shap_importance_df.to_csv(shap_importance_csv_path, index=False)
print("SHAP feature importance saved to:", shap_importance_csv_path)

# Feature Importance Bar Plot Using SHAP
plt.figure(figsize=(10, 6))
plt.bar(shap_importance_df['Feature'], shap_importance_df['SHAP Importance'], color='blue')
plt.xlabel('Features', fontsize=16, fontweight='bold')
plt.ylabel('SHAP Importance', fontsize=16, fontweight='bold')
plt.title('Feature Importance Using SHAP', fontsize=16, fontweight='bold')
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
shap_bar_plot_path = os.path.join(output_dir, 'shap_feature_importance_plot.png')
plt.savefig(shap_bar_plot_path)
print("SHAP feature importance plot saved to:", shap_bar_plot_path)

# Predict Efficiency and Compare to Ground Truth
with torch.no_grad():
    model.eval()
    predicted_efficiency = model(X_val_tensor.to(device)).cpu().numpy()
    true_efficiency = y_val

# Plot Predicted vs True Efficiency
plt.figure(figsize=(10, 6))
plt.scatter(true_efficiency, predicted_efficiency, alpha=0.7, color='blue', label='Predicted')
plt.plot([min(true_efficiency), max(true_efficiency)], [min(true_efficiency), max(true_efficiency)], color='red', linestyle='--', label='Ideal Prediction')
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xlabel('True Efficiency (%)', fontsize=16, fontweight='bold')
plt.ylabel('Predicted Efficiency (%)', fontsize=16, fontweight='bold')
plt.title('Predicted vs True Efficiency', fontsize=16, fontweight='bold')
plt.legend(prop={'weight': 'bold', 'size': 16})
prediction_plot_path = os.path.join(output_dir, 'predicted_vs_true_efficiency.png')
plt.savefig(prediction_plot_path)
print("Predicted vs True Efficiency plot saved to:", prediction_plot_path)