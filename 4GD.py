import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Load dataset from CSV file (Replace with your dataset path)
df = pd.read_csv('/mnt/c/Users/PVRL-01/Desktop/Github/MLPROJECT/output/compiled_features_cleaned.csv')

# Feature matrix (X) and target vector (y)
X = df[['Rsh_(ohm-cm2)', 'Rs_(ohm-cm2)', 'n_at_1_sun', 'n_at_ 1/10_suns', 'Jo2_(A/cm2)']].values
y = df['Efficiency_(percent)'].values.reshape(-1, 1)

# Feature scaling (normalization)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Add bias term to features
X = np.c_[np.ones(X.shape[0]), X]

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Hyperparameters
learning_rate = 0.01
num_epochs = 1000
l2_lambda = 0.1

# Gradient Descent Function
def gradient_descent(X_train, y_train, X_val, y_val, theta, learning_rate, l2_lambda, num_epochs):
    m = len(y_train)
    loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    for epoch in tqdm(range(num_epochs)):
        predictions_train = X_train.mm(theta)
        error = predictions_train - y_train
        loss = (1/(2*m)) * torch.sum(error ** 2) + (l2_lambda / (2 * m)) * torch.sum(theta[1:] ** 2)
        loss_history.append(loss.item())

        y_train_mean = torch.mean(y_train)
        ss_total_train = torch.sum((y_train - y_train_mean) ** 2)
        ss_residual_train = torch.sum((y_train - predictions_train) ** 2)
        r2_train_score = 1 - (ss_residual_train / ss_total_train)
        train_accuracy_history.append(r2_train_score.item())

        with torch.no_grad():
            predictions_val = X_val.mm(theta)
            y_val_mean = torch.mean(y_val)
            ss_total_val = torch.sum((y_val - y_val_mean) ** 2)
            ss_residual_val = torch.sum((y_val - predictions_val) ** 2)
            r2_val_score = 1 - (ss_residual_val / ss_total_val)
            val_accuracy_history.append(r2_val_score.item())

        loss.backward()

        with torch.no_grad():
            theta -= learning_rate * theta.grad
        theta.grad.zero_()

    return theta, loss_history, train_accuracy_history, val_accuracy_history

# Initialize theta
theta = torch.zeros((X_train.shape[1], 1), dtype=torch.float32, requires_grad=True)

# Train model with gradient descent
optimal_theta, loss_history, train_accuracy_history, val_accuracy_history = gradient_descent(
    X_train, y_train, X_val, y_val, theta, learning_rate, l2_lambda, num_epochs
)

# Save optimized weights to CSV
weights_output_path = os.path.join('plotoutputGD', 'optimized_weights.csv')
pd.DataFrame(optimal_theta.detach().numpy(), columns=['Weight']).to_csv(weights_output_path, index=False)
print(f"Optimized weights saved to {weights_output_path}")

# Save final training and validation accuracies and training loss to CSV
final_train_accuracy = train_accuracy_history[-1]
final_val_accuracy = val_accuracy_history[-1]
final_training_loss = loss_history[-1]
accuracy_output_path = os.path.join('plotoutputGD', 'final_metrics.csv')
pd.DataFrame({
    'Metric': ['Training Accuracy (R²)', 'Validation Accuracy (R²)', 'Final Training Loss'],
    'Value': [final_train_accuracy, final_val_accuracy, final_training_loss]
}).to_csv(accuracy_output_path, index=False)
print(f"Final metrics saved to {accuracy_output_path}")

# Print final training and validation accuracies and training loss
print(f"Final Training Accuracy (R²): {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy (R²): {final_val_accuracy:.4f}")
print(f"Final Training Loss: {final_training_loss:.4f}")

# Predict function
def predict(X, theta):
    return X.mm(theta)

# Predictions for training and validation sets
predictions_train = predict(X_train, optimal_theta)
predictions_val = predict(X_val, optimal_theta)

# Training Loss Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss (MSE)', color='red', linewidth=7)
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xlabel('Epoch', fontsize=16, fontweight='bold')
plt.ylabel('Loss (MSE)', fontsize=16, fontweight='bold')
plt.title('Training Loss Over Epochs', fontsize=16, fontweight='bold')
plt.legend(prop={'weight': 'bold', 'size': 16})
#plt.grid(True, linestyle='--', linewidth=0.7)
training_loss_plot_path = os.path.join('plotoutputGD', 'training_loss_plot.png')
plt.savefig(training_loss_plot_path, dpi=300, bbox_inches='tight')
print(f"Training loss plot saved to {training_loss_plot_path}")

# Accuracy Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_accuracy_history, label='Training Accuracy (R²)', color='blue', linewidth=7)
plt.plot(range(1, num_epochs + 1), val_accuracy_history, label='Validation Accuracy (R²)', color='orange', linewidth=4)
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xlabel('Epoch', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy (R²)', fontsize=16, fontweight='bold')
plt.title('Training and Validation Accuracy Over Epochs', fontsize=16, fontweight='bold')
plt.legend(prop={'weight': 'bold', 'size': 16})
#plt.grid(True, linestyle='--', linewidth=3)
accuracy_plot_path = os.path.join('plotoutputGD', 'accuracy_plot.png')
plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
print(f"Accuracy plot saved to {accuracy_plot_path}")

# Feature Importance Plot
feature_names = ['Bias', 'Rsh_(ohm-cm2)', 'Rs_(ohm-cm2)', 'n_at_1_sun', 'n_at_ 1/10_suns', 'Jo2_(A/cm2)']
weights = optimal_theta.detach().numpy().flatten()
feature_importance = np.abs(weights[1:])  # Exclude the bias term

plt.figure(figsize=(10, 6))
plt.bar(feature_names[1:], feature_importance, color='blue')
plt.xlabel('Features', fontsize=16, fontweight='bold')
plt.ylabel('Importance (Absolute Weight)', fontsize=16, fontweight='bold')
plt.title('Feature Importance Based on Optimized Weights', fontsize=16, fontweight='bold')
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
#plt.grid(True, linestyle='--', linewidth=0.7)
feature_importance_path = os.path.join('plotoutputGD', 'feature_importance_plot.png')
plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
print(f"Feature importance plot saved to {feature_importance_path}")

# Scatter plot with ideal prediction line
def plot_predicted_efficiency_with_ideal_line(y_true, y_pred, title, output_dir):
    save_path = os.path.join(output_dir, 'predicted_vs_true_with_ideal_line.png')

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='blue', label='Predicted vs True', s=40, alpha=0.7)

    min_val = max(0, min(y_true.min(), y_pred.min()))  # Ensure x-axis starts at 0
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Prediction')

    plt.xlabel('True Efficiency (%)', fontsize=16, fontweight='bold')
    plt.ylabel('Predicted Efficiency (%)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=0, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(prop={'weight': 'bold', 'size': 16})
    #plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {save_path}")

# Plot the validation scatter with ideal line
plot_predicted_efficiency_with_ideal_line(
    y_val.detach().numpy().flatten(),
    predictions_val.detach().numpy().flatten(),
    'Predicted vs True Efficiency (Polynomial Features)',
    'plotoutputGD'
)
