import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import os
from sklearn.model_selection import train_test_split

# Load dataset from CSV file
df = pd.read_csv('/mnt/c/Users/dinta/OneDrive/Desktop/mlproj/output/compiled_features_cleaned.csv')

# Feature matrix (X) and target vector (y)
X = df[['Rsh_(ohm-cm2)', 'Rs_(ohm-cm2)', 'n_at_1_sun', 'n_at_ 1/10_suns', 'Jo2_(A/cm2)']].values
y = df['Efficiency_(percent)'].values.reshape(-1, 1)

# Feature scaling (normalization)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Adding a bias column (intercept term)
X = np.c_[np.ones(X.shape[0]), X]

# Split dataset into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors for GPU support
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Hyperparameters
learning_rate = 0.01  # Use the best learning rate found earlier
num_epochs = 1000
l2_lambda = 0.1 #regularization

# Create output directory if it doesn't exist
output_dir = 'plotoutputGD'
os.makedirs(output_dir, exist_ok=True)

# Gradient Descent with Accuracy Tracking
def gradient_descent(X_train, y_train, X_val, y_val, theta, learning_rate, l2_lambda, num_epochs):
    m = len(y_train)
    loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    for epoch in tqdm(range(num_epochs)):
        # Calculate the prediction for training set
        predictions_train = X_train.mm(theta)
        
        # Calculate the error
        error = predictions_train - y_train
        
        # Calculate the loss (Mean Squared Error) with regularization
        loss = (1/(2*m)) * torch.sum(error ** 2) + (l2_lambda / (2 * m)) * torch.sum(theta[1:] ** 2)
        loss_history.append(loss.item())

        # Calculate training accuracy (R² score in this case)
        y_train_mean = torch.mean(y_train)
        ss_total_train = torch.sum((y_train - y_train_mean) ** 2)
        ss_residual_train = torch.sum((y_train - predictions_train) ** 2)
        r2_train_score = 1 - (ss_residual_train / ss_total_train)
        train_accuracy_history.append(r2_train_score.item())

        # Calculate validation accuracy (R² score)
        with torch.no_grad():
            predictions_val = X_val.mm(theta)
            y_val_mean = torch.mean(y_val)
            ss_total_val = torch.sum((y_val - y_val_mean) ** 2)
            ss_residual_val = torch.sum((y_val - predictions_val) ** 2)
            r2_val_score = 1 - (ss_residual_val / ss_total_val)
            val_accuracy_history.append(r2_val_score.item())

        # Backpropagation to calculate the gradient
        loss.backward()

        # Update the parameters using the gradients
        with torch.no_grad():
            theta -= learning_rate * theta.grad

        # Zero the gradients after updating
        theta.grad.zero_()

        # Output the loss, training accuracy, and validation accuracy at the end of each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Training Accuracy: {r2_train_score.item():.4f}, Validation Accuracy: {r2_val_score.item():.4f}")
    
    # Return final metrics and theta
    return theta, loss_history, train_accuracy_history, val_accuracy_history, r2_train_score.item(), r2_val_score.item()

# Initialize theta with correct shape
theta = torch.zeros((X_train.shape[1], 1), dtype=torch.float32, requires_grad=True)

# Run gradient descent for 100 epochs
optimal_theta, loss_history, train_accuracy_history, val_accuracy_history, final_train_accuracy, final_val_accuracy = gradient_descent(
    X_train, y_train, X_val, y_val, theta, learning_rate, l2_lambda, num_epochs
)

# Plotting Training Loss History
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss (MSE)', color='b', linewidth=7)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Epoch', fontsize=16, fontweight='bold')
plt.ylabel('Loss (MSE)', fontsize=16, fontweight='bold')
plt.title('Training Loss Over Epochs', fontsize=16, fontweight='bold')
plt.legend(prop={'weight': 'bold', 'size': 16})

# Save the training loss plot to the output directory
training_loss_path = os.path.join(output_dir, 'training_loss_plot.png')
plt.savefig(training_loss_path)

print("Training loss plot saved to:", training_loss_path)

# Combined Training and Validation Accuracy Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_accuracy_history, label='Training Accuracy (R² Score)', color='g', linewidth=7)
plt.plot(range(1, num_epochs + 1), val_accuracy_history, label='Validation Accuracy (R² Score)', color='r', linewidth=4)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (R² Score)', fontsize=12, fontweight='bold')
plt.title('Training and Validation Accuracy Over Epochs', fontsize=16, fontweight='bold')
plt.legend(prop={'weight': 'bold', 'size': 16})

# Save the combined accuracy plot to the output directory
accuracy_plot_path = os.path.join(output_dir, 'accuracy_plot.png')
plt.savefig(accuracy_plot_path)

print("Training and validation accuracy plot saved to:", accuracy_plot_path)

# Save the optimized weights (theta values) to a CSV file
weights_output_path = os.path.join(output_dir, 'optimized_weights.csv')
weights_df = pd.DataFrame(optimal_theta.detach().numpy(), columns=['Weight'])
weights_df.to_csv(weights_output_path, index=False)

# Display the optimized weights (theta values) for the final model
print("Optimized weights (theta):")
print(optimal_theta.detach().numpy())

# Display final training and validation accuracies
print(f"Final Training Accuracy (R² Score): {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy (R² Score): {final_val_accuracy:.4f}")

# Predict efficiency using the optimized weights
def predict(X, theta):
    return X.mm(theta)

# Predict efficiency for the training set
predictions_train = predict(X_train, optimal_theta)
predictions_val = predict(X_val, optimal_theta)

# Convert predictions to numpy array for easier handling
predictions_train_np = predictions_train.detach().numpy()
predictions_val_np = predictions_val.detach().numpy()

# Save predictions to a CSV file
predictions_output_path_train = os.path.join(output_dir, 'efficiency_predictions_train.csv')
predictions_output_path_val = os.path.join(output_dir, 'efficiency_predictions_val.csv')
predictions_df_train = pd.DataFrame(predictions_train_np, columns=['Predicted_Efficiency_(percent)'])
predictions_df_val = pd.DataFrame(predictions_val_np, columns=['Predicted_Efficiency_(percent)'])
predictions_df_train.to_csv(predictions_output_path_train, index=False)
predictions_df_val.to_csv(predictions_output_path_val, index=False)

print("Training set predictions saved to:", predictions_output_path_train)
print("Validation set predictions saved to:", predictions_output_path_val)

# Feature Importance Plot
feature_names = ['Bias', 'Rsh_(ohm-cm2)', 'Rs_(ohm-cm2)', 'n_at_1_sun', 'n_at_1/10_suns', 'Jo2_(A/cm2)']
weights = optimal_theta.detach().numpy().flatten()
feature_importance = np.abs(weights[1:])  # Exclude the bias term

plt.figure(figsize=(10, 6))
plt.bar(feature_names[1:], feature_importance, color='b')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel('Features', fontsize=16, fontweight='bold')
plt.ylabel('Importance (Absolute Weight)', fontsize=16, fontweight='bold')
plt.title('Feature Importance Based on Optimized Weights', fontsize=16, fontweight='bold')

# Save the feature importance plot to the output directory
feature_importance_path = os.path.join(output_dir, 'feature_importance_plot.png')
plt.savefig(feature_importance_path)

print("Feature importance plot saved to:", feature_importance_path)

# Scatter Plot for Predicted Efficiency
def plot_predicted_efficiency(y_true_train, y_pred_train, y_true_val, y_pred_val, output_dir='plotoutputGD'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'predicted_efficiency_plot.png')
    
    plt.figure(figsize=(10, 6))

    # Training data
    plt.scatter(y_true_train, y_pred_train, color='blue', label='Training Data', s=40, edgecolor='black', linewidth=1.5)
    # Validation data
    plt.scatter(y_true_val, y_pred_val, color='red', label='Validation Data', s=40, edgecolor='black', linewidth=1.5)

    # Plot formatting
    plt.xlabel('True Efficiency (%)', fontsize=16, fontweight='bold')
    plt.ylabel('Predicted Efficiency (%)', fontsize=16, fontweight='bold')
    plt.title('True vs Predicted Efficiency', fontsize=16, fontweight='bold')
    plt.legend(prop={'weight': 'bold', 'size': 16})
    plt.grid(True, linewidth=0.8, linestyle='--')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.gca().spines['top'].set_linewidth(3)
    plt.gca().spines['right'].set_linewidth(3)
    plt.gca().spines['left'].set_linewidth(3)
    plt.gca().spines['bottom'].set_linewidth(3)

    plt.tight_layout()

    # Save the plot to the specified path
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved as {save_path}")

    # Show the plot
    

# After predictions are made, call this function
# Convert tensors to numpy arrays
y_true_train_np = y_train.detach().numpy().flatten()
y_pred_train_np = predictions_train.detach().numpy().flatten()
y_true_val_np = y_val.detach().numpy().flatten()
y_pred_val_np = predictions_val.detach().numpy().flatten()

# Save the scatter plot in the same output directory as other plots
plot_predicted_efficiency(
    y_true_train_np,
    y_pred_train_np,
    y_true_val_np,
    y_pred_val_np,
    output_dir='plotoutputGD'
)

