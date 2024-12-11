import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import os
import joblib

# Load dataset from CSV file
df = pd.read_csv('/mnt/c/Users/PVRL-01/Documents/Donald Intal/mlproj/output/compiled_features_cleaned.csv')

# Feature matrix (X) and target vector (y)
X = df[['Rsh_(ohm-cm2)', 'Rs_(ohm-cm2)', 'n_at_1_sun', 'n_at_ 1/10_suns', 'Jo2_(A/cm2)']].values
y = df['Efficiency_(percent)'].values

# Split dataset into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Create output directory if it doesn't exist
output_dir = 'plotoutputSVR'
os.makedirs(output_dir, exist_ok=True)

# Lists to store metrics for each iteration
train_r2_scores = []
val_r2_scores = []
train_mses = []
val_mses = []
num_epochs = 1

# Train the SVM model for multiple epochs (iterations)
for epoch in range(num_epochs):
    # Create and train the SVM model with RBF kernel
    svm_model = SVR(kernel='rbf', C=1, epsilon=0.1)
    svm_model.fit(X_train_scaled, y_train)

    # Predict efficiency for the training and validation sets
    y_train_pred = svm_model.predict(X_train_scaled)
    y_val_pred = svm_model.predict(X_val_scaled)

    # Calculate R² Score for training and validation sets
    train_r2_score = r2_score(y_train, y_train_pred)
    val_r2_score = r2_score(y_val, y_val_pred)
    train_r2_scores.append(train_r2_score)
    val_r2_scores.append(val_r2_score)

    # Calculate Mean Squared Error for training and validation sets
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    train_mses.append(train_mse)
    val_mses.append(val_mse)

    # Print the results for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Training R² Score: {train_r2_score:.4f}, Validation R² Score: {val_r2_score:.4f}, Training MSE: {train_mse:.4f}, Validation MSE: {val_mse:.4f}")

# Plot Training and Validation R² Score over Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_r2_scores, label='Training Accuracy (R² Score)', color='g')
plt.plot(range(1, num_epochs + 1), val_r2_scores, label='Validation Accuracy (R² Score)', color='r')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (R² Score)')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()

# Save the training and validation accuracy plot
accuracy_plot_path = os.path.join(output_dir, 'svr_training_validation_accuracy.png')
plt.savefig(accuracy_plot_path)
print("Training and validation accuracy plot saved to:", accuracy_plot_path)

# Plot Training Loss over Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_mses, label='Training Loss (MSE)', color='b')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Over Epochs')
plt.legend()

# Save the training loss plot
loss_plot_path = os.path.join(output_dir, 'svr_training_loss.png')
plt.savefig(loss_plot_path)
print("Training loss plot saved to:", loss_plot_path)

# Plot predictions vs actual values for training and validation sets
plt.figure(figsize=(14, 6))

# Training Set Plot
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
plt.xlabel('Actual Efficiency (%)')
plt.ylabel('Predicted Efficiency (%)')
plt.title('Training Set: Actual vs Predicted Efficiency')

# Validation Set Plot
plt.subplot(1, 2, 2)
plt.scatter(y_val, y_val_pred, color='red', alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('Actual Efficiency (%)')
plt.ylabel('Predicted Efficiency (%)')
plt.title('Validation Set: Actual vs Predicted Efficiency')

plt.tight_layout()

# Save the plot to the output directory
plot_path = os.path.join(output_dir, 'svr_actual_vs_predicted.png')
plt.savefig(plot_path)
print("Plot saved to:", plot_path)
