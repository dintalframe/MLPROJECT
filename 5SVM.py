import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os
import joblib
from tqdm import tqdm
import shap

# Load dataset from CSV file
df = pd.read_csv('/mnt/c/Users/PVRL-01/Desktop/Github/MLPROJECT/output/compiled_features_cleaned.csv')

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
output_dir = 'plotoutputSVM'
os.makedirs(output_dir, exist_ok=True)

# Save the scaler for future use
scaler_path = os.path.join(output_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)

# Support Vector Regression (SVR) Model with RBF Kernel
svm_model = SVR()

# Path to save the trained model and losses
model_path = os.path.join(output_dir, 'svm_model.pkl')
losses_path = os.path.join(output_dir, 'training_losses.npy')

# Initialize variables
training_losses = None
training_r2_scores = None
validation_r2_scores = None

# Check if the model already exists
if os.path.exists(model_path) and os.path.exists(losses_path):
    print("Loading saved model and training losses...")
    svm_model = joblib.load(model_path)
    training_losses = np.load(losses_path)
else:
    print("Training model from scratch...")

    # Simulating epoch-like training for loss tracking with progressive regularization decay
    pseudo_epochs = 1000
    initial_C = 0.1
    final_C = 100
    training_losses = []
    training_r2_scores = []
    validation_r2_scores = []

    for epoch in tqdm(range(pseudo_epochs), desc="Training Epochs"):
        # Update regularization parameter (C) progressively
        C_current = initial_C - (initial_C - final_C) * (epoch / pseudo_epochs)
        svm_model.set_params(C=C_current)

        # Train the model on the full dataset
        svm_model.fit(X_train_scaled, y_train)

        # Predict on the training and validation sets
        y_train_pred = svm_model.predict(X_train_scaled)
        y_val_pred = svm_model.predict(X_val_scaled)

        # Compute training and validation R^2 scores
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        training_r2_scores.append(train_r2)
        validation_r2_scores.append(val_r2)

        # Compute training loss
        train_loss = mean_squared_error(y_train, y_train_pred)
        training_losses.append(train_loss)

    # Save the trained model and training losses
    joblib.dump(svm_model, model_path)
    np.save(losses_path, training_losses)

# Plot Training Loss if Available
if training_losses is not None:
    # Apply a moving average to smooth the training loss
    from scipy.ndimage import uniform_filter1d
    training_losses_smooth = uniform_filter1d(training_losses, size=10)

    # Plot Smoothed Training Loss Over Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses_smooth) + 1), training_losses_smooth, label='Training Loss (Smoothed)', color='blue', linewidth=7)
    plt.xticks(rotation=0, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=16, fontweight='bold')
    plt.title('Training Loss Over Epochs (With Decaying Regularization)', fontsize=16, fontweight='bold')
    plt.legend(prop={'weight': 'bold', 'size': 16})
    #plt.grid()
    loss_plot_path = os.path.join(output_dir, 'training_loss_with_decay_svm.png')
    plt.savefig(loss_plot_path)
    plt.close()

# Plot Training and Validation Accuracy if Available
if training_r2_scores is not None and validation_r2_scores is not None:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, pseudo_epochs + 1), training_r2_scores, label='Training Accuracy (R²)', color='green', linewidth=7)
    plt.plot(range(1, pseudo_epochs + 1), validation_r2_scores, label='Validation Accuracy (R²)', color='red', linewidth=4)
    plt.xticks(rotation=0, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy (R²)', fontsize=16, fontweight='bold')
    plt.title('Training and Validation Accuracy Over Epochs', fontsize=16, fontweight='bold')
    plt.legend(prop={'weight': 'bold', 'size': 16})
    #plt.grid()
    accuracy_plot_path = os.path.join(output_dir, 'training_validation_accuracy_svm.png')
    plt.savefig(accuracy_plot_path)
    plt.close()

# Predictions for training and validation sets
y_train_pred = svm_model.predict(X_train_scaled)
y_val_pred = svm_model.predict(X_val_scaled)

# Calculate R² scores for training and validation sets
r2_train_score = r2_score(y_train, y_train_pred)
r2_val_score = r2_score(y_val, y_val_pred)

# Calculate Mean Squared Error (MSE) for training and validation sets
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

# Plot Actual vs. Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, label='Predicted vs True', alpha=0.7, color='blue')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel('True Efficiency (%)', fontsize=16, fontweight='bold')
plt.ylabel('Predicted Efficiency (%)', fontsize=16, fontweight='bold')
plt.title('Predicted vs True Efficiency', fontsize=16, fontweight='bold')
plt.legend(prop={'weight': 'bold', 'size': 16})
actual_vs_predicted_path = os.path.join(output_dir, 'actual_vs_predicted_efficiency_svm.png')
plt.savefig(actual_vs_predicted_path)
plt.close()

# Save predictions to CSV files
predictions_output_path_train = os.path.join(output_dir, 'efficiency_predictions_train_svm.csv')
predictions_output_path_val = os.path.join(output_dir, 'efficiency_predictions_val_svm.csv')

predictions_df_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
predictions_df_val = pd.DataFrame({'Actual': y_val, 'Predicted': y_val_pred})

predictions_df_train.to_csv(predictions_output_path_train, index=False)
predictions_df_val.to_csv(predictions_output_path_val, index=False)

# SHAP Analysis
print("Performing SHAP analysis...")
# Summarize background data using k-means clustering (e.g., K=100)
background = shap.kmeans(X_train_scaled, 100)

explainer = shap.KernelExplainer(svm_model.predict, background)
shap_values = explainer.shap_values(X_val_scaled[:100])

# Bar Plot for Feature Importance Using SHAP
mean_shap_values = np.mean(np.abs(shap_values), axis=0)
feature_names = ['Rsh_(ohm-cm2)', 'Rs_(ohm-cm2)', 'n_at_1_sun', 'n_at_1/10_suns', 'Jo2_(A/cm2)']
plt.figure(figsize=(10, 6))
plt.bar(feature_names, mean_shap_values, color='blue')
plt.xlabel('Features', fontsize=16, fontweight='bold')
plt.ylabel('SHAP Importance', fontsize=16, fontweight='bold')
plt.title('Feature Importance Using SHAP', fontsize=16, fontweight='bold')
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
#plt.grid(axis='y')
shap_bar_plot_path = os.path.join(output_dir, 'shap_feature_importance_bar.png')
plt.savefig(shap_bar_plot_path)
plt.close()

# Save SHAP feature importance to CSV
shap_importance_path = os.path.join(output_dir, 'shap_feature_importance.csv')
shap_importance_df = pd.DataFrame({'Feature': feature_names, 'Mean SHAP Importance': mean_shap_values})
shap_importance_df.to_csv(shap_importance_path, index=False)

# Summary plot
shap.summary_plot(shap_values, X_val_scaled[:100], feature_names=feature_names, show=False)
shap_plot_path = os.path.join(output_dir, 'shap_summary_plot.png')
plt.savefig(shap_plot_path)
plt.close()

# Print results
print(f"Final Training R² Score: {r2_train_score:.4f}")
print(f"Final Validation R² Score: {r2_val_score:.4f}")
print(f"Final Training MSE: {train_mse:.4f}")
print(f"Final Validation MSE: {val_mse:.4f}")
if training_losses is not None:
    print("Training Loss plot saved to:", loss_plot_path)
if training_r2_scores is not None and validation_r2_scores is not None:
    print("Accuracy plot saved to:", accuracy_plot_path)
print("Actual vs. Predicted plot saved to:", actual_vs_predicted_path)
print("SHAP summary plot saved to:", shap_plot_path)
print("SHAP feature importance bar plot saved to:", shap_bar_plot_path)
print("SHAP feature importance saved to:", shap_importance_path)
print("Training set predictions saved to:", predictions_output_path_train)
print("Validation set predictions saved to:", predictions_output_path_val)
