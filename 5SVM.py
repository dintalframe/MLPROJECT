import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os
import joblib

# Load dataset from CSV file
df = pd.read_csv('/mnt/c/Users/dinta/OneDrive/Desktop/mlproj/output/compiled_features_cleaned.csv')
# adfasdfasf
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
# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'epsilon': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

# Initialize SVR model
svr = SVR()

# Set up GridSearchCV
grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)

# Check if parameters have already been saved
best_params_path = os.path.join(output_dir, 'best_svm_params.pkl')

if os.path.exists(best_params_path):
    print("Loading best parameters from saved file...")
    best_params = joblib.load(best_params_path)
    svm_model = SVR(**best_params)
else:
    # Fit the model using GridSearchCV
    import time
    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    end_time = time.time()
    tuning_time = end_time - start_time
    print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds.")

    # Save the best parameters to file
    joblib.dump(grid_search.best_params_, best_params_path)
    print("Best parameters saved to:", best_params_path)
    svm_model = grid_search.best_estimator_

# Incremental Training Simulation for Loss Plot
data_sizes = np.linspace(0.1, 1.0, 10)
training_losses = []
validation_losses = []
training_r2_scores = []
validation_r2_scores = []

for size in data_sizes:
    subset_size = int(size * len(X_train_scaled))
    X_train_subset = X_train_scaled[:subset_size]
    y_train_subset = y_train[:subset_size]

    # Train the model on the subset
    svm_model.fit(X_train_subset, y_train_subset)

    # Predict on training and validation sets
    y_train_subset_pred = svm_model.predict(X_train_subset)
    y_val_pred = svm_model.predict(X_val_scaled)

    # Compute losses
    train_loss = mean_squared_error(y_train_subset, y_train_subset_pred)
    val_loss = mean_squared_error(y_val, y_val_pred)

    # Compute R² scores
    train_r2 = r2_score(y_train_subset, y_train_subset_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    training_losses.append(train_loss)
    validation_losses.append(val_loss)
    training_r2_scores.append(train_r2)
    validation_r2_scores.append(val_r2)

# Plot Training and Validation Loss Over Data Subset Sizes
plt.figure(figsize=(10, 6))
plt.plot(data_sizes * 100, training_losses, label='Training Loss (MSE)', color='blue', marker='o')
plt.plot(data_sizes * 100, validation_losses, label='Validation Loss (MSE)', color='orange', marker='o')
plt.xlabel('Training Data Size (%)')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Over Training Data Sizes')
plt.legend()
loss_plot_path = os.path.join(output_dir, 'incremental_training_loss_plot_svm.png')
plt.savefig(loss_plot_path)
plt.close()

# Plot Training and Validation Accuracy (R²) Over Data Subset Sizes
plt.figure(figsize=(10, 6))
plt.plot(data_sizes * 100, training_r2_scores, label='Training Accuracy (R²)', color='green', marker='o')
plt.plot(data_sizes * 100, validation_r2_scores, label='Validation Accuracy (R²)', color='red', marker='o')
plt.xlabel('Training Data Size (%)')
plt.ylabel('Accuracy (R²)')
plt.title('Training and Validation Accuracy Over Training Data Sizes')
plt.legend()
accuracy_plot_path = os.path.join(output_dir, 'incremental_training_accuracy_plot_svm.png')
plt.savefig(accuracy_plot_path)
plt.close()

# Fit the best estimator to the training data
svm_model.fit(X_train_scaled, y_train)

# Save the trained model
model_path = os.path.join(output_dir, 'svm_model.pkl')
joblib.dump(svm_model, model_path)

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
plt.scatter(y_train, y_train_pred, label='Training Data', alpha=0.7)
plt.scatter(y_val, y_val_pred, label='Validation Data', alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], color='black', linestyle='--')
plt.xlabel('Actual Efficiency (%)')
plt.ylabel('Predicted Efficiency (%)')
plt.title('Actual vs. Predicted Efficiency')
plt.legend()
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

# Print results
print(f"Final Training R² Score: {r2_train_score:.4f}")
print(f"Final Validation R² Score: {r2_val_score:.4f}")
print(f"Final Training MSE: {train_mse:.4f}")
print(f"Final Validation MSE: {val_mse:.4f}")
print("Incremental Training Loss plot saved to:", loss_plot_path)
print("Incremental Training Accuracy plot saved to:", accuracy_plot_path)
print("Actual vs. Predicted plot saved to:", actual_vs_predicted_path)
print("Training set predictions saved to:", predictions_output_path_train)
print("Validation set predictions saved to:", predictions_output_path_val)
