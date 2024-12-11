import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# Load dataset from CSV file
with tqdm(total=7, desc='Overall Progress') as pbar:
    df = pd.read_csv('/mnt/c/Users/dinta/OneDrive/Desktop/mlproj/output/compiled_features_cleaned.csv')
    pbar.update(1)

    # Feature matrix (X) and target vector (y)
    X = df[['Rsh_(ohm-cm2)', 'Rs_(ohm-cm2)', 'n_at_1_sun', 'n_at_ 1/10_suns', 'Jo2_(A/cm2)']].values
    y = df['Efficiency_(percent)'].values
    pbar.update(1)

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pbar.update(1)

    # Feature scaling (standardization)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    pbar.update(1)

    # Create output directory if it doesn't exist
    output_dir = 'plotoutputRFR'
    os.makedirs(output_dir, exist_ok=True)
    pbar.update(1)

    # Train Random Forest Regressor
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(X_train_scaled, y_train)
    pbar.update(1)

    # Predicting and calculating loss (Mean Squared Error) on the test set
    y_pred = rfr.predict(X_test_scaled)
    loss_value = mean_squared_error(y_test, y_pred)
    print(f"Loss for Random Forest Regressor on Test Set: {loss_value}")
    pbar.update(1)

# Save the loss to a CSV file
loss_records = [{'model': 'Random Forest Regressor', 'loss': loss_value}]
loss_csv_path = os.path.join(output_dir, 'rfr_loss_records.csv')
loss_df = pd.DataFrame(loss_records)
loss_df.to_csv(loss_csv_path, index=False)

# Feature importance for Random Forest Regressor
feature_importances = rfr.feature_importances_

# Plot feature importance
feature_names = ['Rsh_(ohm-cm2)', 'Rs_(ohm-cm2)', 'n_at_1_sun', 'n_at_ 1/10_suns', 'Jo2_(A/cm2)']
plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importances, color='b')
plt.xlabel('Features')
plt.ylabel('Importance (Random Forest Feature Importance)')
plt.title('Feature Importance for Random Forest Regressor')

# Save the feature importance plot to the output directory
feature_importance_path = os.path.join(output_dir, 'rfr_feature_importance_plot.png')
plt.savefig(feature_importance_path)

# Plotting loss for Random Forest Regressor
plt.figure(figsize=(10, 6))
plt.bar(['Random Forest Regressor'], [loss_value], color='g')
plt.xlabel('Model')
plt.ylabel('Loss (MSE)')
plt.title('Random Forest Regressor Loss on Test Set')

# Save the loss plot to the output directory
loss_plot_path = os.path.join(output_dir, 'rfr_loss_plot.png')
plt.savefig(loss_plot_path)
