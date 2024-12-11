import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from tqdm import tqdm
import os

# Load dataset from CSV file
with tqdm(total=8, desc='Overall Progress') as pbar:
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
    output_dir = 'plotoutputKNN'
    os.makedirs(output_dir, exist_ok=True)
    pbar.update(1)

    # Train K-Nearest Neighbors Regressor
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    pbar.update(1)

    # Predicting and calculating loss (Mean Squared Error) on the test set
    y_pred = knn.predict(X_test_scaled)
    loss_value = mean_squared_error(y_test, y_pred)
    print(f"Loss for K-Nearest Neighbors Regressor on Test Set: {loss_value}")
    pbar.update(1)

    # Permutation importance for K-Nearest Neighbors Regressor
    result = permutation_importance(knn, X_test_scaled, y_test, n_repeats=10, random_state=42)
    feature_importances = result.importances_mean

    # Plot feature importance
    feature_names = ['Rsh_(ohm-cm2)', 'Rs_(ohm-cm2)', 'n_at_1_sun', 'n_at_ 1/10_suns', 'Jo2_(A/cm2)']
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_importances, color='b')
    plt.xlabel('Features')
    plt.ylabel('Importance (Mean Decrease in Score)')
    plt.title('Feature Importance for K-Nearest Neighbors Regressor - Permutation Importance')

    # Save the feature importance plot to the output directory
    feature_importance_path = os.path.join(output_dir, 'knn_feature_importance_plot.png')
    plt.savefig(feature_importance_path)
    pbar.update(1)

# Save the loss to a CSV file
loss_records = [{'model': 'K-Nearest Neighbors Regressor', 'loss': loss_value}]
loss_csv_path = os.path.join(output_dir, 'knn_loss_records.csv')
loss_df = pd.DataFrame(loss_records)
loss_df.to_csv(loss_csv_path, index=False)

# Plotting loss for K-Nearest Neighbors Regressor
plt.figure(figsize=(10, 6))
plt.bar(['K-Nearest Neighbors Regressor'], [loss_value], color='g')
plt.xlabel('Model')
plt.ylabel('Loss (MSE)')
plt.title('K-Nearest Neighbors Regressor Loss on Test Set')

# Save the loss plot to the output directory
loss_plot_path = os.path.join(output_dir, 'knn_loss_plot.png')
plt.savefig(loss_plot_path)
