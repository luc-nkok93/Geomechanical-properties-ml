# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:32:29 2025

@author: yvann
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load your real dataset
df = pd.read_csv(r"C:\Users\yvann\OneDrive - North Dakota University System\Application of ML in Petrophysics\Sample_dataset.csv")  # Replace with your file path

# Handle missing values if necessary
df = df.replace(-999.25, np.nan).dropna()

# Define features (X) and targets (y)
X = df.iloc[:, :6].values  # First 6 columns as features
y = df.iloc[:, 6:].values  # Last 2 columns as targets

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree Regressors for each target variable
models = []
for i in range(y_train.shape[1]):
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X_train, y_train[:, i])
    models.append(model)

# Make predictions
y_pred_all = np.column_stack([model.predict(X_test) for model in models])

# Evaluate performance
for i, target_name in enumerate(["DTCO", "DTSM"]):
    r2 = r2_score(y_test[:, i], y_pred_all[:, i])
    rmse = mean_squared_error(y_test[:, i], y_pred_all[:, i], squared=False)
    mae = mean_absolute_error(y_test[:, i], y_pred_all[:, i])
    print(f"{target_name} - R²: {r2:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Visualize results for DTCO
plt.figure(figsize=(8, 6))
plt.scatter(y_test[:, 0], y_pred_all[:, 0], color='blue', label='Predicted vs Actual')
plt.plot([min(y_test[:, 0]), max(y_test[:, 0])],
         [min(y_test[:, 0]), max(y_test[:, 0])], 'r--', label='Ideal Fit')
plt.xlabel("Actual DTCO")
plt.ylabel("Predicted DTCO")
plt.title("Decision Tree Regression - DTCO")
plt.legend()
plt.grid(True)
plt.show()

# Visualize results for DTSM
plt.figure(figsize=(8, 6))
plt.scatter(y_test[:, 1], y_pred_all[:, 1], color='green', label='Predicted vs Actual')
plt.plot([min(y_test[:, 1]), max(y_test[:, 1])],
         [min(y_test[:, 1]), max(y_test[:, 1])], 'r--', label='Ideal Fit')
plt.xlabel("Actual DTSM")
plt.ylabel("Predicted DTSM")
plt.title("Decision Tree Regression - DTSM")
plt.legend()
plt.grid(True)
plt.show()