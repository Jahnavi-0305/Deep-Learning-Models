# Predicting Residential EV Charging Loads using Neural Networks

# Setup - import basic data libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Task Group 1 - Load, Inspect, and Merge Datasets
# Task 1: Load EV charging reports
ev_charging_reports = pd.read_csv('datasets/EV charging reports.csv')
print(ev_charging_reports.head())

# Task 2: Load traffic reports
traffic_reports = pd.read_csv('datasets/Local traffic distribution.csv')
print(traffic_reports.head())

# Task 3: Merge datasets
ev_charging_traffic = ev_charging_reports.merge(
    traffic_reports,
    left_on='Start_plugin_hour',
    right_on='Date_from',
    how='inner'
)

# Task 4: Inspect merged dataset
print(ev_charging_traffic.info())

# Task Group 2 - Data Cleaning and Preparation
# Task 5: Drop unnecessary columns
columns_to_drop = [
    'session_ID', 'Garage_ID', 'User_ID', 'Shared_ID',
    'Plugin_category', 'Duration_category', 'Start_plugin',
    'Start_plugin_hour', 'End_plugout', 'End_plugout_hour',
    'Date_from', 'Date_to'
]
ev_charging_traffic.drop(columns=[col for col in columns_to_drop if col in ev_charging_traffic.columns], inplace=True)

# Task 6: Replace commas with dots and convert to float
ev_charging_traffic['El_kWh'] = ev_charging_traffic['El_kWh'].astype(str).str.replace(',', '.').astype(float)
ev_charging_traffic['Duration_hours'] = ev_charging_traffic['Duration_hours'].astype(str).str.replace(',', '.').astype(float)
ev_charging_traffic['User_private'] = ev_charging_traffic['User_private'].astype(str).str.replace(',', '.').astype(float)

# Task 7: Convert all columns to float
ev_charging_traffic = ev_charging_traffic.astype(float)

# Task Group 3 - Train Test Split
# Task 8: Create feature and target datasets
X = ev_charging_traffic.drop(columns='El_kWh')
y = ev_charging_traffic['El_kWh']

# Task 9: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=2
)

# Task Group 4 - Linear Regression Baseline
# Task 10 & 11: Train and evaluate Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Linear Regression Mean Squared Error: {test_mse:.4f}")

# Task Group 5 - Train a Neural Network Using PyTorch
# Task 12: Import PyTorch libraries - already imported

# Task 13: Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# Task 14: Create the neural network model
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 56),
    nn.ReLU(),
    nn.Linear(56, 26),
    nn.ReLU(),
    nn.Linear(26, 1)
)

# Task 15: Define loss and optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0007)

# Task 16: Training loop
for epoch in range(3000):
    model.train()
    optimizer.zero_grad()
    y_pred_train = model(X_train_tensor)
    train_loss = loss(y_pred_train, y_train_tensor)
    train_loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, MSE: {train_loss.item():.4f}")

# Task 17: Save the model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/model.pth')

# Task 18: Evaluate on test set
model.eval()
y_pred_test = model(X_test_tensor)
test_loss = loss(y_pred_test, y_test_tensor).item()
print(f"Test Loss (Neural Network): {test_loss:.4f}")

# Task 19: Load pre-trained model and evaluate
model_4500 = nn.Sequential(
    nn.Linear(X_train.shape[1], 56),
    nn.ReLU(),
    nn.Linear(56, 26),
    nn.ReLU(),
    nn.Linear(26, 1)
)
model_4500.load_state_dict(torch.load('models/model4500.pth'))
model_4500.eval()
y_pred_test_4500 = model_4500(X_test_tensor)
test_loss_4500 = loss(y_pred_test_4500, y_test_tensor).item()
print(f"Test Loss (Model Trained 4500 Epochs): {test_loss_4500:.4f}")
