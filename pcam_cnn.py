# PCam Tumor Detection CNN Project
# ------------------------------------------
# This script implements preprocessing, training, and evaluation
# of a CNN model for the PatchCamelyon (PCam) dataset.
# ------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report

# -------------------------------
# 1. Custom PCam Dataset Loader
# -------------------------------
class PCamDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]  # assumes first column is image path
        label = self.data.iloc[idx, 1]     # assumes second column is label

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# -------------------------------
# 2. Image Preprocessing
# -------------------------------

# Training transformations (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Validation/Testing transformations (no augmentation)
val_test_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------------------------------
# 3. Load Datasets and Dataloaders
# -------------------------------
train_dataset = PCamDataset(csv_file="data/train_labels.csv", transform=train_transform)
val_dataset = PCamDataset(csv_file="data/validation_labels.csv", transform=val_test_transform)
test_dataset = PCamDataset(csv_file="data/test_labels.csv", transform=val_test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------------------------------
# 4. Define CNN Model
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = torch.relu(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = torch.relu(self.conv3(x))
        x = nn.MaxPool2d(2)(x)

        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)).squeeze(1)
        return x

cnn_model = SimpleCNN()

# -------------------------------
# 5. Device Configuration
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = cnn_model.to(device)

# -------------------------------
# 6. Loss Function and Optimizer
# -------------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.0005)

# -------------------------------
# 7. Training Loop
# -------------------------------
num_epochs = 5
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    cnn_model.train()
    total_train_loss = 0.0
    total_val_loss = 0.0

    # Training
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    # Validation
    cnn_model.eval()
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# -------------------------------
# 8. Testing Predictions
# -------------------------------
cnn_model.eval()
test_pred_probs = []
test_pred_labels = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        outputs = cnn_model(images)

        test_pred_probs.extend(outputs.cpu().numpy())
        pred_labels = torch.round(outputs)
        test_pred_labels.extend(pred_labels.cpu().numpy())

test_pred_probs = np.array(test_pred_probs)
test_pred_labels = np.array(test_pred_labels)

# -------------------------------
# 9. Classification Report
# -------------------------------
test_true_labels = []
for images, labels in test_dataloader:
    test_true_labels.extend(labels.numpy())

test_true_labels = np.array(test_true_labels)
pcam_classes = ['Normal', 'Tumor']

report = classification_report(test_true_labels, test_pred_labels, target_names=pcam_classes)
print(report)
