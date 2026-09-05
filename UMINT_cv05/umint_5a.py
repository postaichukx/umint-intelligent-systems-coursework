# ============================================================
# 5a - Klasifikácia bodov pomocou MLP v PyTorch
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ============================================================
# 1. Seed a device
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. Načítanie dát
# ============================================================
URL = "https://raw.githubusercontent.com/STU-FEI-OUI/UMINT-UNS_data/main/Python_(CSV)/databody.csv"
df = pd.read_csv("databody.csv") if os.path.exists("databody.csv") else pd.read_csv(URL)

X_all = df[["x", "y", "z"]].values.astype(np.float32)
y_raw = df["label"].values

classes = np.sort(np.unique(y_raw))
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
y_all = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)
num_classes = len(classes)

# ============================================================
# 3. Rozdelenie dát (Max 80% train, 20% test)
# ============================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_all, y_all, train_size=0.80, random_state=SEED)

# Normalizácia
x_mean = X_train_raw.mean(axis=0, keepdims=True)
x_std = X_train_raw.std(axis=0, keepdims=True) + 1e-8

X_train = (X_train_raw - x_mean) / x_std
X_test = (X_test_raw - x_mean) / x_std

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(SimpleDataset(X_test, y_test), batch_size=32, shuffle=False)

# ============================================================
# 4. Model, Loss, Optimizer
# ============================================================
hidden_dim = 16 # Môžete meniť pre hľadanie najjednoduchšej siete
epochs = 200
lr = 0.002

model = nn.Sequential(
    nn.Linear(3, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, num_classes)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ============================================================
# 5. Tréning
# ============================================================
history_loss = []
model.train()

for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(X_batch)
    history_loss.append(epoch_loss / len(X_train))

# ============================================================
# 6. Graf učenia a Confusion Matrix
# ============================================================
plt.figure(figsize=(7, 4))
plt.plot(history_loss, label="Train Loss")
plt.title("5a - Priebeh chyby siete počas učenia (Loss vs. Epoch)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

model.eval()
y_pred_all = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        logits = model(X_batch.to(DEVICE))
        y_pred_all.extend(logits.argmax(dim=1).cpu().numpy())

cm = confusion_matrix(y_test, y_pred_all)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (Test Data)")
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.colorbar()
plt.show()

# ============================================================
# 7. Otestovanie 5 definovaných bodov
# ============================================================
print("\nOtestovanie 5 bodov:")
sample_points = X_test_raw[:5]
sample_scaled = (sample_points - x_mean) / x_std
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    preds = model(sample_tensor).argmax(dim=1).cpu().numpy()

for i, p in enumerate(sample_points):
    print(f"Bod [x={p[0]:.2f}, y={p[1]:.2f}, z={p[2]:.2f}] -> Klasifikovaný ako trieda: {idx_to_class[preds[i]]}")