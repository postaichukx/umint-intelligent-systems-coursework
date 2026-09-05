# ============================================================
# 5b - Aproximácia nelineárnej funkcie pomocou MLP
# ============================================================

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. Seed a device
# ============================================================
SEED = 999
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. Načítanie dát (Nepriestrelná metóda)
# ============================================================
url_fun = "https://raw.githubusercontent.com/STU-FEI-OUI/UMINT-UNS_data/main/Python_(CSV)/datafun.csv"
url_idx = "https://raw.githubusercontent.com/STU-FEI-OUI/UMINT-UNS_data/main/Python_(CSV)/datafunindx.csv"

df_fun = pd.read_csv(url_fun, header=None)
df_indx = pd.read_csv(url_idx, header=None)

# Konverzia čiarok na bodky pre istotu
x_all_orig = df_fun[0].astype(str).str.replace(',', '.').astype(np.float32).values.reshape(-1, 1)
y_all_orig = df_fun[1].astype(str).str.replace(',', '.').astype(np.float32).values.reshape(-1, 1)
split_labels = df_indx[1].astype(str).str.replace(',', '.').astype(float).astype(int).values.squeeze()

# Rozdelenie dát
train_mask = (split_labels == 1)
test_mask = (split_labels == 2)

x_train, y_train = x_all_orig[train_mask], y_all_orig[train_mask]
x_test, y_test = x_all_orig[test_mask], y_all_orig[test_mask]

# Normalizácia (Z-score)
x_mean, x_std = x_train.mean(), x_train.std() + 1e-8
y_mean, y_std = y_train.mean(), y_train.std() + 1e-8

x_train_s, y_train_s = (x_train - x_mean) / x_std, (y_train - y_mean) / y_std
x_test_s, y_test_s = (x_test - x_mean) / x_std, (y_test - y_mean) / y_std


class RegDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.x)

    def __getitem__(self, idx): return self.x[idx], self.y[idx]


train_loader = DataLoader(RegDataset(x_train_s, y_train_s), batch_size=32, shuffle=True)
# ============================================================
# 3. Model, Loss, Optimizer (АБСОЛЮТНО ТОЧНА МОДЕЛЬ)
# ============================================================
hidden_dim = 128
epochs = 2500     # Даємо більше часу на ювелірну роботу з краями
lr = 0.001        # Зменшуємо крок, щоб вона акуратно дотягнула кінці

# Додали ТРЕТІЙ прихований шар! Тепер у неї достатньо "згинів",
# щоб ідеально лягти на крайні точки.
model = nn.Sequential(
    nn.Linear(1, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim), # <--- Той самий додатковий шар
    nn.Tanh(),
    nn.Linear(hidden_dim, 1)
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# ============================================================
# 4. Tréning
# ============================================================
history_train_loss = []
history_test_loss = []

x_test_tensor = torch.tensor(x_test_s, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test_s, dtype=torch.float32).to(DEVICE)

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x_b), y_b)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(x_b)

    history_train_loss.append(epoch_loss / len(x_train_s))

    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(x_test_tensor), y_test_tensor).item()
        history_test_loss.append(test_loss)

# ============================================================
# 5. Výsledky a Metriky (SSE, MSE, MAE)
# ============================================================
model.eval()
with torch.no_grad():
    y_train_pred_s = model(torch.tensor(x_train_s).to(DEVICE)).cpu().numpy()
    y_test_pred_s = model(x_test_tensor).cpu().numpy()
    y_all_pred_s = model(torch.tensor((x_all_orig - x_mean) / x_std).to(DEVICE)).cpu().numpy()

# Inverzná transformácia do pôvodnej škály
y_train_pred = y_train_pred_s * y_std + y_mean
y_test_pred = y_test_pred_s * y_std + y_mean
y_all_pred = y_all_pred_s * y_std + y_mean


def regression_metrics(y_true, y_pred):
    err = y_true - y_pred
    return float(np.sum(err ** 2)), float(np.mean(err ** 2)), float(np.mean(np.abs(err)))


train_sse, train_mse, train_mae = regression_metrics(y_train, y_train_pred)
test_sse, test_mse, test_mae = regression_metrics(y_test, y_test_pred)

print(f"Train Metriky: SSE={train_sse:.4f}, MSE={train_mse:.4f}, MAE={train_mae:.4f}")
print(f"Test Metriky : SSE={test_sse:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}")

# ============================================================
# 6. Grafy
# ============================================================
plt.figure(figsize=(12, 5))

# Graf 1: Loss vs Epoch
plt.subplot(1, 2, 1)
plt.plot(history_train_loss, label="Train Loss")
plt.plot(history_test_loss, label="Test Loss")
plt.title("Priebeh chyby (Loss vs. Epoch)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (Scaled)")
plt.legend()
plt.grid(True)

# Graf 2: Porovnanie funkcie
plt.subplot(1, 2, 2)
plt.scatter(x_train, y_train, label="Train dáta", alpha=0.6, color='blue', s=10)
plt.scatter(x_test, y_test, label="Test dáta", alpha=0.6, color='red', s=10)

# Zoradenie pre pevnú čiaru výstupu
order = np.argsort(x_all_orig.squeeze())
plt.plot(x_all_orig[order], y_all_pred[order], color='black', linewidth=2.5, label="Výstup MLP")

plt.title("Pôvodné dáta vs. Výstup siete")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()