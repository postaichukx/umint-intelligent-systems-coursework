# ============================================================
# 6 - Klasifikácia medicínskych dát (CTG) pomocou MLP v PyTorch
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ============================================================
# 1. Seed a device
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# ============================================================
# 2. Načítanie dát
# ============================================================

DATA_URL = "https://raw.githubusercontent.com/STU-FEI-OUI/UMINT-UNS_data/main/Python_(CSV)/CTGdata.csv"
FILE_NAME = "CTGdata.csv"

if os.path.exists(FILE_NAME):
    df = pd.read_csv(FILE_NAME)
else:
    df = pd.read_csv(DATA_URL)

# Prvých 25 stĺpcov sú vstupy, posledný je trieda
X_all_raw = df.iloc[:, :25].values.astype(np.float32)
y_raw = df.iloc[:, -1].values.astype(np.int64)

num_classes = len(np.unique(y_raw))

# PyTorch CrossEntropyLoss očakáva triedy od 0, takže: 1,2,3 -> 0,1,2
y_all = y_raw - 1
idx_to_class = {0: "Normálny (1)", 1: "Podozrivý (2)", 2: "Patologický (3)"}


# ============================================================
# 3. Dataset trieda
# ============================================================

class CTGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 4. Model MLP (dynamický pre rôzne štruktúry z Tabuľky 1)
# ============================================================

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super().__init__()
        layers = []
        in_f = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_f, h))
            layers.append(nn.ReLU())
            in_f = h
        layers.append(nn.Linear(in_f, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 5. Train a eval slučky
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_count += len(X_batch)

    return total_loss / total_count, total_correct / total_count


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(X_batch)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_count += len(X_batch)

    return total_loss / total_count, total_correct / total_count


# ============================================================
# 6. Hlavný experiment: Porovnanie 3 štruktúr (M1, M2, M3)
# ============================================================

structures = {
    "M1": [10],  # 1 skrytá vrstva (10 neurónov)
    "M2": [20],  # 1 skrytá vrstva (20 neurónov)
    "M3": [20, 10]  # 2 skryté vrstvy (20 a 10 neurónov)
}

epochs = 300
batch_size = 32
lr = 0.001
runs = 5
patience = 20  # Early Stopping

log_dir_base = "runs/6_ctg_classification"

all_results_summary = []
best_global_acc = 0
best_global_model = None
best_global_name = ""
best_test_loader = None
best_scaler_stats = None

print("Začíname trénovanie modelov. Prosím čakajte...\n")

for model_name, layers in structures.items():
    print(f"{'=' * 60}")
    print(f"Trénovanie štruktúry: {model_name} (Vrstvy: {layers})")

    test_accs, test_losses = [], []
    run_results = []

    best_local_acc = 0
    best_local_history = None

    for run in range(runs):
        # Rozdelenie: 60% train, 20% val, 20% test
        X_train_raw, X_temp, y_train, y_temp = train_test_split(X_all_raw, y_all, test_size=0.4, random_state=run)
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=run)

        # Z-score Normalizácia
        x_mean = X_train_raw.mean(axis=0, keepdims=True)
        x_std = X_train_raw.std(axis=0, keepdims=True) + 1e-8

        X_train = (X_train_raw - x_mean) / x_std
        X_val = (X_val_raw - x_mean) / x_std
        X_test = (X_test_raw - x_mean) / x_std

        train_loader = DataLoader(CTGDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(CTGDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(CTGDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        model = DynamicMLP(input_dim=25, hidden_layers=layers, output_dim=num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        writer = SummaryWriter(log_dir=f"{log_dir_base}/{model_name}_run{run + 1}")

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(1, epochs + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            v_loss, v_acc = evaluate(model, val_loader, criterion, DEVICE)

            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)

            writer.add_scalar("loss/train", t_loss, epoch)
            writer.add_scalar("loss/val", v_loss, epoch)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                import copy

                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        writer.close()

        # Obnovíme najlepšie váhy pre tento beh
        model.load_state_dict(best_model_weights)

        final_train_loss, final_train_acc = evaluate(model, train_loader, criterion, DEVICE)
        final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, DEVICE)

        test_accs.append(final_test_acc * 100)
        test_losses.append(final_test_loss)

        run_results.append({
            "Beh": run + 1,
            "Train loss": round(final_train_loss, 3),
            "Test loss": round(final_test_loss, 3),
            "Train acc [%]": round(final_train_acc * 100, 1),
            "Test acc [%]": round(final_test_acc * 100, 1)
        })

        if final_test_acc > best_local_acc:
            best_local_acc = final_test_acc
            best_local_history = history

        if final_test_acc > best_global_acc:
            best_global_acc = final_test_acc
            best_global_model = model
            best_global_name = model_name
            best_test_loader = test_loader
            best_scaler_stats = (x_mean, x_std)

    print(f"\nTabuľka 2: Výsledky 5 behov pre štruktúru {model_name}")
    print(pd.DataFrame(run_results).to_string(index=False))

    # Príprava zhrnutia do Tabuľky 3
    all_results_summary.append({
        "Model": model_name,
        "Min test acc [%]": round(np.min(test_accs), 1),
        "Max test acc [%]": round(np.max(test_accs), 1),
        "Priemer test acc [%]": round(np.mean(test_accs), 1),
        "Priemer test loss": round(np.mean(test_losses), 3)
    })

# ============================================================
# 7. Finálne porovnanie (Tabuľka 3)
# ============================================================

print(f"\n{'=' * 60}")
print("Tabuľka 3: Súhrnné porovnanie porovnávaných štruktúr")
print(pd.DataFrame(all_results_summary).to_string(index=False))

# ============================================================
# 8. Confusion matrix pre najlepšiu sieť
# ============================================================

print(f"\n{'=' * 60}")
print(f"Analýza najlepšej siete: {best_global_name} (Test Acc: {best_global_acc * 100:.1f}%)")

best_global_model.eval()
y_true_all, y_pred_all = [], []

with torch.no_grad():
    for X_batch, y_batch in best_test_loader:
        X_batch = X_batch.to(DEVICE)
        logits = best_global_model(X_batch)
        y_pred = logits.argmax(dim=1).cpu().numpy()

        y_true_all.extend(y_batch.numpy())
        y_pred_all.extend(y_pred)

cm = confusion_matrix(y_true_all, y_pred_all)

# Vykreslenie CM
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix - {best_global_name}")
plt.xlabel("Predikovaná trieda")
plt.ylabel("Skutočná trieda")
plt.xticks(range(num_classes), [1, 2, 3])
plt.yticks(range(num_classes), [1, 2, 3])

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="black" if cm[i, j] < cm.max() / 2 else "white")

plt.colorbar()
plt.tight_layout()
plt.show()

# ============================================================
# 9. Senzitivita a Špecificita
# ============================================================

print("\nMetriky pre jednotlivé triedy:")
for i in range(num_classes):
    TP = cm[i, i]
    FN = np.sum(cm[i, :]) - TP
    FP = np.sum(cm[:, i]) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print(f"{idx_to_class[i]}: Senzitivita = {sensitivity:.3f}, Špecificita = {specificity:.3f}")

# ============================================================
# Koniec
# ============================================================