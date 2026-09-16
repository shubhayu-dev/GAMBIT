"""
Hugging Face PyTorch Training Pipeline with Train/Val Split & Diagnostic Metrics.

Loads ssingh22/chess-evaluations ('evals_large'), splits into train and validation sets,
and evaluates:
  - MAE (pawn error)
  - R^2 score (generalization)
  - Directional Accuracy (who is winning)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from environment import Board
from features import board_to_features
from neural_agent import ChessValueNet, MODEL_PATH

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from datasets import load_dataset
    import numpy as np
    from sklearn.metrics import r2_score, mean_absolute_error
except ImportError:
    print("Please run: pip install torch datasets scikit-learn")
    sys.exit(1)


class ChessHFDataset(Dataset):
    def __init__(self, hf_dataset_slice):
        self.data = hf_dataset_slice

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        board = Board(row["FEN"])
        features = board_to_features(board)

        # Parse Stockfish centipawn/mate evaluation into clipped pawns [-20, 20]
        raw_eval = str(row["Evaluation"])
        if "#" in raw_eval or "M" in raw_eval.upper():
            val = 20.0 if "-" not in raw_eval else -20.0
        else:
            try:
                val = float(raw_eval) / 100.0
                val = max(-20.0, min(20.0, val))
            except ValueError:
                val = 0.0

        return torch.tensor(features, dtype=torch.float32), torch.tensor([val], dtype=torch.float32)


def evaluate_metrics(model, val_loader, device):
    """Computes Loss, MAE, R^2, and Directional Advantage Accuracy on held-out positions."""
    model.eval()
    criterion = nn.MSELoss()
    total_val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_val_loss += loss.item()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    val_loss = total_val_loss / len(val_loader)

    # Directional Accuracy: when eval is non-trivial (|eval| >= 0.5 pawns),
    # does the network predict the correct winning side?
    decisive_mask = np.abs(y_true) >= 0.5
    if np.any(decisive_mask):
        correct_direction = np.sign(y_pred[decisive_mask]) == np.sign(y_true[decisive_mask])
        directional_acc = float(np.mean(correct_direction)) * 100.0
    else:
        directional_acc = 0.0

    return val_loss, mae, r2, directional_acc


def main():
    print("=" * 60)
    print(" GAMBIT: Training NeuralAgent (Train/Val Validation)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware backend: {device.upper()}")

    print("\n[1/4] Streaming ssingh22/chess-evaluations ('evals_large') from HF...")
    full_dataset = load_dataset("ssingh22/chess-evaluations", "evals_large", split="train")

    # Sample a manageable subset (e.g., 50,000 positions for fast CPU training; increase if on GPU)
    max_positions = 50000
    subset = full_dataset.select(range(min(max_positions, len(full_dataset))))

    # 85% Train / 15% Validation split
    split_data = subset.train_test_split(test_size=0.15, seed=42)
    train_ds = ChessHFDataset(split_data["train"])
    val_ds = ChessHFDataset(split_data["test"])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    print(f"  Total positions: {len(subset)}")
    print(f"  Split: {len(train_ds)} train / {len(val_ds)} validation")

    print("\n[2/4] Initializing ChessValueNet...")
    model = ChessValueNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 5
    best_val_loss = float("inf")

    print(f"\n[3/4] Training for {epochs} epochs with per-epoch validation...")
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 25 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f}", end="\r")

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_mae, val_r2, dir_acc = evaluate_metrics(model, val_loader, device)

        print(f"\n  --- Epoch {epoch+1}/{epochs} ({time.time()-t0:.1f}s) ---")
        print(f"      Train Loss (MSE) : {avg_train_loss:.4f}")
        print(f"      Val Loss (MSE)   : {val_loss:.4f}")
        print(f"      Val MAE (pawns)  : {val_mae:.3f}")
        print(f"      Val R^2 Score    : {val_r2:.3f}")
        print(f"      Directional Acc  : {dir_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            MODEL_PATH.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"      -> Checkpoint saved (val_loss improved to {val_loss:.4f})")

    print("\n[4/4] Training complete. Production weights verified on held-out data.")


if __name__ == "__main__":
    main()