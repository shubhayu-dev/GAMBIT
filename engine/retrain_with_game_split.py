"""
retrain_with_game_split.py — audit fix: game-level train/val/test split
(docs/12_audit_report.md section 4, "positions from the same game can
appear across splits" -- now fixed) instead of the original position-level
random split.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_play_data import generate_value_training_data_with_game_ids, game_level_split
from neural_agent import generate_distillation_training_data, train_value_network, save_model
from evaluation import evaluate as classical_evaluate, evaluate_material_only
from environment import Board
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def main():
    print("Generating self-play position pool WITH game_idx tracking...")
    t0 = time.time()
    records = generate_value_training_data_with_game_ids(n_games=250, max_plies=60, sample_every=2, seed=42)
    print(f"  {len(records)} positions across 250 games in {time.time()-t0:.1f}s")

    train_recs, val_recs, test_recs = game_level_split(records, train_frac=0.7, val_frac=0.15, seed=7)
    print(f"  Game-level split: {len(train_recs)} train / {len(val_recs)} val / {len(test_recs)} test positions")

    # Verify zero game-id overlap across splits (the actual bug being fixed)
    train_games = set(r["game_idx"] for r in train_recs)
    val_games = set(r["game_idx"] for r in val_recs)
    test_games = set(r["game_idx"] for r in test_recs)
    assert not (train_games & val_games), "LEAK: train/val share games"
    assert not (train_games & test_games), "LEAK: train/test share games"
    assert not (val_games & test_games), "LEAK: val/test share games"
    print(f"  Verified: {len(train_games)}/{len(val_games)}/{len(test_games)} games in train/val/test, zero overlap")

    # Subsample train to keep labeling cost comparable to the original run
    import random
    rng = random.Random(7)
    train_sample = rng.sample(train_recs, min(1400, len(train_recs)))
    val_sample = rng.sample(val_recs, min(300, len(val_recs)))
    test_sample = rng.sample(test_recs, min(300, len(test_recs)))

    print(f"\nLabeling {len(train_sample)} train / {len(val_sample)} val / {len(test_sample)} test "
          f"positions with MaterialAgent(depth=2) reference evaluation...")
    t0 = time.time()
    X_train, y_train = generate_distillation_training_data(
        [{"fen": r["fen"]} for r in train_sample], reference_depth=2)
    X_val, y_val = generate_distillation_training_data(
        [{"fen": r["fen"]} for r in val_sample], reference_depth=2)
    X_test, y_test = generate_distillation_training_data(
        [{"fen": r["fen"]} for r in test_sample], reference_depth=2)
    print(f"  Labeled in {time.time()-t0:.1f}s")

    print(f"\nTraining MLPRegressor on {len(X_train)} positions (game-level held-out val/test)...")
    t0 = time.time()
    model = train_value_network(X_train, y_train, seed=7)
    print(f"  Trained in {time.time()-t0:.1f}s, n_iter_={model.n_iter_}")

    print("\n=== Fit quality, game-level splits (no leakage) ===")
    fens_test = [r["fen"] for r in test_sample]
    for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        pred = model.predict(X)
        print(f"  {name:5s} (n={len(y):4d}): MAE={mean_absolute_error(y, pred):.3f}  "
              f"RMSE={mean_squared_error(y, pred)**0.5:.3f}  R^2={r2_score(y, pred):.3f}")

    print("\n=== Baselines vs neural, on TEST split (game-level held out) ===")
    y_test_arr = np.array(y_test)
    const_pred = np.full_like(y_test_arr, np.mean(y_train))
    mat_pred = np.array([evaluate_material_only(Board(f)) for f in fens_test])
    classical_pred = np.array([classical_evaluate(Board(f)) for f in fens_test])
    neural_pred = model.predict(X_test)

    for name, pred in [("Constant", const_pred), ("Material-only", mat_pred),
                        ("Classical evaluate()", classical_pred), ("Neural MLP (game-split)", neural_pred)]:
        print(f"  {name:24s}: MAE={mean_absolute_error(y_test_arr, pred):.3f}  "
              f"RMSE={mean_squared_error(y_test_arr, pred)**0.5:.3f}")

    save_model(model, Path(__file__).resolve().parent.parent / "data" / "neural_value_model.pkl")
    print(f"\nModel saved to data/neural_value_model.pkl (Main model updated)")


if __name__ == "__main__":
    main()
