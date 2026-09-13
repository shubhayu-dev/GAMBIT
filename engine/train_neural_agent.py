"""
train_neural_agent.py — builds the distillation training set (self-play
position pool + MaterialAgent reference labels), trains the MLPRegressor,
reports honest train/test fit quality, and saves the model.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_play_data import generate_value_training_data
from neural_agent import generate_distillation_training_data, train_value_network, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    print("Generating self-play position pool (RandomAgent, for position diversity only)...")
    t0 = time.time()
    fens, _unused_outcomes = generate_value_training_data(n_games=250, max_plies=60, sample_every=2, seed=42)
    print(f"  Pool: {len(fens)} positions in {time.time()-t0:.1f}s")

    rng = random.Random(7)
    sample_size = min(2000, len(fens))
    sampled_fens = rng.sample(fens, sample_size)
    positions = [{"fen": f} for f in sampled_fens]

    print(f"Labeling {sample_size} positions with MaterialAgent(depth=2) reference evaluation "
          f"(distillation target — see engine/neural_agent.py docstring)...")
    t0 = time.time()
    X, y = generate_distillation_training_data(positions, reference_depth=2)
    print(f"  Labeled in {time.time()-t0:.1f}s")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    print(f"Training MLPRegressor on {len(X_train)} positions ({len(X_test)} held out for fit-quality check)...")
    t0 = time.time()
    model = train_value_network(X_train, y_train, seed=7)
    print(f"  Trained in {time.time()-t0:.1f}s")

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    print("\nFit quality (predicting MaterialAgent's depth-2 evaluation, in pawns):")
    print(f"  Train MAE: {mean_absolute_error(y_train, train_pred):.3f}  R^2: {r2_score(y_train, train_pred):.3f}")
    print(f"  Test  MAE: {mean_absolute_error(y_test, test_pred):.3f}  R^2: {r2_score(y_test, test_pred):.3f}")
    print("  (This measures how well the net approximates the classical evaluator it was trained "
          "on — not chess strength. A test R^2 well below the train R^2 would flag overfitting.)")

    save_model(model)
    print(f"\nModel saved to {Path(__file__).resolve().parent.parent / 'data' / 'neural_value_model.pkl'}")


if __name__ == "__main__":
    main()
