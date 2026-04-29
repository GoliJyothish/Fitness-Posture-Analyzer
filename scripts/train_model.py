"""
scripts/train_model.py
----------------------
LSTM form classification model training script.

Fixes applied (per analysis MD):
1. Added np.random.seed(42) and tf.random.set_seed(42) for reproducibility
   — previously training was non-deterministic across runs.
2. Added sklearn classification_report at evaluation time
   — provides per-class precision, recall, F1 so accuracy numbers can be
     documented in the README and MODEL_CARD.md.
3. Training history saved to models/training_history.json
   — allows plotting loss/accuracy curves and comparing runs.
"""

import numpy as np
import os
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# FIX (MD §5): Set random seeds at the top of main() for reproducibility.
RANDOM_SEED = 42

# Configuration — import from shared/constants.py if available, else use defaults
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.constants import SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_FEATURES, FORM_MODEL_PATH
except ImportError:
    SEQUENCE_LENGTH = 100
    NUM_LANDMARKS = 33
    NUM_FEATURES = 4
    FORM_MODEL_PATH = "models/form_classification_model.keras"

DATA_DIR = "data/processed/train"
BATCH_SIZE = 32
EPOCHS = 10
HISTORY_SAVE_PATH = "models/training_history.json"


def load_data(data_dir: str):
    """
    Load all .npy landmark sequences from the processed data directory.

    Expected folder structure:
        data/processed/train/
            <exercise_name>/
                correct/    <- .npy files labelled 1
                incorrect/  <- .npy files labelled 0

    Returns:
        Tuple of (sequences array, labels array).
    """
    all_sequences = []
    all_labels = []

    exercise_folders = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    for exercise in tqdm(exercise_folders, desc="Loading Exercises"):
        exercise_path = os.path.join(data_dir, exercise)

        correct_path = os.path.join(exercise_path, "correct")
        if os.path.exists(correct_path):
            for npy_file in os.listdir(correct_path):
                if npy_file.endswith(".npy"):
                    sequence = np.load(os.path.join(correct_path, npy_file))
                    all_sequences.append(sequence)
                    all_labels.append(1)  # 1 = correct form

        incorrect_path = os.path.join(exercise_path, "incorrect")
        if os.path.exists(incorrect_path):
            for npy_file in os.listdir(incorrect_path):
                if npy_file.endswith(".npy"):
                    sequence = np.load(os.path.join(incorrect_path, npy_file))
                    all_sequences.append(sequence)
                    all_labels.append(0)  # 0 = incorrect form

    return np.array(all_sequences, dtype=object), np.array(all_labels)


def preprocess_sequences(
    sequences,
    sequence_length: int,
    num_landmarks: int,
    num_features: int,
) -> np.ndarray:
    """
    Preprocess raw landmark sequences for LSTM input.

    Steps per sequence:
        1. Fill NaN with 0.0
        2. Flatten (N, 33, 4) -> (N, 132)
        3. Pad or truncate to fixed sequence_length

    Returns:
        np.ndarray of shape (n_samples, sequence_length, num_landmarks * num_features)
    """
    processed_sequences = []
    for seq in tqdm(sequences, desc="Preprocessing Sequences"):
        seq = np.nan_to_num(seq, nan=0.0)
        seq_flattened = seq.reshape(seq.shape[0], num_landmarks * num_features)

        if seq_flattened.shape[0] < sequence_length:
            padding = np.zeros(
                (sequence_length - seq_flattened.shape[0], num_landmarks * num_features)
            )
            processed_seq = np.vstack((seq_flattened, padding))
        else:
            processed_seq = seq_flattened[:sequence_length]

        processed_sequences.append(processed_seq)

    return np.array(processed_sequences)


def build_model(input_shape: tuple) -> keras.Model:
    """
    Build the LSTM binary classification model.

    Architecture: Input(100, 132) -> LSTM(64) -> Dropout(0.2)
                  -> LSTM(32) -> Dropout(0.2) -> Dense(1, sigmoid)

    Args:
        input_shape: (sequence_length, num_landmarks * num_features)

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    # FIX (MD §5): Set seeds before anything else so results are reproducible
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    print(f"Random seeds set to {RANDOM_SEED} for reproducibility.")

    print("Loading data...")
    sequences, labels = load_data(DATA_DIR)
    print(f"Loaded {len(sequences)} sequences with {len(labels)} labels.")

    if len(sequences) == 0:
        print("No data found. Please ensure data/processed/train contains .npy files.")
        print("See DATA.md for instructions on downloading and processing training data.")
        return

    print("Preprocessing data...")
    processed_sequences = preprocess_sequences(
        sequences, SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_FEATURES
    )

    # Split with fixed random_state for reproducibility
    X_train, X_val, y_train, y_val = train_test_split(
        processed_sequences, labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    print("Building model...")
    input_shape = (SEQUENCE_LENGTH, NUM_LANDMARKS * NUM_FEATURES)
    model = build_model(input_shape)
    model.summary()

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
    )

    # FIX (MD §5): Save training history for plotting and run comparison
    history_data = {
        "accuracy":     history.history.get("accuracy", []),
        "val_accuracy": history.history.get("val_accuracy", []),
        "loss":         history.history.get("loss", []),
        "val_loss":     history.history.get("val_loss", []),
        "epochs":       EPOCHS,
        "random_seed":  RANDOM_SEED,
    }
    os.makedirs(os.path.dirname(HISTORY_SAVE_PATH), exist_ok=True)
    with open(HISTORY_SAVE_PATH, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"Training history saved to {HISTORY_SAVE_PATH}")

    # Evaluate
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Loss:     {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # FIX (MD §5): Full classification report — put these numbers in README and MODEL_CARD.md
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["Incorrect", "Correct"]))

    # Save model
    os.makedirs(os.path.dirname(FORM_MODEL_PATH), exist_ok=True)
    model.save(FORM_MODEL_PATH)
    print(f"Model saved to {FORM_MODEL_PATH}")


if __name__ == "__main__":
    main()
