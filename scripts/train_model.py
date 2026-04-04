import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm # For progress bar

# Configuration
DATA_DIR = "data/processed/train"
SEQUENCE_LENGTH = 100 # Example fixed sequence length. Adjust based on typical video lengths.
NUM_LANDMARKS = 33
NUM_FEATURES = 4 # x, y, z, visibility
BATCH_SIZE = 32
EPOCHS = 10

def load_data(data_dir):
    all_sequences = []
    all_labels = []
    
    # Get all exercise folders
    exercise_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for exercise in tqdm(exercise_folders, desc="Loading Exercises"):
        exercise_path = os.path.join(data_dir, exercise)
        
        # Load 'correct' samples
        correct_path = os.path.join(exercise_path, "correct")
        if os.path.exists(correct_path):
            for npy_file in os.listdir(correct_path):
                if npy_file.endswith(".npy"):
                    sequence = np.load(os.path.join(correct_path, npy_file))
                    all_sequences.append(sequence)
                    all_labels.append(1) # Label 1 for correct form

        # Load 'incorrect' samples
        incorrect_path = os.path.join(exercise_path, "incorrect")
        if os.path.exists(incorrect_path):
            for npy_file in os.listdir(incorrect_path):
                if npy_file.endswith(".npy"):
                    sequence = np.load(os.path.join(incorrect_path, npy_file))
                    all_sequences.append(sequence)
                    all_labels.append(0) # Label 0 for incorrect form
                    
    return np.array(all_sequences, dtype=object), np.array(all_labels)

def preprocess_sequences(sequences, sequence_length, num_landmarks, num_features):
    processed_sequences = []
    for seq in tqdm(sequences, desc="Preprocessing Sequences"):
        # Handle NaN values (e.g., fill with 0 or last known value)
        # For simplicity, filling NaNs with 0. A more sophisticated approach might interpolate.
        seq = np.nan_to_num(seq, nan=0.0) 
        
        # Flatten the landmarks for a single timestep (33 * 4 features)
        # This transforms (N_frames, 33, 4) to (N_frames, 33*4)
        seq_flattened = seq.reshape(seq.shape[0], num_landmarks * num_features)

        # Pad or truncate sequences to a fixed length
        if seq_flattened.shape[0] < sequence_length:
            # Pad with zeros
            padding = np.zeros((sequence_length - seq_flattened.shape[0], num_landmarks * num_features))
            processed_seq = np.vstack((seq_flattened, padding))
        else:
            # Truncate
            processed_seq = seq_flattened[:sequence_length]
        
        processed_sequences.append(processed_seq)
        
    return np.array(processed_sequences)

def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid') # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Loading data...")
    sequences, labels = load_data(DATA_DIR)
    print(f"Loaded {len(sequences)} sequences with {len(labels)} labels.")
    
    if len(sequences) == 0:
        print("No data found for training. Please ensure data/processed/train contains .npy files.")
        return

    print("Preprocessing data...")
    processed_sequences = preprocess_sequences(sequences, SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_FEATURES)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(processed_sequences, labels, test_size=0.2, random_state=42)
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Build and train model
    print("Building and training model...")
    input_shape = (SEQUENCE_LENGTH, NUM_LANDMARKS * NUM_FEATURES)
    model = build_model(input_shape)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    # Evaluate model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Save model
    model_save_path = "models/form_classification_model.keras" # Added .keras extension
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
