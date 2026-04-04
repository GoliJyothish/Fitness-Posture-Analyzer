import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Configuration
PROCESSED_DATA_DIR = "data/processed/train"
CSV_OUTPUT_DIR = "data/csv_landmarks"
NUM_LANDMARKS = 33
NUM_FEATURES = 4 # x, y, z, visibility

def convert_npy_to_csv(npy_filepath, csv_output_dir):
    """
    Loads an .npy file containing landmark data and converts it into a CSV file.
    The CSV file will have columns for frame index and then x, y, z, visibility
    for each of the 33 landmarks.
    """
    try:
        data = np.load(npy_filepath)
    except Exception as e:
        print(f"Error loading {npy_filepath}: {e}")
        return

    # Ensure data is 3D (num_frames, NUM_LANDMARKS, NUM_FEATURES)
    if data.ndim != 3 or data.shape[1] != NUM_LANDMARKS or data.shape[2] != NUM_FEATURES:
        print(f"Warning: {npy_filepath} has unexpected shape {data.shape}. Skipping.")
        return

    num_frames = data.shape[0]
    
    # Create column names
    columns = ["frame_idx"]
    for i in range(NUM_LANDMARKS):
        columns.extend([f"L{i}_x", f"L{i}_y", f"L{i}_z", f"L{i}_visibility"])
    
    # Flatten the data: (num_frames, 33, 4) -> (num_frames, 33*4)
    flattened_data = data.reshape(num_frames, NUM_LANDMARKS * NUM_FEATURES)
    
    # Add frame index
    df = pd.DataFrame(flattened_data, columns=columns[1:])
    df.insert(0, "frame_idx", range(num_frames))

    # Construct output CSV filepath
    relative_path = os.path.relpath(npy_filepath, PROCESSED_DATA_DIR)
    csv_relative_path = relative_path.replace(".npy", ".csv")
    csv_filepath = os.path.join(csv_output_dir, csv_relative_path)

    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    
    df.to_csv(csv_filepath, index=False)
    # print(f"Converted {npy_filepath} to {csv_filepath}")

def main():
    print(f"Starting conversion from {PROCESSED_DATA_DIR} to {CSV_OUTPUT_DIR}...")
    
    # Ensure output directory exists
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    total_files = 0
    for root, _, files in os.walk(PROCESSED_DATA_DIR):
        total_files += sum(1 for f in files if f.endswith(".npy"))

    if total_files == 0:
        print(f"No .npy files found in {PROCESSED_DATA_DIR}. Exiting.")
        return

    with tqdm(total=total_files, desc="Converting .npy to .csv") as pbar:
        for root, _, files in os.walk(PROCESSED_DATA_DIR):
            for file in files:
                if file.endswith(".npy"):
                    npy_filepath = os.path.join(root, file)
                    convert_npy_to_csv(npy_filepath, CSV_OUTPUT_DIR)
                    pbar.update(1)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
