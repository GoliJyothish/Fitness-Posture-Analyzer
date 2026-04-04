
import os
import subprocess
import sys
import shutil # Import shutil for rmtree

VENV_DIR = ".venv"

def create_folder_structure():
    print("Creating folder structure...")
    
    # Base directory for data
    data_base_path = "data"
    
    # /data/raw
    os.makedirs(os.path.join(data_base_path, "raw"), exist_ok=True)
    
    # /data/processed/train with subfolders
    processed_train_path = os.path.join(data_base_path, "processed", "train")
    exercises = [
        "bicep_curl", "lat_pulldown_row", "chest_press", "shoulder_press",
        "pushups", "squats", "lunges", "plank"
    ]
    for exercise in exercises:
        os.makedirs(os.path.join(processed_train_path, exercise, "correct"), exist_ok=True)
        os.makedirs(os.path.join(processed_train_path, exercise, "incorrect"), exist_ok=True)
        
    # /data/test_custom
    test_custom_path = os.path.join(data_base_path, "test_custom")
    for i in range(1, 5):
        os.makedirs(os.path.join(test_custom_path, f"person_{i}"), exist_ok=True)
        
    # /models and /scripts
    os.makedirs("models", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)
    
    print("Folder structure created successfully.")

def move_archive_data():
    print("Checking for existing 'archive' data...")
    if os.path.exists("archive"):
        destination_path = os.path.join("data", "raw", "archive")
        os.makedirs(destination_path, exist_ok=True)
        for item in os.listdir("archive"):
            s = os.path.join("archive", item)
            d = os.path.join(destination_path, item)
            try:
                os.rename(s, d)
                print(f"Moved '{s}' to '{d}'")
            except OSError as e:
                print(f"Error moving '{s}' to '{d}': {e}")
        # Only remove the directory if it's empty
        if not os.listdir("archive"):
            os.rmdir("archive")
        else:
            print(f"Warning: 'archive' directory not empty after move. Remaining items: {os.listdir('archive')}")

        print("Existing 'archive' data moved to 'data/raw/archive'.")
    else:
        print("No 'archive' directory found to move.")

def setup_virtual_env():
    print(f"Setting up virtual environment in {VENV_DIR}...")
    if not os.path.exists(VENV_DIR):
        try:
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
            print("Virtual environment created.")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            sys.exit(1)
    else:
        print("Virtual environment already exists.")

def get_venv_python():
    """Returns the path to the virtual environment's python executable."""
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")

def install_dependencies():
    print("Installing dependencies...")
    
    venv_python = get_venv_python()
    if not os.path.exists(venv_python):
        print(f"Error: Virtual environment python executable not found at {venv_python}. Please ensure venv is set up correctly.")
        sys.exit(1)

    required_packages = [
        "opencv-python",
        "mediapipe",
        "numpy",
        "tensorflow",
        "pyttsx3",
        "kaggle"
    ]
    
    for package in required_packages:
        try:
            subprocess.check_call([venv_python, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            sys.exit(1)
            
    print("All dependencies installed successfully.")

if __name__ == "__main__":
    create_folder_structure()
    move_archive_data()
    setup_virtual_env()
    install_dependencies()
    print("Setup complete!")
