import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the AI Fitness Coach App")
    parser.add_argument("--gui", action="store_true", help="Run the GUI version (default)")
    parser.add_argument("--cli", action="store_true", help="Run the CLI version")
    
    args = parser.parse_args()

    if args.cli:
        print("Starting CLI version...")
        # Note: app/main_app.py seems to have missing dependencies/logic in the provided snippet
        # but we run it from root so it can find 'models/' and 'data/' if it were fixed.
        subprocess.run([sys.executable, "app/main_app.py"])
    else:
        print("Starting GUI version...")
        subprocess.run([sys.executable, "app/main_app_gui.py"])

if __name__ == "__main__":
    main()
