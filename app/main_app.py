import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import collections
import traceback
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult, PoseLandmarker

MODEL_PATH = "models/pose_landmarker_heavy.task"

# A custom class to mimic MediaPipe's Connection object
class Connection:
    def __init__(self, start, end):
        self.start = start
        self.end = end

# Hardcoded POSE_CONNECTIONS for drawing
POSE_CONNECTIONS_TUPLES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (23, 24), (24, 26), (26, 28), (28, 30), (28, 32),
    (23, 25), (25, 27), (27, 29), (27, 31)
]
POSE_CONNECTIONS = [Connection(start, end) for start, end in POSE_CONNECTIONS_TUPLES]

# Global variables (used by pose_detection_thread)
latest_frame = None
feedback_text = "N/A"
rep_count = 0
mode = "Home"
exercise = "Pushups"
latest_detection_result = None


def _init_pose_landmarker():
    """Create a synchronous, VIDEO-mode pose landmarker.

    FIX (bug): this file previously had NO landmarker initialized at all,
    and pose_detection_thread() was an empty placeholder (`pass`) that
    never set latest_detection_result. Skeleton overlay and rep counting
    were completely non-functional in this CLI entry point. VIDEO mode is
    used (rather than a background LIVE_STREAM thread) so detection is
    always synchronous with the frame currently being displayed.
    """
    BaseOptions = python.BaseOptions
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
    )
    return PoseLandmarker.create_from_options(options)


def main_app():
    global latest_frame, feedback_text, rep_count, mode, exercise

    print("--- AI Fitness Coach ---")
    print("Select Mode:")
    print("1. Gym Mode")
    print("2. Home Mode")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        mode = "Gym"
        print("Select Exercise (Gym Mode):")
        print("1. Bicep Curl")
        print("2. Lat Pulldown/Row")
        print("3. Chest Press")
        print("4. Shoulder Press")
        ex_choice = input("Enter your choice (1-4): ")
        if ex_choice == '1':
            exercise = "Bicep Curl"
        elif ex_choice == '2':
            exercise = "Lat Pulldown/Row"
        elif ex_choice == '3':
            exercise = "Chest Press"
        elif ex_choice == '4':
            exercise = "Shoulder Press"
        else:
            print("Invalid choice, defaulting to Bicep Curl.")
    elif choice == '2':
        mode = "Home"
        print("Select Exercise (Home Mode):")
        print("1. Pushups")
        print("2. Squats")
        print("3. Lunges")
        print("4. Plank")
        ex_choice = input("Enter your choice (1-4): ")
        if ex_choice == '1':
            exercise = "Pushups"
        elif ex_choice == '2':
            exercise = "Squats"
        elif ex_choice == '3':
            exercise = "Lunges"
        elif ex_choice == '4':
            exercise = "Plank"
        else:
            print("Invalid choice, defaulting to Pushups.")
    else:
        print("Invalid choice, defaulting to Home Mode - Pushups.")

    print(f"Starting {mode} Mode - {exercise}...")

    print("\nSelect Input Source:")
    print("1. Online Training (Webcam)")
    print("2. Upload Video File")
    input_choice = input("Enter your choice (1 or 2): ")

    if input_choice == '1':
        print("Starting Online Training with Webcam...")
        cap = None
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Successfully opened camera with index {i}.")
                break
            else:
                print(f"Could not open camera with index {i}. Trying next...")

        if not cap or not cap.isOpened():
            print("Error: Could not open any video stream.")
            print("Please ensure:")
            print("1. A webcam is connected and recognized by your system.")
            print("2. No other application is currently using the webcam.")
            print("3. Your operating system grants permission for this application to access the webcam.")
            print("You might also try restarting your computer or checking device drivers.")
            return

        try:
            landmarker = _init_pose_landmarker()
        except Exception as e:
            print(f"Error initializing PoseLandmarker: {e}")
            traceback.print_exc()
            cap.release()
            return
        timestamp_ms = 0

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    time.sleep(0.1)
                    continue

                image = cv2.flip(image, 1)
                latest_frame = image.copy()

                # FIX: run detection synchronously on this exact frame (was
                # previously never run at all — see _init_pose_landmarker).
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33
                try:
                    latest_detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as e:
                    print(f"Pose detection error: {e}")

                # Draw skeleton using OpenCV (manual loop — avoids vision.drawing_utils API issues)
                if latest_detection_result and latest_detection_result.pose_landmarks:
                    for pose_landmarks in latest_detection_result.pose_landmarks:
                        h, w = image.shape[:2]
                        for lm in pose_landmarks:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)
                        for conn in POSE_CONNECTIONS:
                            lm_s = pose_landmarks[conn.start]
                            lm_e = pose_landmarks[conn.end]
                            cv2.line(image,
                                     (int(lm_s.x * w), int(lm_s.y * h)),
                                     (int(lm_e.x * w), int(lm_e.y * h)),
                                     (200, 200, 0), 2)

                cv2.putText(image, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Exercise: {exercise}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Reps: {rep_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Feedback: {feedback_text}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow('AI Fitness Coach', image)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred during webcam processing: {e}")
            traceback.print_exc()

        print("Exited camera loop.")

    elif input_choice == '2':
        video_path = input("Enter the path to the video file: ")
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}.")
            return

        print(f"Processing video file: {video_path}")

        try:
            landmarker = _init_pose_landmarker()
        except Exception as e:
            print(f"Error initializing PoseLandmarker: {e}")
            traceback.print_exc()
            cap.release()
            return
        timestamp_ms = 0

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("End of video or failed to read frame.")
                    break

                image = cv2.flip(image, 1)
                latest_frame = image.copy()

                # FIX: run detection synchronously on this exact frame (was
                # previously never run at all — see _init_pose_landmarker).
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33
                try:
                    latest_detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as e:
                    print(f"Pose detection error: {e}")

                # Draw skeleton using OpenCV (manual loop)
                if latest_detection_result and latest_detection_result.pose_landmarks:
                    for pose_landmarks in latest_detection_result.pose_landmarks:
                        h, w = image.shape[:2]
                        for lm in pose_landmarks:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)
                        for conn in POSE_CONNECTIONS:
                            lm_s = pose_landmarks[conn.start]
                            lm_e = pose_landmarks[conn.end]
                            cv2.line(image,
                                     (int(lm_s.x * w), int(lm_s.y * h)),
                                     (int(lm_e.x * w), int(lm_e.y * h)),
                                     (200, 200, 0), 2)

                cv2.putText(image, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Exercise: {exercise}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Reps: {rep_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Feedback: {feedback_text}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow('AI Fitness Coach', image)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    print("Exiting video processing: 'q' pressed.")
                    break
        except Exception as e:
            print(f"An error occurred during video processing: {e}")
            traceback.print_exc()

        print("Finished video processing.")
    else:
        print("Invalid input source choice. Exiting.")
        return

    cap.release()
    cv2.destroyAllWindows()
    print("Application stopped.")


if __name__ == "__main__":
    main_app()
