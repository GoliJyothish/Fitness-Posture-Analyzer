import cv2
import mediapipe as mp # Ensure this import is present
import numpy as np
import os
import glob

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the pose landmarker model.
MODEL_PATH = "models/pose_landmarker_heavy.task"

def process_video_for_landmarks(video_path, output_dir):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_landmarks = []
    
    # Initialize PoseLandmarker
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_timestamp_ms = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Convert the BGR image to RGB.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Get current timestamp in milliseconds
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Detect pose landmarks for the current frame
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if detection_result.pose_landmarks:
                landmarks = []
                for landmark in detection_result.pose_landmarks[0]: # Assuming single person detection
                    landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                frame_landmarks.append(landmarks)
            else:
                # If no landmarks are detected, append an array of NaNs or zeros
                frame_landmarks.append(np.full((33, 4), np.nan).tolist()) # Assuming 33 landmarks, each with 4 values (x,y,z,visibility)
            
    cap.release()

    if frame_landmarks:
        output_filename = os.path.basename(video_path).replace(".mp4", "_landmarks.npy")
        output_filepath = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_filepath, np.array(frame_landmarks))
        print(f"Saved landmarks to {output_filepath}")
    else:
        print(f"No landmarks found for video {video_path}, skipping save.")

def main():
    raw_data_base_path = os.path.join("data", "raw", "archive")
    processed_train_base_path = os.path.join("data", "processed", "train")

    if not os.path.exists(raw_data_base_path):
        print(f"Raw data archive path not found: {raw_data_base_path}")
        print("Please run setup_project.py first and ensure 'archive' data is present.")
        return

    exercise_folders = [d for d in os.listdir(raw_data_base_path) if os.path.isdir(os.path.join(raw_data_base_path, d))]

    if not exercise_folders:
        print(f"No exercise folders found in {raw_data_base_path}. Exiting.")
        return

    for exercise_name in exercise_folders:
        exercise_raw_path = os.path.join(raw_data_base_path, exercise_name)
        # For now, all processed videos from archive will be considered "correct"
        exercise_processed_output_path = os.path.join(processed_train_base_path, exercise_name.replace(" ", "_"), "correct")
        
        video_files = glob.glob(os.path.join(exercise_raw_path, "*.mp4"))
        
        if not video_files:
            print(f"No video files found for exercise: {exercise_name}, skipping.")
            continue

        for video_file in video_files:
            process_video_for_landmarks(video_file, exercise_processed_output_path)
    
    print("All raw videos processed for landmarks.")

if __name__ == "__main__":
    main()
