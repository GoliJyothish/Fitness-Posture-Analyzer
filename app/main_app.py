import cv2
import mediapipe as mp
import numpy as np
import threading
import pyttsx3
import time
import collections # Import collections for deque
import traceback # Import traceback for error reporting

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult, PoseLandmarker # Add this import


def main_app():
    global latest_frame, feedback_text, rep_count, mode, exercise

    cap = None
    for i in range(5): # Try camera indices from 0 to 4
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

    # Start the pose detection thread
    detection_thread = threading.Thread(target=pose_detection_thread)
    detection_thread.daemon = True # Allow main program to exit even if thread is running
    detection_thread.start()

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
        if ex_choice == '1': exercise = "Bicep Curl"
        elif ex_choice == '2': exercise = "Lat Pulldown/Row"
        elif ex_choice == '3': exercise = "Chest Press"
        elif ex_choice == '4': exercise = "Shoulder Press"
        else: print("Invalid choice, defaulting to Bicep Curl.")
    elif choice == '2':
        mode = "Home"
        print("Select Exercise (Home Mode):")
        print("1. Pushups")
        print("2. Squats")
        print("3. Lunges")
        print("4. Plank")
        ex_choice = input("Enter your choice (1-4): ")
        if ex_choice == '1': exercise = "Pushups"
        elif ex_choice == '2': exercise = "Squats"
        elif ex_choice == '3': exercise = "Lunges"
        elif ex_choice == '4': exercise = "Plank"
        else: print("Invalid choice, defaulting to Pushups.")
    else:
        print("Invalid choice, defaulting to Home Mode - Pushups.")

    print(f"Starting {mode} Mode - {exercise}...")

    print("\nSelect Input Source:")
    print("1. Online Training (Webcam)")
    print("2. Upload Video File")
    input_choice = input("Enter your choice (1 or 2): ")

    if input_choice == '1': # Online Training (Webcam)
        print("Starting Online Training with Webcam...")
        cap = None
        for i in range(5): # Try camera indices from 0 to 4
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

        # Start the pose detection thread
        detection_thread = threading.Thread(target=pose_detection_thread)
        detection_thread.daemon = True # Allow main program to exit even if thread is running
        detection_thread.start()

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # Added a small delay to prevent a tight loop if camera consistently fails
                    time.sleep(0.1) 
                    continue

                # Flip the image horizontally for a selfie-view display.
                image = cv2.flip(image, 1)
                latest_frame = image.copy() # Make a copy for the detection thread

                # Draw the pose annotation on the image.
                if latest_detection_result and latest_detection_result.pose_landmarks:
                    annotated_image = image.copy() # Make a copy to draw on
                    for pose_landmarks in latest_detection_result.pose_landmarks: # Iterate through detected poses
                        vision.drawing_utils.draw_landmarks(
                            image=annotated_image,
                            landmark_list=pose_landmarks,
                        landmark_drawing_spec=vision.drawing_styles.get_default_pose_landmarks_style(),
                        connections=POSE_CONNECTIONS
                    )
                    image = annotated_image # Update image to display annotated version            
                # Display feedback and rep count
                cv2.putText(image, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Exercise: {exercise}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Reps: {rep_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Feedback: {feedback_text}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow('AI Fitness Coach', image)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred during webcam processing: {e}")
            import traceback
            traceback.print_exc()

        print("Exited camera loop.")

    elif input_choice == '2': # Upload Video File
        video_path = input("Enter the path to the video file: ")
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}.")
            return

        print(f"Processing video file: {video_path}")

        # Start the pose detection thread
        detection_thread = threading.Thread(target=pose_detection_thread)
        detection_thread.daemon = True # Allow main program to exit even if thread is running
        detection_thread.start()
        
        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("End of video or failed to read frame.")
                    break # Exit loop at end of video

                # Flip the image horizontally for a selfie-view display. (Assuming user wants this for uploaded video too)
                image = cv2.flip(image, 1)
                latest_frame = image.copy() # Make a copy for the detection thread

                # Draw the pose annotation on the image.
                if latest_detection_result and latest_detection_result.pose_landmarks:
                    annotated_image = image.copy()
                    for pose_landmarks in latest_detection_result.pose_landmarks:
                        vision.drawing_utils.draw_landmarks(
                            image=annotated_image,
                            landmark_list=pose_landmarks,
                            landmark_drawing_spec=vision.drawing_styles.get_default_pose_landmarks_style(),
                            connections=POSE_CONNECTIONS
                        )
                    image = annotated_image
                # Display feedback and rep count            cv2.putText(image, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Exercise: {exercise}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Reps: {rep_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Feedback: {feedback_text}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow('AI Fitness Coach', image)
                
                if cv2.waitKey(25) & 0xFF == ord('q'): # Use a higher delay for video playback
                    print("Exiting video processing: 'q' pressed.")
                    break
        except Exception as e:
            print(f"An error occurred during video processing: {e}")
            import traceback
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