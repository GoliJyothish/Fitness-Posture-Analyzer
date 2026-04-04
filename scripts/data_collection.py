import cv2
import mediapipe as mp
import numpy as np
import os
import datetime

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def collect_data():
    person_id = input("Enter person ID (e.g., person_1): ")
    exercise_name = input("Enter exercise name (e.g., bicep_curl): ")
    
    output_dir = os.path.join("data", "test_custom", person_id, exercise_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Collecting data for {person_id} - {exercise_name}. Press 'q' to quit.")
    
    cap = cv2.VideoCapture(0)  # Open default camera
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # For webcam input:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        all_landmarks = []
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
            # Convert the BGR image to RGB.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and find pose landmarks.
            results = pose.process(image_rgb)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Extract landmarks
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                all_landmarks.append(landmarks)
                
            cv2.imshow('MediaPipe Pose', image)
            
            frame_count += 1

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Save landmarks
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{exercise_name}_{timestamp}_landmarks.npy")
        np.save(output_file, np.array(all_landmarks))
        print(f"Landmarks saved to {output_file}")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
