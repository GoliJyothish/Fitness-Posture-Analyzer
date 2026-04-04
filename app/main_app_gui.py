import customtkinter as ctk
from tkinter import filedialog
import threading # Import threading for managing threads

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import collections # Import collections for deque
import traceback # Import traceback for error reporting

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult # Only import what's directly needed here
from PIL import Image, ImageTk # For displaying video frames in tkinter

# Define available exercises by mode
EXERCISES_BY_MODE = {
    "Gym": ["Bicep Curl", "Lat Pulldown/Row", "Chest Press", "Shoulder Press"],
    "Home": ["Pushups", "Squats", "Lunges", "Plank"]
}

# A custom class to mimic MediaPipe's Connection object,
# as POSE_CONNECTIONS is not directly available or in the expected format in mediapipe.tasks
class Connection:
    def __init__(self, start, end):
        self.start = start
        self.end = end

# Hardcoded POSE_CONNECTIONS as they are not directly available in mediapipe.tasks.python.vision
# These connections are based on MediaPipe's standard pose model.
# Converted to Connection objects for compatibility with drawing_utils.draw_landmarks
POSE_CONNECTIONS_TUPLES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Face and head
    (9, 10), # Shoulders (not directly connected in many pose models, but useful)
    (11, 12), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), # Right arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), # Left arm
    (23, 24), (24, 26), (26, 28), (28, 30), (28, 32), # Right leg
    (23, 25), (25, 27), (27, 29), (27, 31) # Left leg
]
POSE_CONNECTIONS = [Connection(start, end) for start, end in POSE_CONNECTIONS_TUPLES]

def calculate_angle(a, b, c):
    a = np.array(a) # First point (shoulder)
    b = np.array(b) # Mid point (elbow)
    c = np.array(c) # End point (wrist)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))

    if angle > 180.0:
        angle = 360 - angle
    return angle

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Model Configuration (must match train_model.py)
SEQUENCE_LENGTH = 100 
NUM_LANDMARKS = 33
NUM_FEATURES = 4 # x, y, z, visibility
MODEL_SAVE_PATH = "models/form_classification_model.keras"

# Path to the pose landmarker model.
MODEL_PATH = "models/pose_landmarker_heavy.task"

# Bicep Curl Repetition Counting Configuration
BICEP_CURL_START_ANGLE = 150  # More forgiving (was 160)
BICEP_CURL_END_ANGLE = 50    # More forgiving (was 45)

# Squat Repetition Counting Configuration
SQUAT_START_ANGLE = 150      # More forgiving (was 160)
SQUAT_END_ANGLE = 100        # More forgiving (was 90)

# Pushup Repetition Counting Configuration
PUSHUP_START_ANGLE = 150     # More forgiving (was 160)
PUSHUP_END_ANGLE = 100       # More forgiving (was 90)

# Lat Pulldown/Row Configuration
LAT_PULLDOWN_START_ANGLE = 150
LAT_PULLDOWN_END_ANGLE = 100

# Chest Press Configuration
CHEST_PRESS_START_ANGLE = 150
CHEST_PRESS_END_ANGLE = 100

# Shoulder Press Configuration
SHOULDER_PRESS_START_ANGLE = 150
SHOULDER_PRESS_END_ANGLE = 100

# Plank Configuration
PLANK_THRESHOLD_ANGLE = 160



class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Fitness Coach")
        self.geometry("800x600")

        # Load the trained Keras model
        try:
            self.form_classification_model = keras.models.load_model(MODEL_SAVE_PATH)
            print(f"Successfully loaded form classification model from {MODEL_SAVE_PATH}")
        except Exception as e:
            print(f"Error loading form classification model from {MODEL_SAVE_PATH}: {e}")
            self.form_classification_model = None # Set to None if loading fails

        # Initialize Text-to-Speech engine (moved from global)
        self.engine = pyttsx3.init()

        ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.frames = {}

        # Define placeholder classes for different screens
        class StartScreen(ctk.CTkFrame):
            def __init__(self, master, controller):
                super().__init__(master)
                self.controller = controller
                label = ctk.CTkLabel(self, text="Welcome to AI Fitness Coach!")
                label.pack(pady=10, padx=10)
                # Add a button to navigate to Mode Selection
                start_button = ctk.CTkButton(self, text="Start", command=lambda: self.controller._show_frame_controller("ModeSelectionScreen"))
                start_button.pack(pady=10)

        class ModeSelectionScreen(ctk.CTkFrame):
            def __init__(self, master, controller):
                super().__init__(master)
                self.controller = controller
                label = ctk.CTkLabel(self, text="Select Mode")
                label.pack(pady=10, padx=10)
                # Mode selection buttons
                gym_button = ctk.CTkButton(self, text="Gym Mode", command=lambda: self.controller._show_frame_controller("ExerciseSelectionScreen", mode="Gym"))
                gym_button.pack(pady=5)
                home_button = ctk.CTkButton(self, text="Home Mode", command=lambda: self.controller._show_frame_controller("ExerciseSelectionScreen", mode="Home"))
                home_button.pack(pady=5)
                back_button = ctk.CTkButton(self, text="Back", command=lambda: self.controller._show_frame_controller("StartScreen"))
                back_button.pack(pady=10, side="bottom")

        class ExerciseSelectionScreen(ctk.CTkFrame):
            def __init__(self, master, controller):
                super().__init__(master)
                self.controller = controller
                self.selected_mode = None
                self.buttons = [] # To keep track of exercise buttons for dynamic clearing
                self.mode_exercise_label = None # To keep track of the dynamic label

                label = ctk.CTkLabel(self, text="Select Exercise")
                label.pack(pady=10, padx=10)

                self.exercise_frame = ctk.CTkFrame(self) # Frame to hold dynamic exercise buttons
                self.exercise_frame.pack(pady=10, padx=10, fill="both", expand=True)

                back_button = ctk.CTkButton(self, text="Back", command=lambda: self.controller._show_frame_controller("ModeSelectionScreen"))
                back_button.pack(pady=10, side="bottom")

            def update_content(self, mode):
                # Clear previous exercise buttons
                for button in self.buttons:
                    button.destroy()
                self.buttons.clear()

                self.selected_mode = mode
                
                # Clear previous exercise-specific label
                if self.mode_exercise_label and self.mode_exercise_label.winfo_exists():
                    self.mode_exercise_label.destroy()

                self.mode_exercise_label = ctk.CTkLabel(self, text=f"Select Exercise for {self.selected_mode} Mode")
                self.mode_exercise_label.pack(pady=10, padx=10)

                exercises = EXERCISES_BY_MODE.get(self.selected_mode, [])
                if not exercises:
                    no_exercises_label = ctk.CTkLabel(self.exercise_frame, text="No exercises defined for this mode.")
                    no_exercises_label.pack(pady=5, padx=5)
                    self.buttons.append(no_exercises_label)
                    return

                for exercise in exercises:
                    exercise_button = ctk.CTkButton(self.exercise_frame, text=exercise, command=lambda e=exercise: self.controller._show_frame_controller("InputSourceScreen", mode=self.selected_mode, exercise=e))
                    exercise_button.pack(pady=5, padx=5)
                    self.buttons.append(exercise_button)

        class InputSourceScreen(ctk.CTkFrame):
            def __init__(self, master, controller):
                super().__init__(master)
                self.controller = controller
                self.selected_mode = None
                self.selected_exercise = None
                self.dynamic_label = None # To keep track of the dynamic label

                label = ctk.CTkLabel(self, text="Select Input Source")
                label.pack(pady=10, padx=10)

                webcam_button = ctk.CTkButton(self, text="Online Training (Webcam)", command=lambda: self.controller._show_frame_controller("LiveSessionScreen", mode=self.selected_mode, exercise=self.selected_exercise, input_source="webcam"))
                webcam_button.pack(pady=5)
                upload_button = ctk.CTkButton(self, text="Upload Video File", command=lambda: self._select_video_file())
                upload_button.pack(pady=5)
                
                back_button = ctk.CTkButton(self, text="Back", command=lambda: self.controller._show_frame_controller("ExerciseSelectionScreen", mode=self.selected_mode))
                back_button.pack(pady=10, side="bottom")

            def update_content(self, mode, exercise):
                self.selected_mode = mode
                self.selected_exercise = exercise
                # Clear previous dynamic label
                if self.dynamic_label and self.dynamic_label.winfo_exists():
                    self.dynamic_label.destroy()
                
                self.dynamic_label = ctk.CTkLabel(self, text=f"Input for {self.selected_mode} - {self.selected_exercise}")
                self.dynamic_label.pack(pady=10, padx=10)

            def _select_video_file(self):
                video_path = filedialog.askopenfilename(
                    title="Select Video File",
                    filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
                )
                if video_path:
                    print(f"Selected video file: {video_path}")
                    self.controller._show_frame_controller("LiveSessionScreen", mode=self.selected_mode, exercise=self.selected_exercise, input_source="video_file", video_path=video_path)

        class LiveSessionScreen(ctk.CTkFrame):
            def __init__(self, master, controller):
                super().__init__(master)
                self.controller = controller
                self.selected_mode = None
                self.selected_exercise = None
                self.input_source = None
                self.video_path = None

                self.video_cap = None
                self.pose_landmarker = None # Renamed from pose_detector
                self.running_thread = None
                self.stop_event = threading.Event()
                self.frame_update_id = None # To store after() ID

                # Instance variables for pose detection data
                self.latest_frame_rgb = None # Store RGB frame for MediaPipe
                self.latest_detection_result = None
                self.feedback_text = ""
                self.rep_count_left = 0
                self.rep_count_right = 0
                self.rep_count = 0
                self.is_extended_left = False
                self.is_flexed_left = False
                self.is_extended_right = False
                self.is_flexed_right = False
                self.is_extended = False
                self.is_flexed = False
                self.feedback_text_rep = ""
                self.landmark_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
                self.inference_counter = 0
                self.angle_history_left = collections.deque(maxlen=5)
                self.angle_history_right = collections.deque(maxlen=5)

                # UI Elements
                self.video_display_frame = ctk.CTkFrame(self)
                self.video_display_frame.pack(pady=10, padx=10, fill="both", expand=True)
                self.video_label = ctk.CTkLabel(self.video_display_frame, text="")
                self.video_label.pack(fill="both", expand=True)

                self.feedback_label = ctk.CTkLabel(self, text="Feedback: ")
                self.feedback_label.pack(pady=5, padx=10)

                self.reps_label = ctk.CTkLabel(self, text="L Reps: 0 | R Reps: 0")
                self.reps_label.pack(pady=5, padx=10)

                back_button = ctk.CTkButton(self, text="End Session", command=self._end_session)
                back_button.pack(pady=10, side="bottom")

            def update_content(self, mode, exercise, input_source, video_path=None):
                # Ensure any previous session is stopped
                self._end_session()

                self.selected_mode = mode
                self.selected_exercise = exercise
                self.input_source = input_source
                self.video_path = video_path

                self.feedback_label.configure(text=f"Feedback: Preparing...")
                self.reps_label.configure(text="L Reps: 0 | R Reps: 0")
                self.rep_count_left = 0
                self.rep_count_right = 0
                self.rep_count = 0
                self.feedback_text_rep = "" # Reset feedback text
                self.landmark_buffer.clear() # Clear buffer for new session
                self.angle_history_left.clear()
                self.angle_history_right.clear()
                self.is_extended_left = False
                self.is_flexed_left = False
                self.is_extended_right = False
                self.is_flexed_right = False
                self.is_extended = False
                self.is_flexed = False
                
                # Plank variables
                self.plank_start_time = None
                self.plank_accumulated_time = 0
                
                # Start the video stream and pose detection
                self._start_session()

            def _start_session(self):
                self.stop_event.clear() # Clear stop event for new session

                if self.input_source == "webcam":
                    self.video_cap = cv2.VideoCapture(0) # Simple webcam index
                    self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                elif self.input_source == "video_file" and self.video_path:
                    self.video_cap = cv2.VideoCapture(self.video_path)
                else:
                    print("Error: Invalid input source or video path.")
                    self._end_session()
                    return

                if not self.video_cap.isOpened():
                    print("Error: Could not open video source.")
                    self._end_session()
                    return

                # Initialize MediaPipe PoseLandmarker
                BaseOptions = mp.tasks.BaseOptions
                PoseLandmarker = mp.tasks.vision.PoseLandmarker
                PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                def result_callback(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
                    self.latest_detection_result = result

                    if result.pose_landmarks:
                        landmarks = []
                        for landmark in result.pose_landmarks[0]: # Assuming single person detection
                            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                        self.landmark_buffer.append(landmarks)

                        # Extract landmarks and visibility
                        lm = result.pose_landmarks[0]
                        v_l_sh, v_l_el, v_l_wr = lm[11].visibility, lm[13].visibility, lm[15].visibility
                        v_r_sh, v_r_el, v_r_wr = lm[12].visibility, lm[14].visibility, lm[16].visibility
                        v_l_hi, v_l_kn, v_l_an = lm[23].visibility, lm[25].visibility, lm[27].visibility
                        v_r_hi, v_r_kn, v_r_an = lm[24].visibility, lm[26].visibility, lm[28].visibility

                        l_sh, l_el, l_wr = [lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]
                        r_sh, r_el, r_wr = [lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]
                        l_hi, l_kn, l_an = [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
                        r_hi, r_kn, r_an = [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]

                        # Visibility threshold
                        VIS_THRESH = 0.5

                        # Calculate raw angles for both sides
                        raw_l_arm = calculate_angle(l_sh, l_el, l_wr)
                        raw_r_arm = calculate_angle(r_sh, r_el, r_wr)
                        raw_l_leg = calculate_angle(l_hi, l_kn, l_an)
                        raw_r_leg = calculate_angle(r_hi, r_kn, r_an)
                        raw_l_body = calculate_angle(l_sh, l_hi, l_an)
                        raw_r_body = calculate_angle(r_sh, r_hi, r_an)

                        # Mapping, Smoothing and Visibility checking
                        if self.selected_exercise in ["Bicep Curl", "Pushups", "Lat Pulldown/Row", "Chest Press", "Shoulder Press"]:
                            self.angle_history_left.append(raw_l_arm)
                            self.angle_history_right.append(raw_r_arm)
                            vis_l = (v_l_sh + v_l_el + v_l_wr) / 3
                            vis_r = (v_r_sh + v_r_el + v_r_wr) / 3
                        elif self.selected_exercise in ["Squats", "Lunges"]:
                            self.angle_history_left.append(raw_l_leg)
                            self.angle_history_right.append(raw_r_leg)
                            vis_l = (v_l_hi + v_l_kn + v_l_an) / 3
                            vis_r = (v_r_hi + v_r_kn + v_r_an) / 3
                        else: # Plank
                            self.angle_history_left.append(raw_l_body)
                            self.angle_history_right.append(raw_r_body)
                            vis_l = (v_l_sh + v_l_hi + v_l_an) / 3
                            vis_r = (v_r_sh + v_r_hi + v_r_an) / 3

                        angle_l = sum(self.angle_history_left) / len(self.angle_history_left)
                        angle_r = sum(self.angle_history_right) / len(self.angle_history_right)

                        # Configuration mapping
                        if self.selected_exercise == "Bicep Curl":
                            s_th, e_th = BICEP_CURL_START_ANGLE, BICEP_CURL_END_ANGLE
                            f_up, f_dn = "Curl Up", "Lower"
                        elif self.selected_exercise in ["Lat Pulldown/Row", "Chest Press", "Shoulder Press"]:
                            if self.selected_exercise == "Lat Pulldown/Row": s_th, e_th, f_up, f_dn = LAT_PULLDOWN_START_ANGLE, LAT_PULLDOWN_END_ANGLE, "Pull", "Extend"
                            elif self.selected_exercise == "Chest Press": s_th, e_th, f_up, f_dn = CHEST_PRESS_START_ANGLE, CHEST_PRESS_END_ANGLE, "Press", "Lower"
                            else: s_th, e_th, f_up, f_dn = SHOULDER_PRESS_START_ANGLE, SHOULDER_PRESS_END_ANGLE, "Press Up", "Lower"
                        elif self.selected_exercise in ["Squats", "Lunges"]:
                            s_th, e_th, f_up, f_dn = SQUAT_START_ANGLE, SQUAT_END_ANGLE, "Up", "Lower"
                        elif self.selected_exercise == "Pushups":
                            s_th, e_th, f_up, f_dn = PUSHUP_START_ANGLE, PUSHUP_END_ANGLE, "Up", "Lower"
                        
                        # Logic
                        if self.selected_exercise != "Plank":
                            if self.selected_mode == "Home":
                                # Unified Logic for Home Mode (Lunges, Squats, Pushups)
                                # Pick the most flexed angle (minimum) that's visible
                                if vis_l > VIS_THRESH and vis_r > VIS_THRESH:
                                    angle = min(angle_l, angle_r)
                                    vis_combined = (vis_l + vis_r) / 2
                                elif vis_l > VIS_THRESH:
                                    angle = angle_l
                                    vis_combined = vis_l
                                elif vis_r > VIS_THRESH:
                                    angle = angle_r
                                    vis_combined = vis_r
                                else:
                                    vis_combined = 0

                                if vis_combined > VIS_THRESH:
                                    if angle > s_th:
                                        self.is_extended = True
                                        if self.is_flexed:
                                            self.rep_count += 1
                                            self.is_extended, self.is_flexed = False, False
                                            self.feedback_text_rep = f"Rep {self.rep_count}"
                                        else: self.feedback_text_rep = f_dn
                                    elif angle < e_th:
                                        if self.is_extended: self.is_flexed = True; self.feedback_text_rep = f_up
                                        else: self.feedback_text_rep = f_dn
                                    else: self.feedback_text_rep = "..."
                                else:
                                    self.feedback_text_rep = "Hidden"
                            else:
                                # Left Logic (only if visible)
                                if vis_l > VIS_THRESH:
                                    if angle_l > s_th:
                                        self.is_extended_left = True
                                        if self.is_flexed_left:
                                            self.rep_count_left += 1
                                            self.is_extended_left, self.is_flexed_left = False, False
                                            self.feedback_l = f"L Rep {self.rep_count_left}"
                                        else: self.feedback_l = f_dn
                                    elif angle_l < e_th:
                                        if self.is_extended_left: self.is_flexed_left = True; self.feedback_l = f_up
                                        else: self.feedback_l = f_dn
                                    else: self.feedback_l = "..."
                                else: self.feedback_l = "Hidden"

                                # Right Logic (only if visible)
                                if vis_r > VIS_THRESH:
                                    if angle_r > s_th:
                                        self.is_extended_right = True
                                        if self.is_flexed_right:
                                            self.rep_count_right += 1
                                            self.is_extended_right, self.is_flexed_right = False, False
                                            self.feedback_r = f"R Rep {self.rep_count_right}"
                                        else: self.feedback_r = f_dn
                                    elif angle_r < e_th:
                                        if self.is_extended_right: self.is_flexed_right = True; self.feedback_r = f_up
                                        else: self.feedback_r = f_dn
                                    else: self.feedback_r = "..."
                                else: self.feedback_r = "Hidden"
                                
                                self.feedback_text_rep = f"L: {self.feedback_l} | R: {self.feedback_r}"
                        else: # Plank
                            if vis_l > VIS_THRESH and vis_r > VIS_THRESH and angle_l > PLANK_THRESHOLD_ANGLE and angle_r > PLANK_THRESHOLD_ANGLE:
                                if self.plank_start_time is None: self.plank_start_time = time.time()
                                else:
                                    curr = time.time()
                                    self.plank_accumulated_time += curr - self.plank_start_time
                                    self.plank_start_time = curr
                                self.feedback_text_rep = f"Hold: {self.plank_accumulated_time:.1f}s"
                            else:
                                self.plank_start_time = None
                                self.feedback_text_rep = "Adjust Body"

                        # Final update
                        self.feedback_text = f"{self.feedback_text_rep} | L: {int(angle_l)}° R: {int(angle_r)}° | Form: Classifying..."

                    else:
                        self.landmark_buffer.append(np.full((NUM_LANDMARKS, NUM_FEATURES), np.nan, dtype=np.float32).tolist())

                    self.inference_counter += 1
                    if len(self.landmark_buffer) == SEQUENCE_LENGTH and self.inference_counter % 15 == 0:
                        # Perform inference using App's perform_inference method
                        inference_result = self.controller.perform_inference(list(self.landmark_buffer))
                        if inference_result["action"] == "feedback":
                            # Combine form classification feedback with repetition feedback
                            self.feedback_text = f"{self.feedback_text_rep} | Form: {inference_result['feedback']}"
                        elif inference_result["action"] == "rep_increment":
                            pass # Redundant

                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=VisionRunningMode.LIVE_STREAM,
                    result_callback=result_callback
                )

                try:
                    self.pose_landmarker = PoseLandmarker.create_from_options(options)
                except Exception as e:
                    print(f"Error initializing PoseLandmarker: {e}")
                    self._end_session()
                    return

                # Start the pose detection thread
                self.running_thread = threading.Thread(target=self._pose_detection_loop)
                self.running_thread.daemon = True
                self.running_thread.start()

                # Start the GUI frame update loop
                self._update_frame()

            def _pose_detection_loop(self):
                timestamp_ms = 0
                while not self.stop_event.is_set():
                    if self.latest_frame_rgb is not None and self.pose_landmarker:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.latest_frame_rgb)
                        self.pose_landmarker.detect_async(mp_image, timestamp_ms)
                        timestamp_ms += 1
                    time.sleep(0.02) # Control frame rate of MediaPipe processing

            def _update_frame(self):
                if self.stop_event.is_set():
                    return

                success, frame = self.video_cap.read()
                if not success:
                    if self.input_source == "video_file":
                        self.feedback_label.configure(text=f"Feedback: Video Finished! {self.feedback_text}")
                        if self.video_cap:
                            self.video_cap.release()
                        print("Video processing completed. Displaying final results.")
                        return # Stop the loop but keep the last frame and data on screen
                    else:
                        print("Failed to read frame from webcam.")
                        self._end_session()
                        return

                # 1. Capture for MediaPipe (Detect on raw, unflipped frame for accuracy)
                self.latest_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 2. Draw landmarks on the raw BGR frame (while they still match coordinates)
                if self.latest_detection_result and self.latest_detection_result.pose_landmarks:
                    for pose_landmarks in self.latest_detection_result.pose_landmarks:
                        vision.drawing_utils.draw_landmarks(
                            frame,
                            pose_landmarks,
                            connections=POSE_CONNECTIONS
                        )

                # 3. Flip for selfie view (This flips both the person and the drawn landmarks)
                if self.input_source == "webcam":
                    frame = cv2.flip(frame, 1)

                # 4. Fast Resize using OpenCV
                label_width = self.video_display_frame.winfo_width()
                label_height = self.video_display_frame.winfo_height()
                
                if label_width > 10 and label_height > 10:
                    # Calculate scaling to fit
                    h, w = frame.shape[:2]
                    scale = min(label_width/w, label_height/h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # 5. Convert to PIL/Tkinter
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                # 6. Update labels less frequently or more efficiently
                self.feedback_label.configure(text=f"Feedback: {self.feedback_text}")
                if self.selected_exercise == "Plank":
                    self.reps_label.configure(text=f"Time: {self.plank_accumulated_time:.1f}s")
                elif self.selected_mode == "Home":
                    self.reps_label.configure(text=f"Reps: {self.rep_count}")
                else:
                    self.reps_label.configure(text=f"L Reps: {self.rep_count_left} | R Reps: {self.rep_count_right}")

                self.frame_update_id = self.after(40, self._update_frame) # 25 FPS target

            def _end_session(self):
                if self.running_thread and self.running_thread.is_alive():
                    self.stop_event.set()
                    self.running_thread.join(timeout=1) # Wait for thread to finish
                if self.video_cap and self.video_cap.isOpened():
                    self.video_cap.release()
                if self.frame_update_id:
                    self.after_cancel(self.frame_update_id)
                if self.pose_landmarker:
                    self.pose_landmarker.close() # Close the MediaPipe landmarker
                self.is_extended = False
                self.is_flexed = False

                self.video_label.configure(image=None, text="Video Session Ended") # Clear display
                self.feedback_label.configure(text="Feedback: ")
                self.reps_label.configure(text="Reps: 0")

                self.controller._show_frame_controller("InputSourceScreen", mode=self.selected_mode, exercise=self.selected_exercise)


        # Add other placeholder screens here if needed, e.g., InputSourceScreen etc.

        for F in (StartScreen, ModeSelectionScreen, ExerciseSelectionScreen, InputSourceScreen, LiveSessionScreen): # Add other screens here as they are created
            page_name = F.__name__
            frame = F(master=self, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self._show_frame_controller("StartScreen")

    def _show_frame_controller(self, page_name, **kwargs):
        frame = self.frames[page_name]
        # Re-initialize frame if kwargs change, to allow dynamic content
        if hasattr(frame, 'update_content'): # Check if frame has a method to update
            frame.update_content(**kwargs)
        frame.tkraise()

    def perform_inference(self, window_data):
        """
        Performs inference using the loaded Keras model to classify pose form (correct/incorrect).
        Preprocesses the window_data to match the model's input expectations.
        """
        if self.form_classification_model is None:
            return {"action": "feedback", "feedback": "Model not loaded."}

        # Convert window_data (list of lists) to numpy array
        sequence = np.array(window_data, dtype=np.float32)

        # Preprocessing steps (must match train_model.py)
        # Handle NaN values (e.g., fill with 0)
        sequence = np.nan_to_num(sequence, nan=0.0) 
        
        # Flatten the landmarks for a single timestep (33 * 4 features)
        # This transforms (N_frames, 33, 4) to (N_frames, 33*4)
        sequence_flattened = sequence.reshape(sequence.shape[0], NUM_LANDMARKS * NUM_FEATURES)

        # Pad or truncate sequences to a fixed length
        if sequence_flattened.shape[0] < SEQUENCE_LENGTH:
            # Pad with zeros
            padding = np.zeros((SEQUENCE_LENGTH - sequence_flattened.shape[0], NUM_LANDMARKS * NUM_FEATURES))
            processed_seq = np.vstack((sequence_flattened, padding))
        else:
            # Truncate
            processed_seq = sequence_flattened[:SEQUENCE_LENGTH]
        
        # Reshape for model input: (1, SEQUENCE_LENGTH, NUM_LANDMARKS * NUM_FEATURES)
        model_input = tf.convert_to_tensor(np.expand_dims(processed_seq, axis=0), dtype=tf.float32)

        # Make prediction - using direct call for better performance in real-time
        prediction_tensor = self.form_classification_model(model_input, training=False)
        prediction = prediction_tensor.numpy()[0][0]
        
        # Interpret prediction
        threshold = 0.5 # Binary classification threshold
        if prediction > threshold:
            # For simplicity, currently just classifies correct/incorrect.
            # Rep counting and specific feedback would require more sophisticated model output.
            return {"action": "feedback", "feedback": f"Form: Correct (Prob: {prediction:.2f})"}
        else:
            return {"action": "feedback", "feedback": f"Form: Incorrect (Prob: {prediction:.2f})"}

if __name__ == "__main__":
    app = App()
    app.mainloop()