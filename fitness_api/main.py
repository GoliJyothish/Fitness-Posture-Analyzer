"""
fitness_api/main.py
-------------------
FastAPI microservice for the AI Fitness Coach.

v2 changes:
- Landmarks now returned for ALL exercises (was only inside Lat Pulldown block)
- Rep counting REMOVED from API — now handled locally on the phone
- API now only returns: angle, form, feedback, landmarks
- This makes the app faster and more accurate (no network delay on reps)
"""

import os
import threading

from fastapi import FastAPI, File, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
from uuid import uuid4

def calculate_angle(a, b, c):
    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    c = np.array(c[:2], dtype=float)
    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = float(np.abs(np.degrees(radians)))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

app = FastAPI(title="AI Fitness Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
_thread_local = threading.local()

def _get_pose():
    if not hasattr(_thread_local, "pose"):
        _thread_local.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _thread_local.pose

slug_to_display = {
    "bicep_curl":       "Bicep Curl",
    "lat_pulldown_row": "Lat Pulldown/Row",
    "chest_press":      "Chest Press",
    "shoulder_press":   "Shoulder Press",
    "pushups":          "Pushups",
    "squats":           "Squats",
    "lunges":           "Lunges",
    "plank":            "Plank",
}

@app.get("/")
def home():
    return {"message": "Fitness API is running!"}


@app.post("/analyze/{exercise}")
async def analyze(
    exercise: str,
    file: UploadFile = File(...),
    x_session_id: str = Header(default=None),
):
    session_id = x_session_id or str(uuid4())
    exercise = slug_to_display.get(exercise.lower(), exercise)

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose = _get_pose()
    results = pose.process(image)

    # Default response when no pose detected
    if not results.pose_landmarks:
        response = JSONResponse(content={
            "angle_left":  0.0,
            "angle_right": 0.0,
            "angle":       0.0,
            "form":        "No pose detected",
            "feedback":    "Make sure your full body is visible!",
            "landmarks":   [],
        })
        response.headers["X-Session-Id"] = session_id
        return response

    landmarks = results.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def lm(idx):
        return [landmarks[idx].x, landmarks[idx].y]

    angle_left  = 0.0
    angle_right = 0.0
    angle       = 0.0
    form        = "Analyzing..."
    feedback    = "Keep going!"

    # -----------------------------------------------------------------------
    # Per-exercise angle calculation + form feedback ONLY
    # Rep counting is done locally on the phone
    # -----------------------------------------------------------------------
    if exercise == "Bicep Curl":
        angle_left  = calculate_angle(lm(L.LEFT_SHOULDER.value),  lm(L.LEFT_ELBOW.value),  lm(L.LEFT_WRIST.value))
        angle_right = calculate_angle(lm(L.RIGHT_SHOULDER.value), lm(L.RIGHT_ELBOW.value), lm(L.RIGHT_WRIST.value))
        angle = (angle_left + angle_right) / 2
        feedback = f"L: {round(angle_left)}°  R: {round(angle_right)}°"
        form = "Good Form!" if 50 <= angle <= 150 else "Adjust form!"

    elif exercise == "Pushups":
        angle_left  = calculate_angle(lm(L.LEFT_SHOULDER.value),  lm(L.LEFT_ELBOW.value),  lm(L.LEFT_WRIST.value))
        angle_right = calculate_angle(lm(L.RIGHT_SHOULDER.value), lm(L.RIGHT_ELBOW.value), lm(L.RIGHT_WRIST.value))
        angle = (angle_left + angle_right) / 2
        feedback = f"L: {round(angle_left)}°  R: {round(angle_right)}°"
        form = "Good Form!" if angle < 100 else "Go lower!"

    elif exercise == "Squats":
        angle_left  = calculate_angle(lm(L.LEFT_HIP.value),  lm(L.LEFT_KNEE.value),  lm(L.LEFT_ANKLE.value))
        angle_right = calculate_angle(lm(L.RIGHT_HIP.value), lm(L.RIGHT_KNEE.value), lm(L.RIGHT_ANKLE.value))
        angle = (angle_left + angle_right) / 2
        feedback = f"L: {round(angle_left)}°  R: {round(angle_right)}°"
        form = "Good Form!" if angle < 110 else "Go deeper!"

    elif exercise == "Lunges":
        angle_left  = calculate_angle(lm(L.LEFT_HIP.value),  lm(L.LEFT_KNEE.value),  lm(L.LEFT_ANKLE.value))
        angle_right = calculate_angle(lm(L.RIGHT_HIP.value), lm(L.RIGHT_KNEE.value), lm(L.RIGHT_ANKLE.value))
        angle = (angle_left + angle_right) / 2
        feedback = f"L: {round(angle_left)}°  R: {round(angle_right)}°"
        form = "Good Form!" if angle < 110 else "Go deeper!"

    elif exercise == "Plank":
        angle_left  = calculate_angle(lm(L.LEFT_SHOULDER.value),  lm(L.LEFT_HIP.value),  lm(L.LEFT_ANKLE.value))
        angle_right = calculate_angle(lm(L.RIGHT_SHOULDER.value), lm(L.RIGHT_HIP.value), lm(L.RIGHT_ANKLE.value))
        angle = (angle_left + angle_right) / 2
        form = "Good Form!" if 160 <= angle <= 180 else "Adjust form!"
        feedback = "Hold it!" if form == "Good Form!" else "Keep your body straight!"

    elif exercise == "Shoulder Press":
        angle_left  = calculate_angle(lm(L.LEFT_SHOULDER.value),  lm(L.LEFT_ELBOW.value),  lm(L.LEFT_WRIST.value))
        angle_right = calculate_angle(lm(L.RIGHT_SHOULDER.value), lm(L.RIGHT_ELBOW.value), lm(L.RIGHT_WRIST.value))
        angle = (angle_left + angle_right) / 2
        feedback = f"L: {round(angle_left)}°  R: {round(angle_right)}°"
        form = "Good Form!" if angle > 150 else "Press higher!"

    elif exercise == "Chest Press":
        angle_left  = calculate_angle(lm(L.LEFT_SHOULDER.value),  lm(L.LEFT_ELBOW.value),  lm(L.LEFT_WRIST.value))
        angle_right = calculate_angle(lm(L.RIGHT_SHOULDER.value), lm(L.RIGHT_ELBOW.value), lm(L.RIGHT_WRIST.value))
        angle = (angle_left + angle_right) / 2
        feedback = f"L: {round(angle_left)}°  R: {round(angle_right)}°"
        form = "Good Form!" if angle > 150 else "Extend fully!"

    elif exercise == "Lat Pulldown/Row":
        angle_left  = calculate_angle(lm(L.LEFT_SHOULDER.value),  lm(L.LEFT_ELBOW.value),  lm(L.LEFT_WRIST.value))
        angle_right = calculate_angle(lm(L.RIGHT_SHOULDER.value), lm(L.RIGHT_ELBOW.value), lm(L.RIGHT_WRIST.value))
        angle = (angle_left + angle_right) / 2
        feedback = f"L: {round(angle_left)}°  R: {round(angle_right)}°"
        form = "Good Form!" if angle < 100 else "Pull lower!"

    # -----------------------------------------------------------------------
    # Build landmark list for skeleton overlay — runs for ALL exercises
    # -----------------------------------------------------------------------
    landmark_data = [
        {
            "x":          round(lm_pt.x, 4),
            "y":          round(lm_pt.y, 4),
            "visibility": round(lm_pt.visibility, 4),
        }
        for lm_pt in results.pose_landmarks.landmark
    ]

    response = JSONResponse(content={
        "angle_left":  round(angle_left,  1),
        "angle_right": round(angle_right, 1),
        "angle":       round(angle,       1),
        "form":        form,
        "feedback":    feedback,
        "landmarks":   landmark_data,
    })
    response.headers["X-Session-Id"] = session_id
    return response
