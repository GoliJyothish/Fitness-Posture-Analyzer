"""
fitness_api/main.py
-------------------
FastAPI microservice for the AI Fitness Coach.

Fixes applied (per analysis MD):
1. Global counters dict replaced with per-session store keyed by X-Session-Id header
   — previously all users shared one counter, reps bled between clients.
2. Exercise name strings normalized to match GUI exactly
   ("Push-ups" → "Pushups", "Lat Pulldown" → "Lat Pulldown/Row").
3. MediaPipe Pose object moved into threading.local() — the module-level
   singleton was not thread-safe for concurrent FastAPI requests.
4. calculate_angle() inlined directly — avoids Python path issues on Render.
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

# Inlined from shared/utils.py — avoids Python path issues on Render
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
    allow_origins=["*"],  # Restrict to known origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose

# FIX (MD §4): Use threading.local() so each thread gets its own Pose instance.
# The module-level singleton pose = mp_pose.Pose(...) was not thread-safe
# for concurrent FastAPI requests.
_thread_local = threading.local()

def _get_pose():
    """Return a thread-local MediaPipe Pose instance, creating it if needed."""
    if not hasattr(_thread_local, "pose"):
        _thread_local.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _thread_local.pose


def _empty_counter() -> dict:
    return {
        "count": 0,
        "stage": None,
        "left_count": 0,
        "right_count": 0,
        "left_stage": None,
        "right_stage": None,
    }


# FIX (MD §4): Per-session store keyed by X-Session-Id header.
# Replaces the module-level `counters = {}` that was shared across all clients.
session_store: dict[str, dict] = {}


@app.get("/")
def home():
    return {"message": "Fitness API is running!"}


@app.post("/analyze/{exercise}")
async def analyze(
    exercise: str,
    file: UploadFile = File(...),
    x_session_id: str = Header(default=None),
):
    # FIX (MD §4): Assign or reuse session ID so each client has isolated state
    session_id = x_session_id or str(uuid4())
    if session_id not in session_store:
        session_store[session_id] = {}
    counters = session_store[session_id]

    # FIX (MD §4): Normalize exercise name from URL slug to display name.
    # Previously "Push-ups" and "Lat Pulldown" never matched any branch,
    # silently returning form="Analyzing..." and angle=0.
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
    exercise = slug_to_display.get(exercise.lower(), exercise)

    if exercise not in counters:
        counters[exercise] = _empty_counter()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        response = JSONResponse(content={"error": "Invalid image"}, status_code=400)
        response.headers["X-Session-Id"] = session_id
        return response

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FIX (MD §4): Use thread-local Pose instance
    pose = _get_pose()
    results = pose.process(image)

    if not results.pose_landmarks:
        response = JSONResponse(content={
            "reps":      counters[exercise]["count"],
            "left_reps": counters[exercise]["left_count"],
            "right_reps": counters[exercise]["right_count"],
            "form":      "No pose detected",
            "feedback":  "Make sure your full body is visible!",
            "angle":     0,
        })
        response.headers["X-Session-Id"] = session_id
        return response

    landmarks = results.pose_landmarks.landmark
    angle = 0
    form = "Analyzing..."
    feedback = "Keep going!"

    # Helper to extract [x, y] from a landmark index
    def lm(idx):
        return [landmarks[idx].x, landmarks[idx].y]

    L = mp_pose.PoseLandmark

    # FIX (MD §4): All exercise names now match the GUI exactly
    if exercise == "Bicep Curl":
        left_angle  = calculate_angle(lm(L.LEFT_SHOULDER.value),  lm(L.LEFT_ELBOW.value),  lm(L.LEFT_WRIST.value))
        right_angle = calculate_angle(lm(L.RIGHT_SHOULDER.value), lm(L.RIGHT_ELBOW.value), lm(L.RIGHT_WRIST.value))

        if left_angle > 140:
            counters[exercise]["left_stage"] = "down"
        if left_angle < 60 and counters[exercise]["left_stage"] == "down":
            counters[exercise]["left_stage"] = "up"
            counters[exercise]["left_count"] += 1

        if right_angle > 140:
            counters[exercise]["right_stage"] = "down"
        if right_angle < 60 and counters[exercise]["right_stage"] == "down":
            counters[exercise]["right_stage"] = "up"
            counters[exercise]["right_count"] += 1

        angle = (left_angle + right_angle) / 2
        feedback = f"L: {round(left_angle)}° R: {round(right_angle)}°"
        form = "Good Form!" if 50 <= angle <= 150 else "Adjust form!"

    elif exercise == "Pushups":  # FIX: was "Push-ups"
        angle = calculate_angle(lm(L.LEFT_SHOULDER.value), lm(L.LEFT_ELBOW.value), lm(L.LEFT_WRIST.value))
        if angle > 150:
            counters[exercise]["stage"] = "up"
            feedback = "Go down!"
        if angle < 100 and counters[exercise]["stage"] == "up":
            counters[exercise]["stage"] = "down"
            counters[exercise]["count"] += 1
            feedback = "Push up!"
        form = "Good Form!" if angle < 100 else "Go lower!"

    elif exercise == "Squats":
        angle = calculate_angle(lm(L.LEFT_HIP.value), lm(L.LEFT_KNEE.value), lm(L.LEFT_ANKLE.value))
        if angle > 150:
            counters[exercise]["stage"] = "up"
            feedback = "Squat down!"
        if angle < 100 and counters[exercise]["stage"] == "up":
            counters[exercise]["stage"] = "down"
            counters[exercise]["count"] += 1
            feedback = "Stand up!"
        form = "Good Form!" if angle < 100 else "Go deeper!"

    elif exercise == "Lunges":
        angle = calculate_angle(lm(L.LEFT_HIP.value), lm(L.LEFT_KNEE.value), lm(L.LEFT_ANKLE.value))
        if angle > 150:
            counters[exercise]["stage"] = "up"
            feedback = "Lunge down!"
        if angle < 100 and counters[exercise]["stage"] == "up":
            counters[exercise]["stage"] = "down"
            counters[exercise]["count"] += 1
            feedback = "Stand up!"
        form = "Good Form!" if angle < 100 else "Go deeper!"

    elif exercise == "Plank":
        angle = calculate_angle(lm(L.LEFT_SHOULDER.value), lm(L.LEFT_HIP.value), lm(L.LEFT_ANKLE.value))
        if 160 <= angle <= 180:
            form = "Good Form!"
            feedback = "Hold it!"
        else:
            form = "Adjust form!"
            feedback = "Keep your body straight!"

    elif exercise == "Shoulder Press":
        angle = calculate_angle(lm(L.LEFT_SHOULDER.value), lm(L.LEFT_ELBOW.value), lm(L.LEFT_WRIST.value))
        if angle < 90:
            counters[exercise]["stage"] = "down"
            feedback = "Press up!"
        if angle > 150 and counters[exercise]["stage"] == "down":
            counters[exercise]["stage"] = "up"
            counters[exercise]["count"] += 1
            feedback = "Lower down!"
        form = "Good Form!" if angle > 150 else "Press higher!"

    elif exercise == "Chest Press":
        angle = calculate_angle(lm(L.LEFT_SHOULDER.value), lm(L.LEFT_ELBOW.value), lm(L.LEFT_WRIST.value))
        if angle < 90:
            counters[exercise]["stage"] = "down"
            feedback = "Press out!"
        if angle > 150 and counters[exercise]["stage"] == "down":
            counters[exercise]["stage"] = "up"
            counters[exercise]["count"] += 1
            feedback = "Bring it back!"
        form = "Good Form!" if angle > 150 else "Extend fully!"

    elif exercise == "Lat Pulldown/Row":  # FIX: was "Lat Pulldown"
        angle = calculate_angle(lm(L.LEFT_SHOULDER.value), lm(L.LEFT_ELBOW.value), lm(L.LEFT_WRIST.value))
        if angle > 150:
            counters[exercise]["stage"] = "up"
            feedback = "Pull down!"
        if angle < 100 and counters[exercise]["stage"] == "up":
            counters[exercise]["stage"] = "down"
            counters[exercise]["count"] += 1
            feedback = "Release up!"
        form = "Good Form!" if angle < 100 else "Pull lower!"

    response = JSONResponse(content={
        "reps":      counters[exercise]["count"],
        "left_reps": counters[exercise]["left_count"],
        "right_reps": counters[exercise]["right_count"],
        "form":      form,
        "feedback":  feedback,
        "angle":     round(angle, 1),
    })
    # Always echo the session ID back so the client can reuse it
    response.headers["X-Session-Id"] = session_id
    return response


@app.post("/reset/{exercise}")
def reset(
    exercise: str,
    x_session_id: str = Header(default=None),
):
    """Reset rep counters for a specific exercise in the caller's session."""
    if not x_session_id or x_session_id not in session_store:
        return {"message": "Session not found — nothing to reset."}
    if exercise in session_store[x_session_id]:
        session_store[x_session_id][exercise] = _empty_counter()
    return {"message": f"Reset successful for session {x_session_id}"}


@app.post("/reset_session")
def reset_session(x_session_id: str = Header(default=None)):
    """Clear all exercise counters for the caller's session."""
    if x_session_id and x_session_id in session_store:
        session_store.pop(x_session_id)
    return {"message": "Session cleared."}
