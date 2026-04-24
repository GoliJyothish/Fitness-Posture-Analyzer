from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

counters = {}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

@app.get("/")
def home():
    return {"message": "Fitness API is running!"}

@app.post("/analyze/{exercise}")
async def analyze(exercise: str, file: UploadFile = File(...)):
    global counters

    if exercise not in counters:
        counters[exercise] = {
            "count": 0,
            "stage": None,
            "left_count": 0,
            "right_count": 0,
            "left_stage": None,
            "right_stage": None
        }

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if not results.pose_landmarks:
        return {
            "reps": counters[exercise]["count"],
            "left_reps": counters[exercise]["left_count"],
            "right_reps": counters[exercise]["right_count"],
            "form": "No pose detected",
            "feedback": "Make sure your full body is visible!",
            "angle": 0
        }

    landmarks = results.pose_landmarks.landmark
    angle = 0
    form = "Analyzing..."
    feedback = "Keep going!"

    if exercise == "Bicep Curl":
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        if left_angle > 160:
            counters[exercise]["left_stage"] = "down"
        if left_angle < 30 and counters[exercise]["left_stage"] == "down":
            counters[exercise]["left_stage"] = "up"
            counters[exercise]["left_count"] += 1

        if right_angle > 160:
            counters[exercise]["right_stage"] = "down"
        if right_angle < 30 and counters[exercise]["right_stage"] == "down":
            counters[exercise]["right_stage"] = "up"
            counters[exercise]["right_count"] += 1

        angle = (left_angle + right_angle) / 2
        feedback = f"L: {round(left_angle)}° R: {round(right_angle)}°"
        form = "Good Form!" if 30 <= angle <= 160 else "Adjust form!"

    elif exercise == "Push-ups":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angle = calculate_angle(shoulder, elbow, wrist)

        if angle > 160:
            counters[exercise]["stage"] = "up"
            feedback = "Go down!"
        if angle < 90 and counters[exercise]["stage"] == "up":
            counters[exercise]["stage"] = "down"
            counters[exercise]["count"] += 1
            feedback = "Push up!"

        form = "Good Form!" if angle < 90 else "Go lower!"

    elif exercise == "Squats":
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        angle = calculate_angle(hip, knee, ankle)

        if angle > 160:
            counters[exercise]["stage"] = "up"
            feedback = "Squat down!"
        if angle < 90 and counters[exercise]["stage"] == "up":
            counters[exercise]["stage"] = "down"
            counters[exercise]["count"] += 1
            feedback = "Stand up!"

        form = "Good Form!" if angle < 90 else "Go deeper!"

    elif exercise == "Lunges":
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        angle = calculate_angle(hip, knee, ankle)

        if angle > 160:
            counters[exercise]["stage"] = "up"
            feedback = "Lunge down!"
        if angle < 90 and counters[exercise]["stage"] == "up":
            counters[exercise]["stage"] = "down"
            counters[exercise]["count"] += 1
            feedback = "Stand up!"

        form = "Good Form!" if angle < 90 else "Go deeper!"

    elif exercise == "Plank":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        angle = calculate_angle(shoulder, hip, ankle)

        if 160 <= angle <= 180:
            form = "Good Form!"
            feedback = "Hold it!"
        else:
            form = "Adjust form!"
            feedback = "Keep your body straight!"

    elif exercise == "Shoulder Press":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angle = calculate_angle(shoulder, elbow, wrist)

        if angle < 90:
            counters[exercise]["stage"] = "down"
            feedback = "Press up!"
        if angle > 160 and counters[exercise]["stage"] == "down":
            counters[exercise]["stage"] = "up"
            counters[exercise]["count"] += 1
            feedback = "Lower down!"

        form = "Good Form!" if angle > 160 else "Press higher!"

    elif exercise == "Chest Press":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angle = calculate_angle(shoulder, elbow, wrist)

        if angle < 90:
            counters[exercise]["stage"] = "down"
            feedback = "Press out!"
        if angle > 160 and counters[exercise]["stage"] == "down":
            counters[exercise]["stage"] = "up"
            counters[exercise]["count"] += 1
            feedback = "Bring it back!"

        form = "Good Form!" if angle > 160 else "Extend fully!"

    elif exercise == "Lat Pulldown":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angle = calculate_angle(shoulder, elbow, wrist)

        if angle > 160:
            counters[exercise]["stage"] = "up"
            feedback = "Pull down!"
        if angle < 90 and counters[exercise]["stage"] == "up":
            counters[exercise]["stage"] = "down"
            counters[exercise]["count"] += 1
            feedback = "Release up!"

        form = "Good Form!" if angle < 90 else "Pull lower!"

    return {
        "reps": counters[exercise]["count"],
        "left_reps": counters[exercise]["left_count"],
        "right_reps": counters[exercise]["right_count"],
        "form": form,
        "feedback": feedback,
        "angle": round(angle, 1)
    }

@app.post("/reset/{exercise}")
def reset(exercise: str):
    if exercise in counters:
        counters[exercise] = {
            "count": 0,
            "stage": None,
            "left_count": 0,
            "right_count": 0,
            "left_stage": None,
            "right_stage": None
        }
    return {"message": "Reset successful"}