"""
shared/constants.py
-------------------
Single source of truth for all configuration values used across the app,
the FastAPI microservice, and the training scripts.

Any change to thresholds, paths, or exercise names should be made HERE only.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Model configuration — must match train_model.py and inference in GUI/API
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH: Final[int] = 100
NUM_LANDMARKS: Final[int] = 33
NUM_FEATURES: Final[int] = 4  # x, y, z, visibility

# ---------------------------------------------------------------------------
# Model file paths (relative to project root)
# ---------------------------------------------------------------------------
FORM_MODEL_PATH: Final[str] = "models/form_classification_model.keras"
POSE_LANDMARKER_PATH: Final[str] = "models/pose_landmarker_heavy.task"

# ---------------------------------------------------------------------------
# Exercise mode mapping — used by GUI screens and API routing
# ---------------------------------------------------------------------------
EXERCISES_BY_MODE: Final[dict] = {
    "Gym":  ["Bicep Curl", "Lat Pulldown/Row", "Chest Press", "Shoulder Press"],
    "Home": ["Pushups", "Squats", "Lunges", "Plank"],
}

# ---------------------------------------------------------------------------
# Canonical slug → display name mapping
# Slugs are used as URL path parameters in the FastAPI microservice.
# Display names are what the GUI shows and what the rep FSM matches against.
# ---------------------------------------------------------------------------
EXERCISE_SLUGS: Final[dict] = {
    "bicep_curl":       "Bicep Curl",
    "lat_pulldown_row": "Lat Pulldown/Row",
    "chest_press":      "Chest Press",
    "shoulder_press":   "Shoulder Press",
    "pushups":          "Pushups",
    "squats":           "Squats",
    "lunges":           "Lunges",
    "plank":            "Plank",
}

# Reverse mapping: display name → slug (for GUI → API calls)
EXERCISE_DISPLAY_TO_SLUG: Final[dict] = {v: k for k, v in EXERCISE_SLUGS.items()}

# ---------------------------------------------------------------------------
# Angle thresholds per exercise
# Tuple format: (start_angle, end_angle, prompt_up, prompt_down)
#   start_angle — arm/leg is considered "extended" above this angle
#   end_angle   — arm/leg is considered "flexed" below this angle
# ---------------------------------------------------------------------------
EXERCISE_CONFIG: Final[dict] = {
    "Bicep Curl":       (150.0, 50.0,  "Curl Up",  "Lower"),
    "Lat Pulldown/Row": (150.0, 100.0, "Pull",     "Extend"),
    "Chest Press":      (150.0, 100.0, "Press",    "Lower"),
    "Shoulder Press":   (150.0, 100.0, "Press Up", "Lower"),
    "Pushups":          (150.0, 100.0, "Up",       "Lower"),
    "Squats":           (150.0, 100.0, "Up",       "Lower"),
    "Lunges":           (150.0, 100.0, "Up",       "Lower"),
    "Plank":            (160.0, 0.0,   "Hold",     "Adjust Body"),
}

# ---------------------------------------------------------------------------
# Detection and inference settings
# ---------------------------------------------------------------------------

# Minimum landmark visibility score to consider a joint "visible" (0.0–1.0)
VISIBILITY_THRESHOLD: Final[float] = 0.5

# Binary classification threshold for the LSTM sigmoid output
FORM_CLASSIFICATION_THRESHOLD: Final[float] = 0.5

# Run LSTM inference every N frames (balances latency vs CPU load)
INFERENCE_INTERVAL_FRAMES: Final[int] = 15

# Number of frames to smooth joint angles over (reduces jitter)
ANGLE_SMOOTHING_WINDOW: Final[int] = 5

# Minimum seconds between voice feedback utterances (prevents TTS spam)
VOICE_COOLDOWN_SECONDS: Final[float] = 2.0

# Plank: body must be straighter than this angle to count as "holding"
PLANK_THRESHOLD_ANGLE: Final[float] = 160.0
