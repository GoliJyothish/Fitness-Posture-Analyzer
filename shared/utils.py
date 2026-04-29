"""
shared/utils.py
---------------
Shared utility functions used by both the GUI application and the FastAPI
microservice.

Importing from here ensures calculate_angle() and preprocessing logic
are never duplicated or allowed to drift between files.
"""

import numpy as np
from typing import Sequence


def calculate_angle(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> float:
    """
    Calculate the angle (in degrees) at point b formed by vectors b→a and b→c.

    Uses the arctan2 method which correctly handles all quadrants.
    Result is always in the range [0, 180].

    Args:
        a: [x, y] coordinates of the first point  (e.g. shoulder)
        b: [x, y] coordinates of the vertex point (e.g. elbow)
        c: [x, y] coordinates of the end point    (e.g. wrist)

    Returns:
        Angle in degrees, clamped to [0, 180].

    Example:
        >>> calculate_angle([0, 1], [0, 0], [1, 0])
        90.0
    """
    a_arr = np.array(a[:2], dtype=float)
    b_arr = np.array(b[:2], dtype=float)
    c_arr = np.array(c[:2], dtype=float)

    radians = (
        np.arctan2(c_arr[1] - b_arr[1], c_arr[0] - b_arr[0])
        - np.arctan2(a_arr[1] - b_arr[1], a_arr[0] - b_arr[0])
    )
    angle = float(np.abs(np.degrees(radians)))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def preprocess_landmark_sequence(
    raw_sequence: list,
    sequence_length: int,
    num_landmarks: int,
    num_features: int,
) -> np.ndarray:
    """
    Preprocess a variable-length landmark sequence for LSTM inference.

    Steps:
        1. Convert to float32 numpy array
        2. Replace NaN values with 0.0
        3. Flatten each frame from (num_landmarks, num_features) to
           (num_landmarks * num_features,)
        4. Pad with zeros if shorter than sequence_length,
           or truncate if longer

    Args:
        raw_sequence: List of frames. Each frame is a list of num_landmarks
                      landmarks, each landmark is [x, y, z, visibility].
        sequence_length: Target fixed length — must match model input shape.
        num_landmarks:   Number of pose landmarks (33 for MediaPipe BlazePose).
        num_features:    Features per landmark (4: x, y, z, visibility).

    Returns:
        np.ndarray of shape (sequence_length, num_landmarks * num_features),
        dtype float32, with no NaN values.

    Example:
        >>> seq = [[[0.1, 0.2, 0.0, 1.0]] * 33] * 50  # 50 frames
        >>> out = preprocess_landmark_sequence(seq, 100, 33, 4)
        >>> out.shape
        (100, 132)
    """
    arr = np.array(raw_sequence, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)

    # Flatten: (T, num_landmarks, num_features) → (T, num_landmarks * num_features)
    flat = arr.reshape(arr.shape[0], num_landmarks * num_features)

    if flat.shape[0] < sequence_length:
        # Pad end with zeros
        pad = np.zeros(
            (sequence_length - flat.shape[0], num_landmarks * num_features),
            dtype=np.float32,
        )
        flat = np.vstack([flat, pad])
    else:
        # Truncate to fixed length
        flat = flat[:sequence_length]

    return flat
