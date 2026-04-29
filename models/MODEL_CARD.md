# Model Card — Form Classification Model

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | `form_classification_model.keras` |
| **Model type** | LSTM binary classifier |
| **Framework** | TensorFlow / Keras |
| **Input shape** | `(100, 132)` — 100 frames × (33 landmarks × 4 features) |
| **Output** | Sigmoid probability — `> 0.5` = Correct form, `≤ 0.5` = Incorrect form |
| **File size** | ~2 MB |
| **Random seed** | 42 (set in `scripts/train_model.py`) |

---

## Architecture

```
Input(100, 132)
    ↓
LSTM(64, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(32)
    ↓
Dropout(0.2)
    ↓
Dense(1, activation='sigmoid')
```

- **Optimizer:** Adam (default lr=0.001)
- **Loss:** Binary cross-entropy
- **Metric:** Accuracy
- **Epochs:** 10
- **Batch size:** 32

---

## Input Format

Each input is a sequence of **100 frames** of MediaPipe BlazePose landmarks.

Per frame, each of the 33 landmarks provides 4 features:

| Index | Feature | Range |
|-------|---------|-------|
| 0 | `x` | 0.0 – 1.0 (normalized image width) |
| 1 | `y` | 0.0 – 1.0 (normalized image height) |
| 2 | `z` | depth relative to hip midpoint |
| 3 | `visibility` | 0.0 – 1.0 |

NaN values (occluded landmarks) are replaced with `0.0` before inference.

Sequences shorter than 100 frames are zero-padded at the end.
Sequences longer than 100 frames are truncated to the first 100 frames.

---

## Training Data

| Field | Value |
|-------|-------|
| **Source** | `.npy` landmark files in `data/processed/train/` |
| **Classes** | `correct` (label=1), `incorrect` (label=0) |
| **Split** | 80% train / 20% validation (`random_state=42`) |
| **Exercises covered** | Bicep Curl, Pushups, Squats, Lunges, Plank, Shoulder Press, Chest Press, Lat Pulldown/Row |

> **Note:** Training data `.npy` files are not included in this repository
> due to file size. See `data/README.md` for instructions on how to collect
> and process your own data using `scripts/collect_data.py`.

---

## Performance

> Fill in these numbers after running `scripts/train_model.py`.
> The classification report is printed at the end of training and
> saved to `models/training_history.json`.

| Metric | Correct Form | Incorrect Form |
|--------|-------------|----------------|
| Precision | — | — |
| Recall | — | — |
| F1 Score | — | — |
| **Overall Accuracy** | — | — |

---

## Limitations & Known Issues

- **Single-person only** — model assumes one person is visible in frame.
  Multi-person scenes will use landmarks from the first detected pose only.

- **Camera angle sensitivity** — model was trained on specific camera angles.
  Very oblique or overhead views may degrade accuracy.

- **Occlusion** — heavily occluded joints (visibility < 0.5) are zeroed out,
  which can reduce classification confidence.

- **Not a medical device** — this model provides fitness guidance only and
  should not be used for medical or rehabilitation purposes without
  professional supervision.

---

## Companion Model

| Model | File | Purpose |
|-------|------|---------|
| MediaPipe BlazePose Heavy | `pose_landmarker_heavy.task` | Landmark extraction |
| Form Classifier (this card) | `form_classification_model.keras` | Correct/incorrect classification |

The `pose_landmarker_heavy.task` file must be downloaded separately from the
[MediaPipe Model Zoo](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker).

---

## How to Retrain

```bash
# 1. Collect landmark data
python scripts/collect_data.py

# 2. Train the model (seeds are set automatically)
python scripts/train_model.py

# 3. Run tests to verify shared utilities still pass
pytest tests/ -v
```

Training history (loss/accuracy per epoch) is saved to
`models/training_history.json` for comparison across runs.
