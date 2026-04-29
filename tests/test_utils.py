"""
tests/test_utils.py
-------------------
Unit tests for shared utility functions.

Run with:
    pytest tests/ -v

Tests cover:
- calculate_angle()  : correctness, edge cases, output range
- preprocess_landmark_sequence() : shape, padding, truncation, NaN handling
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from shared.utils import calculate_angle, preprocess_landmark_sequence


# ---------------------------------------------------------------------------
# calculate_angle() tests
# ---------------------------------------------------------------------------

class TestCalculateAngle:

    def test_right_angle(self):
        """Elbow at 90°: shoulder above, wrist to the right."""
        angle = calculate_angle(a=[0, 1], b=[0, 0], c=[1, 0])
        assert abs(angle - 90.0) < 1.0, f"Expected ~90°, got {angle}"

    def test_straight_line(self):
        """Three collinear points should give 180°."""
        angle = calculate_angle(a=[0, 0], b=[1, 0], c=[2, 0])
        assert abs(angle - 180.0) < 1.0, f"Expected ~180°, got {angle}"

    def test_fully_flexed_45(self):
        """Points forming a 45° angle."""
        angle = calculate_angle(a=[0, 1], b=[0, 0], c=[1, 1])
        assert abs(angle - 45.0) < 1.0, f"Expected ~45°, got {angle}"

    def test_angle_always_in_range(self):
        """Angle must always be in [0, 180] for any input."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            pts = rng.random((3, 2)).tolist()
            angle = calculate_angle(pts[0], pts[1], pts[2])
            assert 0.0 <= angle <= 180.0, f"Angle out of range: {angle}"

    def test_accepts_3d_points(self):
        """Should ignore z-coordinate and still return correct 2D angle."""
        angle = calculate_angle(a=[0, 1, 5], b=[0, 0, 3], c=[1, 0, 7])
        assert abs(angle - 90.0) < 1.0, f"Expected ~90°, got {angle}"

    def test_zero_angle(self):
        """Points that form a 0° angle (a and c on same side of b)."""
        angle = calculate_angle(a=[1, 0], b=[0, 0], c=[2, 0])
        assert abs(angle - 0.0) < 1.0, f"Expected ~0°, got {angle}"

    def test_returns_float(self):
        """Return type must be float."""
        result = calculate_angle([0, 1], [0, 0], [1, 0])
        assert isinstance(result, float), f"Expected float, got {type(result)}"


# ---------------------------------------------------------------------------
# preprocess_landmark_sequence() tests
# ---------------------------------------------------------------------------

class TestPreprocessLandmarkSequence:

    SEQ_LEN = 100
    N_LM = 33
    N_FT = 4

    def _make_seq(self, n_frames: int) -> list:
        """Create a dummy landmark sequence with n_frames frames."""
        return [
            [[float(i)] * self.N_FT for _ in range(self.N_LM)]
            for i in range(n_frames)
        ]

    def test_output_shape_short_sequence(self):
        """Sequences shorter than SEQ_LEN should be zero-padded."""
        seq = self._make_seq(50)
        result = preprocess_landmark_sequence(seq, self.SEQ_LEN, self.N_LM, self.N_FT)
        assert result.shape == (self.SEQ_LEN, self.N_LM * self.N_FT), \
            f"Expected ({self.SEQ_LEN}, {self.N_LM * self.N_FT}), got {result.shape}"

    def test_output_shape_exact_sequence(self):
        """Sequences exactly SEQ_LEN long should pass through unchanged."""
        seq = self._make_seq(self.SEQ_LEN)
        result = preprocess_landmark_sequence(seq, self.SEQ_LEN, self.N_LM, self.N_FT)
        assert result.shape == (self.SEQ_LEN, self.N_LM * self.N_FT)

    def test_output_shape_long_sequence(self):
        """Sequences longer than SEQ_LEN should be truncated."""
        seq = self._make_seq(150)
        result = preprocess_landmark_sequence(seq, self.SEQ_LEN, self.N_LM, self.N_FT)
        assert result.shape == (self.SEQ_LEN, self.N_LM * self.N_FT)

    def test_nan_replaced_with_zero(self):
        """All NaN values must be replaced with 0.0."""
        seq = [[[float("nan")] * self.N_FT for _ in range(self.N_LM)]]
        result = preprocess_landmark_sequence(seq, self.SEQ_LEN, self.N_LM, self.N_FT)
        assert not np.isnan(result).any(), "NaN values found in output"
        assert (result[0] == 0.0).all(), "NaN rows should be all zeros"

    def test_padding_is_zeros(self):
        """Padded frames (beyond input length) must be zero."""
        n_input = 30
        seq = self._make_seq(n_input)
        result = preprocess_landmark_sequence(seq, self.SEQ_LEN, self.N_LM, self.N_FT)
        pad_section = result[n_input:]
        assert (pad_section == 0.0).all(), "Padding rows should be all zeros"

    def test_output_dtype_is_float32(self):
        """Output array must be float32 to match model input expectations."""
        seq = self._make_seq(50)
        result = preprocess_landmark_sequence(seq, self.SEQ_LEN, self.N_LM, self.N_FT)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_truncation_keeps_first_frames(self):
        """Truncation should keep the FIRST SEQ_LEN frames, not the last."""
        seq = self._make_seq(150)
        result = preprocess_landmark_sequence(seq, self.SEQ_LEN, self.N_LM, self.N_FT)
        # First frame should have value 0.0 (frame index 0)
        assert result[0, 0] == 0.0
        # Frame at index SEQ_LEN-1 should have value SEQ_LEN-1
        assert result[self.SEQ_LEN - 1, 0] == float(self.SEQ_LEN - 1)

    def test_single_frame_input(self):
        """Single-frame input should be padded to full SEQ_LEN."""
        seq = self._make_seq(1)
        result = preprocess_landmark_sequence(seq, self.SEQ_LEN, self.N_LM, self.N_FT)
        assert result.shape == (self.SEQ_LEN, self.N_LM * self.N_FT)
        assert (result[1:] == 0.0).all()
