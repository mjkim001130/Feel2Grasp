#!/usr/bin/env python
"""
Real-time Circle Detection for SO101 Robot Control

This module provides real-time circle detection for left and right cameras
using background subtraction, adapted from circle.py.

The detector maintains a running background model and detects changes
to determine if a circle (object) is present in the camera view.
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple


class CircleDetector:
    """
    Real-time circle detector using background subtraction.

    This class maintains a background model per camera and detects
    when significant changes occur (indicating an object/circle in view).

    Usage:
        detector = CircleDetector(num_bg_frames=30, k_std=1.5)

        # During initialization phase (collect background)
        for i in range(30):
            left_img, right_img = robot.get_camera_images()
            detector.add_background_frame(left_img, "left")
            detector.add_background_frame(right_img, "right")

        # During control loop
        left_img, right_img = robot.get_camera_images()
        left_circle = detector.detect(left_img, "left")
        right_circle = detector.detect(right_img, "right")
    """

    def __init__(
        self,
        num_bg_frames: int = 30,
        k_std: float = 1.5,
        use_center_crop: bool = True,
        history_size: int = 10,
    ):
        """
        Args:
            num_bg_frames: Number of frames to use for building background model
            k_std: Threshold multiplier (threshold = mean + k_std * std)
            use_center_crop: Whether to use center 60% crop for detection
            history_size: Number of recent scores to track for adaptive thresholding
        """
        self.num_bg_frames = num_bg_frames
        self.k_std = k_std
        self.use_center_crop = use_center_crop
        self.history_size = history_size

        # Storage for background frames and models
        self.bg_frames = {"left": deque(maxlen=num_bg_frames), "right": deque(maxlen=num_bg_frames)}
        self.bg_model = {"left": None, "right": None}

        # Score history for adaptive thresholding
        self.score_history = {"left": deque(maxlen=history_size), "right": deque(maxlen=history_size)}

        self.is_initialized = {"left": False, "right": False}

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale float32."""
        # Handle different input formats
        if isinstance(img, np.ndarray):
            # If already grayscale
            if img.ndim == 2:
                return img.astype(np.float32)

            # If RGB/BGR uint8
            if img.dtype == np.uint8:
                if img.shape[-1] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img
                return gray.astype(np.float32)

            # If float [0, 1]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
                if img.ndim == 3 and img.shape[-1] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img
                return gray.astype(np.float32)

        raise ValueError(f"Unsupported image format: shape={img.shape}, dtype={img.dtype}")

    def add_background_frame(self, img: np.ndarray, camera: str = "left"):
        """
        Add a frame to the background model.

        Args:
            img: Image array (H, W, 3) or (H, W)
            camera: "left" or "right"
        """
        if camera not in ["left", "right"]:
            raise ValueError(f"Camera must be 'left' or 'right', got {camera}")

        gray = self._to_grayscale(img)
        self.bg_frames[camera].append(gray)

        # Update background model if we have enough frames
        if len(self.bg_frames[camera]) >= self.num_bg_frames:
            frames = list(self.bg_frames[camera])
            self.bg_model[camera] = np.mean(frames, axis=0).astype(np.float32)
            self.is_initialized[camera] = True

    def _compute_diff_score(self, img: np.ndarray, bg: np.ndarray) -> float:
        """
        Compute difference score between current frame and background.

        Args:
            img: Current frame (grayscale float32)
            bg: Background model (grayscale float32)

        Returns:
            score: Mean absolute difference
        """
        diff = np.abs(img - bg)

        if self.use_center_crop:
            h, w = diff.shape
            y0, y1 = int(h * 0.2), int(h * 0.8)
            x0, x1 = int(w * 0.2), int(w * 0.8)
            diff = diff[y0:y1, x0:x1]

        return float(np.mean(diff))

    def detect(
        self,
        img: np.ndarray,
        camera: str = "left",
        update_bg: bool = False,
    ) -> int:
        """
        Detect if circle is present in current frame.

        Args:
            img: Current frame (H, W, 3) or (H, W)
            camera: "left" or "right"
            update_bg: Whether to update background model with this frame

        Returns:
            detection: 1 if circle detected, 0 otherwise
        """
        if camera not in ["left", "right"]:
            raise ValueError(f"Camera must be 'left' or 'right', got {camera}")

        if not self.is_initialized[camera]:
            # Not initialized yet, add to background
            self.add_background_frame(img, camera)
            return 0

        gray = self._to_grayscale(img)
        bg = self.bg_model[camera]

        # Compute difference score
        score = self._compute_diff_score(gray, bg)

        # Add to history
        self.score_history[camera].append(score)

        # Compute adaptive threshold from history
        if len(self.score_history[camera]) < 3:
            # Not enough history, return 0
            return 0

        scores = list(self.score_history[camera])
        mean_score = np.mean(scores)
        std_score = np.std(scores) + 1e-8
        threshold = mean_score + self.k_std * std_score

        # Detect
        detection = 1 if score >= threshold else 0

        # Optional: update background model with current frame if no detection
        if update_bg and detection == 0:
            self.add_background_frame(img, camera)

        return detection

    def reset(self, camera: Optional[str] = None):
        """
        Reset detector state.

        Args:
            camera: "left", "right", or None (reset both)
        """
        if camera is None:
            cameras = ["left", "right"]
        else:
            cameras = [camera]

        for cam in cameras:
            self.bg_frames[cam].clear()
            self.bg_model[cam] = None
            self.score_history[cam].clear()
            self.is_initialized[cam] = False

    def get_status(self) -> dict:
        """Get initialization status."""
        return {
            "left": {
                "initialized": self.is_initialized["left"],
                "bg_frames": len(self.bg_frames["left"]),
                "score_history": len(self.score_history["left"]),
            },
            "right": {
                "initialized": self.is_initialized["right"],
                "bg_frames": len(self.bg_frames["right"]),
                "score_history": len(self.score_history["right"]),
            },
        }


class SimpleCircleDetector:
    """
    Simplified circle detector that uses fixed background.

    This version builds background once and uses it throughout execution.
    Useful when the background is stable and doesn't change.

    Usage:
        detector = SimpleCircleDetector()

        # Initialization phase
        for i in range(30):
            left_img, right_img = robot.get_camera_images()
            detector.add_background_frame(left_img, "left")
            detector.add_background_frame(right_img, "right")

        detector.finalize_background()

        # Control loop
        left_img, right_img = robot.get_camera_images()
        left_circle = detector.detect(left_img, "left")
        right_circle = detector.detect(right_img, "right")
    """

    def __init__(self, k_std: float = 1.5, use_center_crop: bool = True):
        self.k_std = k_std
        self.use_center_crop = use_center_crop

        self.bg_frames = {"left": [], "right": []}
        self.bg_model = {"left": None, "right": None}
        self.threshold = {"left": None, "right": None}
        self.is_finalized = {"left": False, "right": False}

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale float32."""
        if img.ndim == 2:
            return img.astype(np.float32)

        if img.dtype == np.uint8 and img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return gray.astype(np.float32)

        raise ValueError(f"Unsupported image format: shape={img.shape}, dtype={img.dtype}")

    def add_background_frame(self, img: np.ndarray, camera: str = "left"):
        """Add frame to background collection."""
        gray = self._to_grayscale(img)
        self.bg_frames[camera].append(gray)

    def finalize_background(self, camera: Optional[str] = None):
        """
        Compute final background model from collected frames.

        Args:
            camera: "left", "right", or None (finalize both)
        """
        if camera is None:
            cameras = ["left", "right"]
        else:
            cameras = [camera]

        for cam in cameras:
            if len(self.bg_frames[cam]) == 0:
                raise ValueError(f"No background frames collected for {cam} camera")

            # Compute background
            self.bg_model[cam] = np.mean(self.bg_frames[cam], axis=0).astype(np.float32)

            # Compute initial threshold from background frames
            scores = []
            for frame in self.bg_frames[cam]:
                diff = np.abs(frame - self.bg_model[cam])
                if self.use_center_crop:
                    h, w = diff.shape
                    y0, y1 = int(h * 0.2), int(h * 0.8)
                    x0, x1 = int(w * 0.2), int(w * 0.8)
                    diff = diff[y0:y1, x0:x1]
                scores.append(float(np.mean(diff)))

            mean_score = np.mean(scores)
            std_score = np.std(scores) + 1e-8
            self.threshold[cam] = mean_score + self.k_std * std_score

            self.is_finalized[cam] = True

            print(f"[{cam}] Background finalized: threshold={self.threshold[cam]:.2f}")

    def detect(self, img: np.ndarray, camera: str = "left") -> int:
        """Detect circle in current frame."""
        if not self.is_finalized[camera]:
            raise ValueError(f"Background not finalized for {camera} camera. Call finalize_background() first.")

        gray = self._to_grayscale(img)
        bg = self.bg_model[camera]

        diff = np.abs(gray - bg)

        if self.use_center_crop:
            h, w = diff.shape
            y0, y1 = int(h * 0.2), int(h * 0.8)
            x0, x1 = int(w * 0.2), int(w * 0.8)
            diff = diff[y0:y1, x0:x1]

        score = float(np.mean(diff))

        return 1 if score >= self.threshold[camera] else 0


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Test with dummy images
    print("Testing CircleDetector...")

    detector = CircleDetector(num_bg_frames=10, k_std=1.5)

    # Simulate background frames (stable background)
    for i in range(10):
        bg_img = np.random.randint(100, 110, (480, 640, 3), dtype=np.uint8)
        detector.add_background_frame(bg_img, "left")
        detector.add_background_frame(bg_img, "right")

    print("Status:", detector.get_status())

    # Simulate detection frames
    # Frame 1: similar to background (no circle)
    test_img1 = np.random.randint(100, 110, (480, 640, 3), dtype=np.uint8)
    left_det1 = detector.detect(test_img1, "left")
    print(f"Frame 1 (no circle): left={left_det1}")

    # Frame 2: different from background (circle present)
    test_img2 = np.random.randint(150, 200, (480, 640, 3), dtype=np.uint8)
    left_det2 = detector.detect(test_img2, "left")
    print(f"Frame 2 (circle present): left={left_det2}")

    print("\nTesting SimpleCircleDetector...")
    simple_detector = SimpleCircleDetector(k_std=1.5)

    for i in range(10):
        bg_img = np.random.randint(100, 110, (480, 640, 3), dtype=np.uint8)
        simple_detector.add_background_frame(bg_img, "left")

    simple_detector.finalize_background("left")

    left_det = simple_detector.detect(test_img2, "left")
    print(f"Simple detector result: left={left_det}")
