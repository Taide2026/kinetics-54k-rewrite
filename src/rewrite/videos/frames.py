"""Uniform frame sampling from a video file with OpenCV."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sample_frames(video_path: str | Path, num_frames: int = 8) -> list[Image.Image]:
    """Sample ``num_frames`` evenly spaced RGB frames as PIL images."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Could not open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise IOError(f"Video reports no frames: {video_path}")
        indices = np.linspace(0, total - 1, num=min(num_frames, total)).astype(int)
        frames: list[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not frames:
            raise IOError(f"Could not decode any frames from: {video_path}")
        return frames
    finally:
        cap.release()
