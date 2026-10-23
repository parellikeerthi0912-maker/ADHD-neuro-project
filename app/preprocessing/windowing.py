from typing import List, Tuple
import numpy as np

from app.config import (
    OVERLAP,
    REQUIRED_CHANNELS,
    STRIDE_SAMPLES,
    TARGET_FS,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
)
from app.utils.logging import log_pipeline_step


def create_eeg_windows(
    data: np.ndarray,
    fs: float = TARGET_FS,
    window_seconds: int = WINDOW_SECONDS,
    overlap: float = OVERLAP
) -> np.ndarray:
    """
    Segment preprocessed EEG into 10-second windows with 50% overlap.

    Parameters:
        data: np.ndarray of shape (time_samples, 19)
        fs: sampling rate (default 128 Hz)
        window_seconds: duration of each window (default 10s)
        overlap: overlap fraction (default 0.5)

    Returns:
        X: np.ndarray of shape (num_windows, 1280, 19) as float32
    """
    window_samples = int(round(window_seconds * fs))
    total_samples = len(data)

    if total_samples < window_samples:
        duration_sec = total_samples / fs
        raise ValueError(
            f"Insufficient EEG duration for model inference. "
            f"Received {duration_sec:.2f} seconds ({total_samples} samples); "
            f"the CNN-Transformer model requires at least {window_seconds} seconds "
            f"({window_samples} samples at {fs:g} Hz)."
        )

    step = max(1, int(round(window_samples * (1.0 - overlap))))

    windows: List[np.ndarray] = [
        data[start : start + window_samples]
        for start in range(0, total_samples - window_samples + 1, step)
    ]

    if not windows:
        raise ValueError(
            f"Could not generate complete windows for recording with {total_samples} samples."
        )

    # For massive multi-trial recordings (e.g. >150 windows = >12 minutes of EEG),
    # uniformly subsample to 150 windows to maintain fast response times and avoid memory pressure
    total_generated = len(windows)
    if len(windows) > 150:
        indices = np.linspace(0, total_generated - 1, 150, dtype=int)
        windows = [windows[i] for i in indices]
        log_pipeline_step(
            "WINDOW",
            f"Generated {total_generated} windows; subsampled to {len(windows)} representative windows."
        )
    else:
        log_pipeline_step("WINDOW", f"Generated {len(windows)} windows (step={step} samples).")

    return np.asarray(windows, dtype=np.float32)


def generate_window_metadata(
    num_windows: int,
    fs: float = TARGET_FS,
    window_seconds: int = WINDOW_SECONDS,
    overlap: float = OVERLAP
) -> List[Tuple[float, float]]:
    """
    Generate (start_seconds, end_seconds) timestamps for each segmented window.
    """
    window_samples = int(round(window_seconds * fs))
    step = max(1, int(round(window_samples * (1.0 - overlap))))
    metadata: List[Tuple[float, float]] = []

    for w_idx in range(num_windows):
        start_sample = w_idx * step
        end_sample = start_sample + window_samples
        start_sec = round(start_sample / fs, 2)
        end_sec = round(end_sample / fs, 2)
        metadata.append((start_sec, end_sec))

    return metadata
