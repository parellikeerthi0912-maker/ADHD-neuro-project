from pathlib import Path
from typing import List, Optional
import mne
import numpy as np

from app.parsers.base import EEGRecording
from app.preprocessing.channels import detect_and_orient_time_dimension
from app.utils.logging import log_pipeline_step
from app.validation.bci_detector import inspect_for_bci_competition_dataset

NON_EEG_CHANNEL_KEYWORDS = {
    "status", "trigger", "event", "events", "stim", "sync", "annotation", "annot",
    "eog", "ecg", "emg", "resp", "gsr", "eda", "temp", "photo", "accel", "gyro"
}


def read_gdf_file(
    path: Path,
    sampling_rate_override: Optional[float] = None
) -> EEGRecording:
    """
    Parse a General Data Format (.gdf) EEG file using MNE-Python.

    Features:
      - Mandatory check: Rejects BCI Competition IV-2a / IV-2b motor-imagery recordings.
      - Uses mne.io.read_raw_gdf(preload=True)
      - Excludes non-EEG channels (stim, status, trigger, EOG, ECG, etc.)
      - Extracts sampling rate and channel names
      - Ensures 2D (time, channels) orientation
    """
    warnings: List[str] = []

    try:
        raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    except Exception as exc:
        raise ValueError(f"Could not read GDF file '{path.name}': {exc}")

    # Check for BCI Competition IV-2a / IV-2b motor imagery signature in file and annotations
    annot_desc = list(raw.annotations.description) if raw.annotations else None
    sfreq = float(raw.info.get("sfreq", 0))
    bci_rejection = inspect_for_bci_competition_dataset(
        path,
        channel_names=raw.ch_names,
        annotations=annot_desc,
        sampling_rate=sfreq
    )
    if bci_rejection:
        raise ValueError(bci_rejection)

    # Pick EEG channels
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) < 2:
        # If channels weren't explicitly tagged as EEG in GDF header,
        # filter out obvious non-EEG channels by name
        filtered_picks = []
        for i, ch_name in enumerate(raw.ch_names):
            clean = str(ch_name).strip().lower()
            if not any(k in clean for k in NON_EEG_CHANNEL_KEYWORDS):
                filtered_picks.append(i)

        picks = filtered_picks if len(filtered_picks) >= 2 else list(range(len(raw.ch_names)))
        if len(picks) < 2:
            raise ValueError(f"GDF file '{path.name}' contains fewer than 2 usable EEG channels.")
        warnings.append(
            f"GDF channels were not explicitly typed as EEG in header; using {len(picks)} signal channels."
        )

    # raw.get_data() returns (channels, time) -> transpose to (time, channels)
    data = raw.get_data(picks=picks).T.astype(np.float64)
    channel_names = [raw.ch_names[i] for i in picks]

    detected_fs = sampling_rate_override or sfreq

    # Ensure orientation is (time, channels)
    data, transposed = detect_and_orient_time_dimension(data, channel_names)
    if transposed:
        warnings.append("Transposed GDF data array to (time, channels).")

    log_pipeline_step(
        "PARSER",
        f"Parsed GDF: shape {data.shape}, channels={len(channel_names)}, fs={detected_fs:g} Hz"
    )

    return EEGRecording(
        data=data,
        sampling_rate=detected_fs,
        channel_names=channel_names,
        source_format="GDF",
        file_name=path.name,
        original_shape=data.shape,
        warnings=warnings,
        metadata={
            "sfreq": sfreq,
            "n_channels": len(channel_names),
            "raw_channel_names": raw.ch_names
        }
    )
