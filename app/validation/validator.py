from pathlib import Path
from typing import List, Optional
import numpy as np

from app.config import (
    MODEL_CHANNEL_NAMES,
    REQUIRED_CHANNELS,
    TARGET_FS,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
)
from app.parsers import parse_eeg_file
from app.preprocessing import (
    clean_channel_name,
    detect_and_orient_time_dimension,
    map_and_reorder_channels,
)
from app.preprocessing.channels import ALIAS_LOOKUP
from app.schemas.responses import ValidationReport
from app.utils.logging import log_pipeline_step
from app.validation.bci_detector import (
    inspect_for_bci_competition_dataset,
    is_bci_archive_name,
    is_bci_recording_filename,
)

NON_EEG_BOTANICAL_KEYWORDS = {
    "sepal", "petal", "flower", "iris", "species", "versicolor",
    "virginica", "setosa", "sepallength", "sepalwidth", "petallength", "petalwidth"
}


def validate_eeg_file(
    path: Path,
    sampling_rate_override: Optional[float] = None,
    fallback_channel_names: Optional[List[str]] = None
) -> ValidationReport:
    """
    Validate an EEG file for compatibility with the CNN-Transformer + EEGNet + PSD/XGBoost
    ensemble ADHD classification pipeline without executing model inference.

    Validates:
      1. BCI Competition IV-2a/2b motor-imagery dataset detection (mandatory rejection).
      2. File integrity and parser validation.
      3. Non-EEG tabular dataset rejection (e.g. flower measurements, patient ID-only).
      4. Signal shape and orientation (must have 19 channels or identifiable 10-20 channels).
      5. Numerical integrity (no NaN, Inf, or flat zero-variance channels).
      6. Sampling rate validity.
      7. Recording duration (must be >= 10.0 seconds / 1280 samples; no padding).
      8. Channel naming and required 19 channels (missing channels rejected, never zero-padded).
    """
    issues: List[str] = []
    warnings: List[str] = []
    filename = path.name

    # 1. Early check for BCI Competition IV archive or file name
    if is_bci_archive_name(path) or is_bci_recording_filename(path):
        issues.append(
            "Unsupported EEG dataset: BCI Competition IV motor-imagery recordings are "
            "incompatible with the ADHD/Control classification model. No prediction was generated."
        )
        return ValidationReport(
            file=filename,
            valid=False,
            format=path.suffix.lower(),
            issues=issues,
            warnings=warnings,
            model_compatible=False
        )

    # 2. Parse recording
    try:
        recording = parse_eeg_file(
            path,
            sampling_rate_override=sampling_rate_override,
            fallback_channel_names=fallback_channel_names
        )
        warnings.extend(recording.warnings)
    except Exception as exc:
        exc_str = str(exc)
        issues.append(f"Parser error: {exc_str}")
        return ValidationReport(
            file=filename,
            valid=False,
            format=path.suffix.lower(),
            issues=issues,
            warnings=warnings,
            model_compatible=False
        )

    data = recording.data
    fs = recording.sampling_rate
    channel_names = recording.channel_names

    # 3. Content-based BCI Competition IV inspection (detects renamed files)
    annot_desc = recording.metadata.get("annotations") if hasattr(recording, "metadata") and recording.metadata else None
    bci_issue = inspect_for_bci_competition_dataset(
        path,
        channel_names=channel_names,
        annotations=annot_desc,
        sampling_rate=fs
    )
    if bci_issue:
        issues.append(bci_issue)

    # 4. Check for non-EEG tabular / flower measurement variables
    if channel_names:
        for ch in channel_names:
            ch_clean = ch.lower().replace("_", "").replace("-", "").replace(" ", "").replace(".", "")
            if any(kw in ch_clean for kw in NON_EEG_BOTANICAL_KEYWORDS):
                issues.append(
                    f"Invalid EEG file: Column '{ch}' corresponds to botanical/flower measurement data, "
                    f"not neurophysiological EEG signals."
                )
                break

    var_name = getattr(recording, "metadata", {}).get("variable_name", "") if hasattr(recording, "metadata") and recording.metadata else ""
    if var_name:
        var_lower = str(var_name).lower()
        if any(kw in var_lower for kw in NON_EEG_BOTANICAL_KEYWORDS):
            issues.append(
                f"Invalid EEG file: Variable '{var_name}' corresponds to non-EEG botanical/flower measurement data."
            )

    # 5. Check signal structure
    if data is None or not isinstance(data, np.ndarray) or data.size == 0 or data.ndim != 2:
        issues.append(f"EEG data must be a non-empty 2-D array. Received shape: {getattr(data, 'shape', None)}.")
        return ValidationReport(
            file=filename,
            valid=False,
            format=getattr(recording, "source_format", "UNKNOWN"),
            issues=issues,
            warnings=warnings,
            model_compatible=False
        )

    # 6. Orient data to (time, channels)
    data, _ = detect_and_orient_time_dimension(data, channel_names)
    samples_count, channels_count = data.shape

    if samples_count < 2:
        issues.append(f"Recording contains insufficient samples ({samples_count}). Minimum required is 2 samples.")

    # 7. Check numerical integrity
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        issues.append("INVALID_EEG_SIGNAL: Signal contains NaN or Inf values.")

    if samples_count > 1:
        stds = np.std(data, axis=0)
        if np.any(stds < 1e-10):
            issues.append("INVALID_EEG_SIGNAL: Signal contains flat channel(s) with zero variance (std < 1e-10).")

    # 8. Check sampling rate
    if fs is None or fs <= 0 or not np.isfinite(fs):
        issues.append(
            "Sampling rate could not be determined automatically. "
            "Please provide 'sampling_rate' parameter in the request."
        )

    # 9. Check channels and compatibility (Strict Policy: NEVER zero-pad or truncate silently)
    required_channels_available = False
    if channel_names:
        cleaned_input = [clean_channel_name(ch) for ch in channel_names]
        mapped_targets = {}
        ambiguous = []
        for raw_ch, clean_ch in zip(channel_names, cleaned_input):
            target = ALIAS_LOOKUP.get(clean_ch)
            if target:
                if target in mapped_targets:
                    ambiguous.append((target, mapped_targets[target], raw_ch))
                else:
                    mapped_targets[target] = raw_ch

        if ambiguous:
            issues.append(
                f"AMBIGUOUS_CHANNEL_MAPPING: Multiple channels mapped to same model electrode: {ambiguous}"
            )

        missing = [ch for ch in MODEL_CHANNEL_NAMES if ch not in mapped_targets]

        if not missing and not ambiguous:
            required_channels_available = True
            extra_count = channels_count - len(MODEL_CHANNEL_NAMES)
            if extra_count > 0:
                warnings.append(
                    f"Recording has {channels_count} channels. Found all {REQUIRED_CHANNELS} required channels; "
                    f"{extra_count} extra channels will be safely discarded."
                )
        elif channels_count == REQUIRED_CHANNELS and len(mapped_targets) < 5:
            # Exactly 19 channels with non-standard names
            required_channels_available = True
            warnings.append(
                f"Channel names ({channel_names[:5]}...) do not match 10-20 aliases, "
                f"but recording has exactly 19 channels. Assuming standard training order."
            )
        else:
            issues.append(
                f"MISSING_REQUIRED_CHANNELS: Missing required EEG channels: {missing}. "
                f"Detected: {list(mapped_targets.keys())}. Missing channels are not zero-padded."
            )
    else:
        # Unnamed channels
        if channels_count == REQUIRED_CHANNELS:
            required_channels_available = True
            warnings.append(
                "Recording has 19 channels without names; standard 10-20 layout in model order will be assumed."
            )
        else:
            issues.append(
                f"UNKNOWN_CHANNEL_ORDER: Recording has {channels_count} channels without names. "
                f"The trained model requires exactly {REQUIRED_CHANNELS} channels."
            )

    # 10. Check duration (Strict: Must be >= 10 seconds / 1280 samples)
    duration_sec: Optional[float] = None
    windows_count: Optional[int] = None
    if fs is not None and fs > 0:
        duration_sec = round(samples_count / fs, 2)
        if duration_sec < WINDOW_SECONDS or samples_count < 1280:
            issues.append(
                f"Recording duration ({duration_sec}s, {samples_count} samples) is shorter than the required "
                f"{WINDOW_SECONDS} seconds ({int(round(WINDOW_SECONDS * fs))} samples) for model windowing. "
                f"Recordings are not artificially padded."
            )
        else:
            effective_samples = int(round(duration_sec * TARGET_FS))
            step = int(round(WINDOW_SAMPLES * 0.5))
            windows_count = max(0, (effective_samples - WINDOW_SAMPLES) // step + 1)

    is_valid = len(issues) == 0
    is_compatible = is_valid and required_channels_available

    log_pipeline_step(
        "VALIDATE",
        f"Validation for '{filename}': valid={is_valid}, compatible={is_compatible}, issues={len(issues)}"
    )

    return ValidationReport(
        file=filename,
        valid=is_valid,
        format=recording.source_format,
        signal_shape=list(recording.original_shape),
        sampling_rate=fs,
        target_sampling_rate=TARGET_FS,
        channels_detected=channels_count,
        channel_names=channel_names,
        required_channels_available=required_channels_available,
        duration_seconds=duration_sec,
        windows_count=windows_count,
        model_compatible=is_compatible,
        issues=issues,
        warnings=warnings
    )
