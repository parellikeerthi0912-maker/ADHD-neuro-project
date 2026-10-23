import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from app.config import (
    CHANNEL_ALIASES,
    MODEL_CHANNEL_NAMES,
    REQUIRED_CHANNELS,
)


def clean_channel_name(name: Any) -> str:
    """
    Normalize channel names conservatively:
    - Uppercase
    - Remove prefixes like 'EEG', 'CH', 'CHANNEL'
    - Remove suffixes like '-REF', '-LE', '_REF', '_LE'
    - Strip punctuation and spaces

    Examples:
        'EEG Fp1-REF' -> 'FP1'
        'Fp-1'         -> 'FP1'
        't7'           -> 'T7'
        'EEG CZ'       -> 'CZ'
        'P7-REF'       -> 'P7'
    """
    text = str(name).strip().upper()
    text = re.sub(r"^(EEG|CH|CHANNEL)\s*[-_.:]?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"[-_](REF|LE|AVG|GROUND|GND)$", "", text, flags=re.IGNORECASE).strip()
    text = text.replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
    return text


def build_alias_lookup() -> Dict[str, str]:
    """
    Build a conservative lookup table from normalized alias string
    to canonical target model channel name with exact casing (e.g. 'FP1' -> 'Fp1', 'T7' -> 'T3').
    """
    lookup: Dict[str, str] = {}
    canonical_map = {clean_channel_name(c): c for c in MODEL_CHANNEL_NAMES}
    for canonical_key, aliases in CHANNEL_ALIASES.items():
        target = canonical_map.get(clean_channel_name(canonical_key), canonical_key)
        lookup[clean_channel_name(canonical_key)] = target
        for alias in aliases:
            lookup[clean_channel_name(alias)] = target
    return lookup


ALIAS_LOOKUP: Dict[str, str] = build_alias_lookup()


def detect_and_orient_time_dimension(
    data: np.ndarray,
    channel_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, bool]:
    """
    Detect whether data is (time, channels) or (channels, time) and orient to (time, channels).

    Returns:
        oriented_data: np.ndarray of shape (samples, channels)
        was_transposed: bool
    """
    if data.ndim != 2:
        raise ValueError(f"EEG data must be 2-dimensional. Received shape: {data.shape}")

    rows, cols = data.shape

    # If channel names are provided, use their count unambiguously
    if channel_names:
        if len(channel_names) == cols:
            return data, False
        if len(channel_names) == rows:
            return data.T, True

    # If one dimension is exactly 19 channels
    if rows == REQUIRED_CHANNELS and cols > rows:
        return data.T, True
    if cols == REQUIRED_CHANNELS and rows > cols:
        return data, False

    # Heuristic: time is almost always the longer dimension in continuous EEG recordings
    if rows < cols and rows <= 256:
        return data.T, True

    return data, False


def map_and_reorder_channels(
    data: np.ndarray,
    channel_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Map, validate, and reorder input channels into the exact 19 canonical model channels:
    ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

    Strict Policy:
      - 19 channels matching canonical or aliases -> reordered to exact training contract.
      - 32/64 channels containing the required 19 -> required 19 extracted by name and reordered; extras discarded.
      - Missing required channels -> REJECT (never zero-pad physiological channels).
      - Ambiguous channel mapping (e.g. both T3 and T7 present) -> REJECT.
      - Unnamed channels:
          - exactly 19 -> accepted with warning (assuming training order).
          - > 19 unnamed -> REJECT (cannot safely determine electrodes).
          - < 19 unnamed -> REJECT.
      - Flat / NaN / Inf signals -> REJECT.

    Returns:
        reordered_data: np.ndarray of shape (time, 19)
        final_channel_names: List[str] matching MODEL_CHANNEL_NAMES
        warnings: List[str]
    """
    # 1. Validate data array
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError(f"Expected 2-D EEG data array, got shape {getattr(data, 'shape', None)}")

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("INVALID_EEG_SIGNAL: Input EEG recording contains NaN or Inf values.")

    time_samples, num_channels = data.shape
    warnings_list: List[str] = []

    # 2. Check for flat channels (std < 1e-10)
    channel_stds = np.std(data, axis=0)
    flat_indices = np.where(channel_stds < 1e-10)[0]
    if len(flat_indices) > 0:
        flat_labels = (
            [channel_names[i] for i in flat_indices if i < len(channel_names)]
            if channel_names
            else [f"col_{i}" for i in flat_indices]
        )
        raise ValueError(
            f"INVALID_EEG_SIGNAL: Flat channel(s) detected with near-zero variance (std < 1e-10): {flat_labels}."
        )

    # CASE A: No channel names provided
    if not channel_names:
        if num_channels == REQUIRED_CHANNELS:
            warnings_list.append(
                "Channel labels were not provided; assuming columns are already in the model's "
                "19-channel training order."
            )
            return data.astype(np.float64), MODEL_CHANNEL_NAMES.copy(), warnings_list

        if num_channels > REQUIRED_CHANNELS:
            raise ValueError(
                f"UNKNOWN_CHANNEL_ORDER: {num_channels} channels detected but channel labels are unavailable, "
                f"so required model channels cannot be identified safely without guessing."
            )

        # num_channels < REQUIRED_CHANNELS
        raise ValueError(
            f"MISSING_REQUIRED_CHANNELS: Recording contains only {num_channels} unnamed channels; "
            f"the trained model requires all {REQUIRED_CHANNELS} channels."
        )

    # CASE B: Channel names are provided
    if len(channel_names) != num_channels:
        raise ValueError(
            f"Channel name count ({len(channel_names)}) does not match data column count ({num_channels})."
        )

    # Map each input column to canonical target channel
    canonical_to_input_idx: Dict[str, int] = {}
    ambiguous_mappings: Dict[str, List[str]] = {}

    for idx, raw_ch in enumerate(channel_names):
        clean_ch = clean_channel_name(raw_ch)
        target = ALIAS_LOOKUP.get(clean_ch)
        if target:
            if target in canonical_to_input_idx:
                prev_idx = canonical_to_input_idx[target]
                prev_name = channel_names[prev_idx]
                if target not in ambiguous_mappings:
                    ambiguous_mappings[target] = [prev_name]
                ambiguous_mappings[target].append(raw_ch)
            else:
                canonical_to_input_idx[target] = idx

    if ambiguous_mappings:
        conflict_desc = "; ".join(
            f"Target '{tgt}' matched multiple input channels: {ch_list}"
            for tgt, ch_list in ambiguous_mappings.items()
        )
        raise ValueError(f"AMBIGUOUS_CHANNEL_MAPPING: Ambiguous channel definitions detected: {conflict_desc}")

    # Check for missing required channels
    missing = [ch for ch in MODEL_CHANNEL_NAMES if ch not in canonical_to_input_idx]
    if missing:
        detected = [ch for ch in MODEL_CHANNEL_NAMES if ch in canonical_to_input_idx]
        extra = [
            channel_names[i]
            for i in range(num_channels)
            if i not in canonical_to_input_idx.values()
        ]
        raise ValueError(
            f"MISSING_REQUIRED_CHANNELS: Required EEG channel(s) missing: {missing}. "
            f"Detected matched: {detected}. Extra channels: {extra}."
        )

    # Extract exactly the required 19 channels in canonical training order
    reordered = np.zeros((time_samples, REQUIRED_CHANNELS), dtype=np.float64)
    for target_idx, target_name in enumerate(MODEL_CHANNEL_NAMES):
        src_col = canonical_to_input_idx[target_name]
        reordered[:, target_idx] = data[:, src_col]

    matched_count = len(MODEL_CHANNEL_NAMES)
    extra_count = num_channels - matched_count
    if extra_count > 0:
        warnings_list.append(
            f"Extracted {matched_count} required channels from {num_channels} available channels. "
            f"Safely discarded {extra_count} non-target channels."
        )

    return reordered, MODEL_CHANNEL_NAMES.copy(), warnings_list
