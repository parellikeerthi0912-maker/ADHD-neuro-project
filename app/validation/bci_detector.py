"""
BCI Competition IV-2a & IV-2b Motor-Imagery Dataset Detection Module.

BCI Competition IV-2a and IV-2b are motor-imagery EEG benchmark datasets, NOT
ADHD-versus-Control clinical datasets. Recordings from these datasets (whether in original
ZIP archives, extracted GDF files, renamed recordings, or MAT files) must be rejected
before reaching neural network and ensemble inference.
"""

from pathlib import Path
import re
from typing import List, Optional, Set, Union
import numpy as np

# Known BCI Competition IV filename signatures
BCI_FILENAME_PATTERNS = [
    re.compile(r"bciciv", re.IGNORECASE),
    re.compile(r"bci[_-]?iv", re.IGNORECASE),
    re.compile(r"bci.*competition", re.IGNORECASE),
    # IV-2a standard file naming: A01T, A01E .. A09E
    re.compile(r"^[a-zA-Z]?0[1-9][te]\.gdf$", re.IGNORECASE),
    # IV-2b standard file naming: B0101T, B0102T .. B0905E
    re.compile(r"^[b-zB-Z]?0[1-9]0[1-5][te]\.gdf$", re.IGNORECASE),
]

# Standardized Graz BCI motor-imagery event codes
# 768 (0x0300) = Start of trial
# 769 (0x0301) = Cue onset left hand (class 1)
# 770 (0x0302) = Cue onset right hand (class 2)
# 771 (0x0303) = Cue onset foot (class 3)
# 772 (0x0304) = Cue onset tongue (class 4)
# 1023 (0x03FF) = Rejected trial
# 32766 (0x7FFE) = Start of a new run
GRAZ_MOTOR_IMAGERY_EVENT_CODES: Set[str] = {
    "768", "769", "770", "771", "772", "1023", "32766",
    "0x0300", "0x0301", "0x0302", "0x0303", "0x0304", "0x03ff", "0x7ffe"
}

# Key cue event codes that definitively define a motor-imagery task
GRAZ_CUE_CODES: Set[str] = {"769", "770", "771", "772", "0x0301", "0x0302", "0x0303", "0x0304"}

REJECTION_MESSAGE = (
    "Unsupported EEG dataset: BCI Competition IV motor-imagery recordings are "
    "incompatible with the ADHD/Control classification model. No prediction was generated."
)


def is_bci_archive_name(archive_path_or_name: Union[Path, str]) -> bool:
    """Check if ZIP archive filename indicates BCI Competition IV data."""
    name = Path(archive_path_or_name).name.lower()
    return any(p.search(name) for p in BCI_FILENAME_PATTERNS[:3])


def is_bci_recording_filename(file_path_or_name: Union[Path, str]) -> bool:
    """Check if recording filename matches BCI Competition IV standard naming conventions."""
    name = Path(file_path_or_name).name.lower()
    return any(p.search(name) for p in BCI_FILENAME_PATTERNS)


def detect_bci_competition_in_annotations(descriptions: List[Union[str, int]]) -> bool:
    """
    Detect Graz motor-imagery event codes in GDF/MNE annotations or event tables.
    Matches event codes 769, 770, 771, 772 (cues for left hand, right hand, foot, tongue).
    """
    if not descriptions:
        return False
    desc_strings = {str(d).strip().lower() for d in descriptions}
    # Check for direct cue codes or combination of trial start (768) + cue
    if any(code in desc_strings for code in GRAZ_CUE_CODES):
        return True
    if "768" in desc_strings and any(c in desc_strings for c in ["276", "277", "1072", "32766"]):
        return True
    return False


def detect_bci_competition_channel_layout(
    channel_names: List[str],
    sampling_rate: Optional[float] = None
) -> Optional[str]:
    """
    Inspect electrode montage for characteristic BCI Competition IV signatures:
    - IV-2a: 22 EEG channels (including FC3, FC1, FCz, CP3, CP1, etc.) + 3 EOG channels (25 channels total)
             or channel names prefixed with EEG-0..EEG-21 + EOG at 250 Hz
    - IV-2b: 3 bipolar channels (C3, Cz, C4) + 3 EOG channels (total 6 channels) at 250 Hz
    """
    if not channel_names:
        return None

    ch_upper = [c.upper().strip() for c in channel_names]
    ch_clean = [
        re.sub(r"^(EEG|EOG)[:\-_]?", "", c).strip()
        for c in ch_upper
    ]

    has_eog = any("EOG" in c for c in ch_upper)

    # 1. BCI Competition IV-2a Layout: 22 EEG + 3 EOG (25 channels total)
    iv2a_characteristic = {"FC3", "FC1", "FCZ", "FC2", "FC4", "CP3", "CP1", "CPZ", "CP2", "CP4"}
    if len(channel_names) in (25, 26) and (has_eog or iv2a_characteristic.intersection(set(ch_clean))):
        return "BCI Competition IV-2a (22 EEG + 3 EOG montage)"

    # Check for numbered EEG channels from IV-2a GDF export (e.g. EEG-Fz, EEG-0 .. EEG-21)
    if any("EEG-0" in c or "EEG-1" in c for c in ch_upper) and has_eog:
        return "BCI Competition IV-2a (numbered GDF electrode montage)"

    # 2. BCI Competition IV-2b Layout: exactly 3 EEG (C3, Cz, C4) + 3 EOG at 250 Hz
    if len(channel_names) in (6, 7):
        bipolar_c = {"C3", "CZ", "C4"}.issubset(set(ch_clean))
        if bipolar_c and (has_eog or (sampling_rate is not None and abs(sampling_rate - 250.0) < 1.0)):
            return "BCI Competition IV-2b (3 bipolar EEG + 3 EOG montage)"

    return None


def inspect_for_bci_competition_dataset(
    file_path: Path,
    channel_names: Optional[List[str]] = None,
    annotations: Optional[List[Union[str, int]]] = None,
    sampling_rate: Optional[float] = None
) -> Optional[str]:
    """
    Comprehensive multi-layer inspection for BCI Competition IV-2a & IV-2b datasets.

    Returns:
        rejection_reason: str if file belongs to BCI Competition IV, or None if clean.
    """
    # Layer 1: Filename / Archive check
    if is_bci_recording_filename(file_path):
        return f"{REJECTION_MESSAGE} (Identified by BCI Competition IV filename signature '{file_path.name}')"

    # Layer 2: Event code / Task annotation inspection (detects renamed files)
    if annotations and detect_bci_competition_in_annotations(annotations):
        return f"{REJECTION_MESSAGE} (Identified by Graz motor-imagery event codes in recording annotations)"

    # Layer 3: Channel layout signature inspection (detects renamed files without annotations)
    if channel_names:
        layout_detected = detect_bci_competition_channel_layout(channel_names, sampling_rate)
        if layout_detected:
            return f"{REJECTION_MESSAGE} (Identified by {layout_detected})"

    return None
