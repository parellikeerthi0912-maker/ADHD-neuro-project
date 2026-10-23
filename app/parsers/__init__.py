from pathlib import Path
from typing import List, Optional

from app.config import SUPPORTED_EXTENSIONS
from app.parsers.base import EEGRecording
from app.parsers.csv_parser import read_csv_file
from app.parsers.edf_parser import read_edf_eeg_file
from app.parsers.gdf_parser import read_gdf_file
from app.parsers.mat_parser import extract_channel_names_from_companion_file, read_mat_file
from app.parsers.numpy_parser import read_numpy_eeg_file
from app.parsers.tsv_parser import read_tsv_file
from app.parsers.zip_parser import extract_zip_archive_safely


def parse_eeg_file(
    path: Path,
    sampling_rate_override: Optional[float] = None,
    fallback_channel_names: Optional[List[str]] = None
) -> EEGRecording:
    """
    Universal EEG parser entry point.
    Dispatches to format-specific parser based on file extension.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported EEG format '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".mat":
        return read_mat_file(path, sampling_rate_override=sampling_rate_override, fallback_channel_names=fallback_channel_names)

    if ext == ".gdf":
        return read_gdf_file(path, sampling_rate_override=sampling_rate_override)

    if ext == ".csv":
        return read_csv_file(path, sampling_rate_override=sampling_rate_override)

    if ext in {".tsv", ".txt"}:
        return read_tsv_file(path, sampling_rate_override=sampling_rate_override)

    if ext in {".npy", ".npz"}:
        return read_numpy_eeg_file(path, sampling_rate_override=sampling_rate_override)

    if ext in {".edf", ".fif", ".set"}:
        return read_edf_eeg_file(path, sampling_rate_override=sampling_rate_override)

    raise ValueError(f"No parser registered for extension: {ext}")


__all__ = [
    "EEGRecording",
    "parse_eeg_file",
    "read_mat_file",
    "read_gdf_file",
    "read_csv_file",
    "read_tsv_file",
    "read_numpy_eeg_file",
    "read_edf_eeg_file",
    "extract_zip_archive_safely",
    "extract_channel_names_from_companion_file",
]
