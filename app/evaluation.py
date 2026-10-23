"""
Evaluation Workflow Module for ADHD EEG Datasets.

Provides evaluation workflows for IEEE DataPort (121 recordings: 61 ADHD, 60 Control),
Mendeley Data (4 recordings: 2 ADHD, 2 Control), and generic EEG dataset directories or archives.
Calculates ground-truth reconciliation, prediction distributions, and evaluation metrics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.api.routes_predict import predict_single_eeg_file
from app.parsers.zip_parser import extract_zip_archive_safely
from app.schemas.responses import FilePredictionResult
from app.utils.files import infer_ground_truth_label
from app.utils.logging import log_pipeline_step, logger


def evaluate_eeg_recordings(
    recording_paths: List[Path],
    dataset_name: str,
    expected_total: int,
    expected_adhd: int,
    expected_control: int,
    base_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluate a collection of EEG recording files against established reference labels.

    Reconciles dataset inventory, processes each recording independently through the
    validated pipeline, predicts ADHD or Control, compares with verified reference labels,
    and calculates evaluation metrics.
    """
    results: List[FilePredictionResult] = []

    for path in sorted(recording_paths):
        display_name = str(path.relative_to(base_dir)) if base_dir and path.is_relative_to(base_dir) else path.name
        res = predict_single_eeg_file(path, display_name=display_name)
        results.append(res)

    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "failed"]

    files_found = len(recording_paths)
    files_evaluated = len(successful)
    files_failed = len(failed)

    pred_adhd = sum(1 for r in successful if r.prediction == "ADHD")
    pred_control = sum(1 for r in successful if r.prediction == "Control")
    unclassified = files_found - files_evaluated

    # Reference ground truth counts
    ref_adhd = sum(1 for r in results if r.reference_label == "ADHD")
    ref_control = sum(1 for r in results if r.reference_label == "Control")

    has_gt = (ref_adhd + ref_control) > 0

    tp = sum(1 for r in successful if r.reference_label == "ADHD" and r.prediction == "ADHD")
    fn = sum(1 for r in successful if r.reference_label == "ADHD" and r.prediction == "Control")
    fp = sum(1 for r in successful if r.reference_label == "Control" and r.prediction == "ADHD")
    tn = sum(1 for r in successful if r.reference_label == "Control" and r.prediction == "Control")

    correct_count = tp + tn
    eval_total = tp + fn + fp + tn

    accuracy_str = "Unavailable"
    accuracy_val: Optional[float] = None
    if has_gt and eval_total > 0:
        accuracy_val = round((tp + tn) / eval_total, 4)
        accuracy_str = f"{accuracy_val * 100:.1f}%"

    precision_val = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall_val = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1_val = round(2 * precision_val * recall_val / (precision_val + recall_val), 4) if (precision_val + recall_val) > 0 else 0.0

    # Status determination
    if files_found == 0:
        status = "Not evaluated (dataset files missing)"
    elif files_evaluated < expected_total:
        status = f"Partial evaluation ({files_evaluated}/{expected_total} recordings evaluated)"
    else:
        status = "Fully evaluated"

    ref_counts_str = f"{expected_adhd} ADHD / {expected_control} Control (Total: {expected_total})"

    report = {
        "dataset_name": dataset_name,
        "expected_reference_counts": ref_counts_str,
        "expected_total": expected_total,
        "expected_adhd": expected_adhd,
        "expected_control": expected_control,
        "files_found": files_found,
        "files_evaluated": files_evaluated,
        "files_failed": files_failed,
        "unclassified": unclassified,
        "predicted_adhd": pred_adhd,
        "predicted_control": pred_control,
        "reference_adhd": ref_adhd,
        "reference_control": ref_control,
        "correct_predictions": correct_count if has_gt else None,
        "accuracy_str": accuracy_str,
        "accuracy": accuracy_val,
        "precision": precision_val if has_gt else None,
        "recall": recall_val if has_gt else None,
        "f1_score": f1_val if has_gt else None,
        "confusion_matrix": {
            "ADHD": {"ADHD": tp, "Control": fn},
            "Control": {"ADHD": fp, "Control": tn}
        } if has_gt else None,
        "status": status,
        "per_recording_results": [
            {
                "file": r.filename,
                "status": r.status,
                "prediction": r.prediction or "Unavailable",
                "probability": r.adhd_probability,
                "reference_label": r.reference_label or "Unknown",
                "is_correct": r.is_correct,
                "error": r.error
            }
            for r in results
        ]
    }

    return report


def evaluate_mendeley_benchmark(fixtures_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Evaluate the Mendeley Data 4-recording benchmark cohort.
    Expected: 4 recordings (2 ADHD: FADHD, MADHD; 2 Control: FC, MC).
    """
    if fixtures_dir is None:
        fixtures_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

    mendeley_files = ["FADHD.mat", "MADHD.mat", "FC.mat", "MC.mat"]
    paths = [fixtures_dir / fname for fname in mendeley_files if (fixtures_dir / fname).exists()]

    report = evaluate_eeg_recordings(
        recording_paths=paths,
        dataset_name="Mendeley Data",
        expected_total=4,
        expected_adhd=2,
        expected_control=2,
        base_dir=fixtures_dir
    )

    report["warning"] = (
        "SMALL SAMPLE SIZE WARNING: The Mendeley evaluation contains only 4 recordings. "
        "Results are too small to establish clinical accuracy or generalizability."
    )
    return report


def evaluate_ieee_benchmark(dataset_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Evaluate the IEEE DataPort ADHD dataset.
    Expected: 121 recordings (61 ADHD, 60 Control).
    """
    paths: List[Path] = []
    base_dir = None

    if dataset_dir and dataset_dir.exists():
        base_dir = dataset_dir
        if dataset_dir.suffix.lower() == ".zip":
            import tempfile
            tmp = Path(tempfile.mkdtemp(prefix="ieee_eval_"))
            eeg_files, _ = extract_zip_archive_safely(dataset_dir, tmp)
            paths = eeg_files
            base_dir = tmp
        else:
            paths = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".mat", ".csv", ".edf", ".bdf", ".npy", ".npz", ".tsv"}]
    else:
        # Search fixtures for IEEE files (v1p.mat, v21p.mat) if full dataset directory is absent
        fixtures_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
        paths = [p for p in [fixtures_dir / "v1p.mat", fixtures_dir / "v21p.mat"] if p.exists()]
        base_dir = fixtures_dir

    report = evaluate_eeg_recordings(
        recording_paths=paths,
        dataset_name="IEEE DataPort",
        expected_total=121,
        expected_adhd=61,
        expected_control=60,
        base_dir=base_dir
    )

    return report


def generate_acceptance_report_table(reports: List[Dict[str, Any]]) -> str:
    """
    Generate Markdown formatted acceptance report summary table as required by Section 11.G.
    """
    lines = [
        "### Benchmark Dataset Acceptance Summary Report",
        "",
        "| Dataset | Expected reference counts | Files found | Files evaluated | Predicted ADHD | Predicted control | Accuracy | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for r in reports:
        row = (
            f"| {r['dataset_name']} | {r['expected_reference_counts']} | "
            f"{r['files_found']} | {r['files_evaluated']} | {r['predicted_adhd']} | "
            f"{r['predicted_control']} | {r['accuracy_str']} | {r['status']} |"
        )
        lines.append(row)

    return "\n".join(lines)


def run_full_evaluation() -> Tuple[List[Dict[str, Any]], str]:
    """Run full evaluation suite on available benchmark datasets and return reports and markdown table."""
    mendeley_report = evaluate_mendeley_benchmark()
    ieee_report = evaluate_ieee_benchmark()

    reports = [ieee_report, mendeley_report]
    table = generate_acceptance_report_table(reports)
    return reports, table


if __name__ == "__main__":
    reports, table = run_full_evaluation()
    print("\n" + table + "\n")
    for r in reports:
        print(f"--- {r['dataset_name']} Detailed Results ---")
        print(f"Files found: {r['files_found']} / {r['expected_total']}")
        print(f"Evaluated: {r['files_evaluated']}, Failed: {r['files_failed']}, Unclassified: {r['unclassified']}")
        print(f"Predictions: {r['predicted_adhd']} ADHD, {r['predicted_control']} Control")
        if r.get("accuracy") is not None:
            print(f"Accuracy: {r['accuracy_str']} ({r['correct_predictions']}/{r['files_evaluated']} correct)")
            print(f"Confusion Matrix: {r['confusion_matrix']}")
        if "warning" in r:
            print(f"Note: {r['warning']}")
        print()
