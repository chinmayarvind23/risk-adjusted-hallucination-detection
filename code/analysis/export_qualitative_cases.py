from __future__ import annotations

"""
Export qualitative case slices from a frozen detector bundle and a standardized CSV.

This script uses a frozen detector bundle and an already-standardized CSV.
Its job is:

1. loads an already-frozen detector, calibrator, and abstention threshold
2. applies that frozen scoring logic to an already-standardized CSV
3. writes a few CSV files that are useful for qualitative inspection

Typical uses:

1. In-domain PHANTOM inspection
   - bundle: results/calibration/phantom_4000_frozen_bundle.json
   - csv: data/full_run/splits/phantom_4000_test_standardized.csv
2. In-domain WikiQA inspection
   - bundle: results/wikiqa/calibration/wikiqa_1300_frozen_bundle.json
   - csv: data/wiki_qa/splits/wikiqa_1300_test_standardized.csv
3. PHANTOM to WikiQA transfer inspection
   - bundle: results/calibration/phantom_4000_frozen_bundle.json
   - csv: data/wiki_qa/train/wikiqa_1300_feature_table_standardized.csv
4. WikiQA to PHANTOM transfer inspection
   - bundle: results/wikiqa/calibration/wikiqa_1300_frozen_bundle.json
   - csv: data/full_run/qwen3_8b_k5_phantom_4000_feature_table_deduped_standardized.csv

The output files are meant for manual review:

- all_cases_with_risk.csv
  Full table with model scores and keep or abstain decisions.
- missed_hallucinations.csv
  Unsupported answers that still looked safe enough to keep.
- correctly_abstained_unsupported.csv
  Unsupported answers that the abstention rule rejected.
- safe_supported_kept.csv
  Supported answers with low calibrated risk.
- unnecessary_abstentions.csv
  Supported answers that the abstention rule rejected.
- qualitative_summary.txt
  Compact counts for the same run.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


def _set_csv_field_limit() -> None:
    """Allow large text fields such as long retrieved context passages."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _load_json(path: Path) -> Dict:
    """Load a JSON artifact such as a frozen detector bundle."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read the standardized feature CSV into memory."""
    _set_csv_field_limit()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sigmoid_scalar(value: float) -> float:
    """Match the sigmoid used for raw unsupported risk."""
    clipped = max(-50.0, min(50.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def _require_keys(mapping: Dict, keys: Sequence[str], label: str) -> None:
    """Fail fast when a required JSON field is missing."""
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")


def _validate_required_columns(rows: Sequence[Dict[str, str]], required_columns: Sequence[str]) -> List[str]:
    """Return the CSV header and check that required columns are present."""
    if not rows:
        raise ValueError("Input CSV has no rows.")

    header = list(rows[0].keys())
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {', '.join(missing)}")
    return header


def _parse_float(value: str, column_name: str, row_index: int) -> float:
    """Convert one CSV cell into float with a useful error message."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_index + 1} has a non-numeric value in column '{column_name}': {value!r}"
        ) from exc


def _parse_int_label(value: str, row_index: int) -> int:
    """Convert the binary judge label into int and validate its range."""
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_index + 1} has a non-integer judge_binary_label: {value!r}"
        ) from exc

    if label not in (0, 1):
        raise ValueError(
            f"Row {row_index + 1} has judge_binary_label={label}, expected 0 or 1."
        )
    return label


def _apply_calibration(method: str, fit_parameters: Dict, raw_logit: float, raw_prob: float) -> float:
    """Apply the frozen calibration rule exactly as the repo does."""
    normalized_method = method.strip().lower()

    if normalized_method == "platt scaling":
        _require_keys(fit_parameters, ["coef", "intercept"], "bundle['calibration']['fit_parameters']")
        calibrated = _sigmoid_scalar(
            float(fit_parameters["coef"]) * raw_logit + float(fit_parameters["intercept"])
        )
        return max(0.0, min(1.0, calibrated))

    if normalized_method == "isotonic regression":
        _require_keys(fit_parameters, ["x_thresholds", "y_thresholds"], "bundle['calibration']['fit_parameters']")
        x_thresholds = np.array(fit_parameters["x_thresholds"], dtype=np.float64)
        y_thresholds = np.array(fit_parameters["y_thresholds"], dtype=np.float64)
        if x_thresholds.size == 0 or y_thresholds.size == 0:
            raise ValueError("Isotonic calibration thresholds are empty.")
        if x_thresholds.size != y_thresholds.size:
            raise ValueError("Isotonic calibration thresholds have mismatched lengths.")

        calibrated = float(
            np.interp(
                raw_prob,
                x_thresholds,
                y_thresholds,
                left=float(y_thresholds[0]),
                right=float(y_thresholds[-1]),
            )
        )
        return max(0.0, min(1.0, calibrated))

    raise ValueError(f"Unsupported calibration method in frozen bundle: {method}")


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    """Write UTF-8 CSV output."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(path: Path, counts: Dict[str, int]) -> None:
    """Write a compact text summary of key qualitative counts."""
    lines = [
        f"total rows: {counts['total_rows']}",
        f"kept rows: {counts['kept_rows']}",
        f"abstained rows: {counts['abstained_rows']}",
        f"unsupported kept: {counts['unsupported_kept']}",
        f"unsupported abstained: {counts['unsupported_abstained']}",
        f"supported kept: {counts['supported_kept']}",
        f"supported abstained: {counts['supported_abstained']}",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _sort_rows(rows: Iterable[Dict[str, object]], key_name: str, reverse: bool) -> List[Dict[str, object]]:
    """Sort rows by one numeric field."""
    return sorted(rows, key=lambda row: float(row[key_name]), reverse=reverse)


def main() -> None:
    """Apply a frozen bundle to a standardized CSV and export qualitative slices."""
    parser = argparse.ArgumentParser(
        description="Export qualitative case CSVs from a frozen detector bundle and a standardized feature CSV."
    )
    parser.add_argument("--bundle", required=True, help="Path to a frozen detector bundle JSON")
    parser.add_argument("--csv", required=True, help="Path to an already-standardized feature CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to save qualitative outputs")
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of rows to keep in each qualitative slice CSV except all_cases_with_risk.csv",
    )
    parser.add_argument(
        "--show-columns",
        action="store_true",
        help="Print CSV column names and exit before writing outputs",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")

    bundle_path = Path(args.bundle)
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    bundle = _load_json(bundle_path)
    rows = _read_csv_rows(csv_path)

    if args.show_columns:
        header = [] if not rows else list(rows[0].keys())
        print("\n".join(header))
        return

    _require_keys(bundle, ["feature_columns", "detector", "calibration", "abstention_policy"], "bundle")
    _require_keys(bundle["detector"], ["coefficients", "intercept"], "bundle['detector']")
    _require_keys(bundle["calibration"], ["method", "fit_parameters"], "bundle['calibration']")
    _require_keys(bundle["abstention_policy"], ["frozen_risk_threshold"], "bundle['abstention_policy']")

    feature_columns = list(bundle["feature_columns"])
    detector_coefficients = bundle["detector"]["coefficients"]
    detector_intercept = float(bundle["detector"]["intercept"])
    calibration_method = str(bundle["calibration"]["method"])
    calibration_fit_parameters = bundle["calibration"]["fit_parameters"]
    frozen_risk_threshold = float(bundle["abstention_policy"]["frozen_risk_threshold"])
    classification_threshold = float(
        bundle["abstention_policy"].get("classification_threshold_for_metrics", 0.5)
    )

    for feature_name in feature_columns:
        if feature_name not in detector_coefficients:
            raise ValueError(f"Detector coefficients are missing feature '{feature_name}'.")

    required_columns = [
        "question",
        "context",
        "served_answer",
        "ground_truth_answer",
        "judge_binary_label",
        *feature_columns,
    ]
    original_fieldnames = _validate_required_columns(rows, required_columns)

    enriched_rows: List[Dict[str, object]] = []
    for row_index, row in enumerate(rows):
        # Rebuild the detector logit in the exact feature order stored in the
        # frozen bundle so per-example analysis matches the saved model.
        raw_logit = detector_intercept
        for feature_name in feature_columns:
            raw_logit += _parse_float(row[feature_name], feature_name, row_index) * float(
                detector_coefficients[feature_name]
            )

        # Raw risk comes directly from the detector. Calibrated risk is the
        # quantity used by the abstention policy.
        raw_risk = _sigmoid_scalar(raw_logit)
        calibrated_risk = _apply_calibration(
            calibration_method,
            calibration_fit_parameters,
            raw_logit,
            raw_risk,
        )
        keep_decision = "keep" if calibrated_risk <= frozen_risk_threshold else "abstain"
        judge_binary_label = _parse_int_label(row["judge_binary_label"], row_index)

        enriched_row: Dict[str, object] = dict(row)
        enriched_row["raw_logit"] = raw_logit
        enriched_row["raw_risk"] = raw_risk
        enriched_row["calibrated_risk"] = calibrated_risk
        enriched_row["frozen_risk_threshold"] = frozen_risk_threshold
        enriched_row["classification_threshold_for_metrics"] = classification_threshold
        enriched_row["keep_decision"] = keep_decision
        enriched_row["judge_binary_label"] = judge_binary_label
        enriched_rows.append(enriched_row)

    all_fieldnames = list(original_fieldnames)
    for new_column in [
        "raw_logit",
        "raw_risk",
        "calibrated_risk",
        "frozen_risk_threshold",
        "classification_threshold_for_metrics",
        "keep_decision",
    ]:
        if new_column not in all_fieldnames:
            all_fieldnames.append(new_column)

    unsupported_kept = [
        row for row in enriched_rows if int(row["judge_binary_label"]) == 1 and row["keep_decision"] == "keep"
    ]
    unsupported_abstained = [
        row for row in enriched_rows if int(row["judge_binary_label"]) == 1 and row["keep_decision"] == "abstain"
    ]
    supported_kept = [
        row for row in enriched_rows if int(row["judge_binary_label"]) == 0 and row["keep_decision"] == "keep"
    ]
    supported_abstained = [
        row for row in enriched_rows if int(row["judge_binary_label"]) == 0 and row["keep_decision"] == "abstain"
    ]

    # These sorted slices are the main qualitative views used in the report.
    # They separate missed unsupported answers, successful rejections, clean
    # kept answers, and over-cautious abstentions.
    missed_hallucinations = _sort_rows(unsupported_kept, "calibrated_risk", reverse=False)[: args.top_k]
    correctly_abstained_unsupported = _sort_rows(unsupported_abstained, "calibrated_risk", reverse=True)[: args.top_k]
    safe_supported_kept = _sort_rows(supported_kept, "calibrated_risk", reverse=False)[: args.top_k]
    unnecessary_abstentions = _sort_rows(supported_abstained, "calibrated_risk", reverse=True)[: args.top_k]

    counts = {
        "total_rows": len(enriched_rows),
        "kept_rows": sum(1 for row in enriched_rows if row["keep_decision"] == "keep"),
        "abstained_rows": sum(1 for row in enriched_rows if row["keep_decision"] == "abstain"),
        "unsupported_kept": len(unsupported_kept),
        "unsupported_abstained": len(unsupported_abstained),
        "supported_kept": len(supported_kept),
        "supported_abstained": len(supported_abstained),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "all_cases_with_risk.csv": output_dir / "all_cases_with_risk.csv",
        "missed_hallucinations.csv": output_dir / "missed_hallucinations.csv",
        "correctly_abstained_unsupported.csv": output_dir / "correctly_abstained_unsupported.csv",
        "safe_supported_kept.csv": output_dir / "safe_supported_kept.csv",
        "unnecessary_abstentions.csv": output_dir / "unnecessary_abstentions.csv",
        "qualitative_summary.txt": output_dir / "qualitative_summary.txt",
    }

    _write_csv(output_paths["all_cases_with_risk.csv"], enriched_rows, all_fieldnames)
    _write_csv(output_paths["missed_hallucinations.csv"], missed_hallucinations, all_fieldnames)
    _write_csv(
        output_paths["correctly_abstained_unsupported.csv"],
        correctly_abstained_unsupported,
        all_fieldnames,
    )
    _write_csv(output_paths["safe_supported_kept.csv"], safe_supported_kept, all_fieldnames)
    _write_csv(output_paths["unnecessary_abstentions.csv"], unnecessary_abstentions, all_fieldnames)
    _write_summary(output_paths["qualitative_summary.txt"], counts)

    print(f"Saved qualitative exports to: {output_dir}")
    print(f"Wrote {len(enriched_rows)} rows to: {output_paths['all_cases_with_risk.csv'].name}")
    print(f"Wrote {len(missed_hallucinations)} rows to: {output_paths['missed_hallucinations.csv'].name}")
    print(
        f"Wrote {len(correctly_abstained_unsupported)} rows to: "
        f"{output_paths['correctly_abstained_unsupported.csv'].name}"
    )
    print(f"Wrote {len(safe_supported_kept)} rows to: {output_paths['safe_supported_kept.csv'].name}")
    print(f"Wrote {len(unnecessary_abstentions)} rows to: {output_paths['unnecessary_abstentions.csv'].name}")
    print(f"Wrote summary to: {output_paths['qualitative_summary.txt'].name}")
    print("")
    print(f"total rows: {counts['total_rows']}")
    print(f"kept rows: {counts['kept_rows']}")
    print(f"abstained rows: {counts['abstained_rows']}")
    print(f"unsupported kept: {counts['unsupported_kept']}")
    print(f"unsupported abstained: {counts['unsupported_abstained']}")
    print(f"supported kept: {counts['supported_kept']}")
    print(f"supported abstained: {counts['supported_abstained']}")
    print("")
    print("Example commands:")
    print("PHANTOM in-domain:")
    print(
        "python code\\analysis\\export_qualitative_cases.py "
        "--bundle results\\calibration\\phantom_4000_frozen_bundle.json "
        "--csv data\\full_run\\splits\\phantom_4000_test_standardized.csv "
        "--output-dir results\\qualitative\\phantom_in_domain"
    )
    print("WikiQA in-domain:")
    print(
        "python code\\analysis\\export_qualitative_cases.py "
        "--bundle results\\wikiqa\\calibration\\wikiqa_1300_frozen_bundle.json "
        "--csv data\\wiki_qa\\splits\\wikiqa_1300_test_standardized.csv "
        "--output-dir results\\qualitative\\wikiqa_in_domain"
    )
    print("PHANTOM -> WikiQA transfer:")
    print(
        "python code\\analysis\\export_qualitative_cases.py "
        "--bundle results\\calibration\\phantom_4000_frozen_bundle.json "
        "--csv data\\wiki_qa\\train\\wikiqa_1300_feature_table_standardized.csv "
        "--output-dir results\\qualitative\\phantom_to_wikiqa"
    )
    print("WikiQA -> PHANTOM transfer:")
    print(
        "python code\\analysis\\export_qualitative_cases.py "
        "--bundle results\\wikiqa\\calibration\\wikiqa_1300_frozen_bundle.json "
        "--csv data\\full_run\\qwen3_8b_k5_phantom_4000_feature_table_deduped_standardized.csv "
        "--output-dir results\\qualitative\\wikiqa_to_phantom"
    )


if __name__ == "__main__":
    main()
