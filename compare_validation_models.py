"""Rank trained models using validation reports, never test metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_candidate(name: str, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("split") != "validation":
        raise ValueError(f"{report_path} is not a validation-split report.")
    threshold = float(report["selected_threshold"])
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Missing threshold results in {report_path}.")
    selected = next(
        (
            result
            for result in results
            if math.isclose(float(result["threshold"]), threshold, abs_tol=1e-8)
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            f"Selected threshold {threshold} is not in the results of {report_path}."
        )
    metrics = selected.get("mean_metrics")
    if not isinstance(metrics, dict) or not {
        "dice",
        "volumetric_error_ml",
    }.issubset(metrics):
        raise ValueError(f"Missing selection metrics in {report_path}.")
    return {
        "name": name,
        "threshold_report": str(report_path),
        "checkpoint": report.get("checkpoint"),
        "threshold": threshold,
        "mean_metrics": metrics,
    }


def _select_model(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one validation candidate is required.")
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["mean_metrics"]["dice"]),
            -float(candidate["mean_metrics"]["volumetric_error_ml"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=REPORT",
        help="Validation-threshold report; may be supplied multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/task_a_validation_model_comparison.json"),
    )
    args = parser.parse_args()
    candidates = []
    seen_names: set[str] = set()
    for specification in args.candidate:
        if "=" not in specification:
            raise ValueError("Each candidate must have the form NAME=REPORT.")
        name, raw_path = specification.split("=", 1)
        if not name or not raw_path or name in seen_names:
            raise ValueError(f"Invalid or duplicate candidate name: {name!r}.")
        report_path = Path(raw_path)
        if not report_path.is_file():
            raise FileNotFoundError(f"Validation report not found: {report_path}")
        seen_names.add(name)
        candidates.append(_read_candidate(name, report_path))

    selected = _select_model(candidates)
    report = {
        "selection_split": "validation",
        "selection_rule": "Highest mean patient Dice; ties use lower mean volume error.",
        "selected_model": selected["name"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_threshold": selected["threshold"],
        "models": sorted(
            candidates,
            key=lambda candidate: (
                float(candidate["mean_metrics"]["dice"]),
                -float(candidate["mean_metrics"]["volumetric_error_ml"]),
            ),
            reverse=True,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Selected by validation only: {selected['name']} "
        f"(threshold={selected['threshold']:.2f}, "
        f"mean_dice={selected['mean_metrics']['dice']:.5f})"
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
