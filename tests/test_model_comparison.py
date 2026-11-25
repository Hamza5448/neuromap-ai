import json

import pytest

from compare_validation_models import _read_candidate, _select_model


def _write_threshold_report(path, split="validation"):
    path.write_text(
        json.dumps(
            {
                "split": split,
                "checkpoint": "candidate.pt",
                "selected_threshold": 0.6,
                "results": [
                    {
                        "threshold": 0.6,
                        "mean_metrics": {
                            "dice": 0.8,
                            "volumetric_error_ml": 2.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_model_comparison_selects_using_validation_dice_then_volume_error(tmp_path):
    baseline = {
        "name": "baseline",
        "mean_metrics": {"dice": 0.8, "volumetric_error_ml": 2.0},
    }
    candidate = {
        "name": "flair",
        "mean_metrics": {"dice": 0.8, "volumetric_error_ml": 1.0},
    }

    assert _select_model([baseline, candidate]) is candidate


def test_model_comparison_rejects_non_validation_reports(tmp_path):
    report_path = tmp_path / "test.json"
    _write_threshold_report(report_path, split="test")

    with pytest.raises(ValueError, match="not a validation-split"):
        _read_candidate("model", report_path)
