import json

import pytest

from app import (
    _completed_epochs,
    _default_model_choice,
    _read_validation_threshold,
)


def test_completed_epochs_reads_latest_training_history(tmp_path):
    checkpoint = tmp_path / "model.pt"
    history = tmp_path / "model_history.csv"
    history.write_text("epoch,train_loss,validation_loss\n49,0.2,0.3\n", encoding="utf-8")

    assert _completed_epochs(checkpoint) == 49


def test_completed_epochs_reports_missing_history(tmp_path):
    checkpoint = tmp_path / "model.pt"

    assert _completed_epochs(checkpoint) is None


def test_default_model_choice_uses_validation_comparison(tmp_path):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps({"selection_split": "validation", "selected_model": "3d"}),
        encoding="utf-8",
    )

    assert _default_model_choice(comparison) == "3D U-Net"


def test_default_model_choice_rejects_test_split(tmp_path):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps({"selection_split": "test", "selected_model": "3d"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation"):
        _default_model_choice(comparison)


def test_threshold_default_uses_validation_report_only(tmp_path):
    report = tmp_path / "thresholds.json"
    report.write_text(
        json.dumps({"split": "validation", "selected_threshold": 0.65}),
        encoding="utf-8",
    )

    assert _read_validation_threshold(report, 0.5) == 0.65


def test_threshold_default_rejects_test_split(tmp_path):
    report = tmp_path / "thresholds.json"
    report.write_text(
        json.dumps({"split": "test", "selected_threshold": 0.65}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation"):
        _read_validation_threshold(report, 0.5)
