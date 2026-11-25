import torch
from torch import nn

from src.neuromap.training import save_checkpoint, write_epoch_metrics


def test_checkpoint_preserves_best_and_latest_states(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    best_state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        model,
        optimizer,
        3,
        checkpoint_path,
        best_model_state_dict=best_state,
        best_validation_loss=0.25,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert checkpoint["epoch"] == 3
    assert checkpoint["best_validation_loss"] == 0.25
    for key, value in best_state.items():
        assert torch.equal(checkpoint["model_state_dict"][key], value)
    assert any(
        not torch.equal(checkpoint["last_model_state_dict"][key], value)
        for key, value in best_state.items()
    )


def test_epoch_history_writes_header_and_appends_rows(tmp_path):
    history_path = tmp_path / "reports" / "training_history.csv"

    write_epoch_metrics(history_path, 1, 1.25, 0.75)
    write_epoch_metrics(history_path, 2, 1.0, 0.5, append=True)

    assert history_path.read_text(encoding="utf-8").splitlines() == [
        "epoch,train_loss,validation_loss",
        "1,1.25,0.75",
        "2,1.0,0.5",
    ]
