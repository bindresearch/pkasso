from pathlib import Path

import pytest

from pkasso import download_unipka_model as downloader
from pkasso.external.unipka.pka_predictor.api import _missing_checkpoint_message


def test_downloads_fold_checkpoints_to_required_output_folder(
    monkeypatch,
    tmp_path,
):
    captured = {}
    output_folder = tmp_path / "models"

    def mock_snapshot_download(**kwargs):
        captured.update(kwargs)
        checkpoint = Path(kwargs["local_dir"]) / "fold_3" / "checkpoint_best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        return str(kwargs["local_dir"])

    monkeypatch.setattr(downloader, "_snapshot_download", mock_snapshot_download)

    result = downloader.download_unipka_model(
        output_folder,
        revision="release-1",
        force_download=True,
    )

    assert result == (output_folder / "fold_3" / "checkpoint_best.pt",)
    assert captured == {
        "repo_id": "bindresearch/pkasso_unipka",
        "revision": "release-1",
        "local_dir": output_folder,
        "allow_patterns": ["fold_*/checkpoint_best.pt"],
        "force_download": True,
    }


def test_requires_output_folder():
    with pytest.raises(SystemExit) as exc_info:
        downloader.main([])

    assert exc_info.value.code == 2


def test_fails_if_repository_contains_no_expected_checkpoints(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        downloader,
        "_snapshot_download",
        lambda **kwargs: str(kwargs["local_dir"]),
    )

    with pytest.raises(RuntimeError, match="No Uni-pKa checkpoints"):
        downloader.download_unipka_model(tmp_path)


def test_missing_checkpoint_message_recommends_downloader_for_model_folder():
    message = _missing_checkpoint_message(
        Path("/models/with spaces/fold_3/checkpoint_best.pt"),
        Path("/models/with spaces"),
    )

    assert (
        "pkasso-download-unipka-model --output-folder '/models/with spaces'"
        in message
    )
