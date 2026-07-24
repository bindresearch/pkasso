"""Download the Uni-pKa model checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


UNIPKA_REPO_ID = "bindresearch/pkasso_unipka"
CHECKPOINT_PATTERN = "fold_*/checkpoint_best.pt"


def _snapshot_download(**kwargs: object) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Downloading the Uni-pKa weights requires the optional Hugging Face "
            "dependency. Install it with `python -m pip install 'pkasso[unipka]'`."
        ) from exc

    return snapshot_download(**kwargs)


def download_unipka_model(
    output_folder: str | Path,
    *,
    revision: str = "main",
    force_download: bool = False,
) -> tuple[Path, ...]:
    """Download all Uni-pKa fold checkpoints into ``output_folder``."""

    destination = Path(output_folder).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    _snapshot_download(
        repo_id=UNIPKA_REPO_ID,
        revision=revision,
        local_dir=destination,
        allow_patterns=[CHECKPOINT_PATTERN],
        force_download=force_download,
    )

    checkpoints = tuple(sorted(destination.glob(CHECKPOINT_PATTERN)))
    if not checkpoints:
        raise RuntimeError(
            f"No Uni-pKa checkpoints were downloaded from {UNIPKA_REPO_ID!r}. "
            f"Expected files matching {CHECKPOINT_PATTERN!r}."
        )
    return checkpoints


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Uni-pKa model checkpoints from Hugging Face. The output "
            "folder can then be supplied to pKasso with --unipka-model-folder."
        )
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        type=Path,
        help="Destination folder for the fold_N/checkpoint_best.pt files.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face repository revision to download (default: main).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the checkpoint files again even if they are cached.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        checkpoints = download_unipka_model(
            args.output_folder,
            revision=args.revision,
            force_download=args.force,
        )
    except (OSError, RuntimeError) as exc:
        _parser().exit(1, f"error: {exc}\n")

    print(f"Downloaded {len(checkpoints)} Uni-pKa checkpoint(s) to {args.output_folder.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
