"""Download a Hugging Face MusicGen checkpoint.

Examples::

    python scripts/utils/download_model.py \
        --model facebook/musicgen-small \
        --output-dir checkpoints/musicgen-small

The script uses ``huggingface_hub.snapshot_download`` so that model weights,
configuration, tokenizer and processor files are kept together in a portable
directory.  The dependency is intentionally imported only when the command is
run; install it with ``pip install huggingface_hub`` if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_MODELS = (
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
    "facebook/musicgen-large",
    "facebook/musicgen-melody",
)


def download_model(
    model: str,
    output_dir: str | Path,
    *,
    revision: str = "main",
    token: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
    local_files_only: bool = False,
) -> Path:
    """Download ``model`` and return its local directory.

    Args:
        model: Hugging Face repository id or a local model directory.
        output_dir: Destination directory.  Existing files are reused.
        revision: Hub branch, tag, or commit.
        token: Optional Hugging Face access token for gated/private models.
        allow_patterns: Optional file glob patterns, e.g. ``["*.json", "*.bin"]``.
        ignore_patterns: Optional file glob patterns to exclude.
        local_files_only: Resolve from the local Hugging Face cache only.
    """

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(model).expanduser()
    if source.exists():
        if not source.is_dir():
            raise NotADirectoryError(f"local model path is not a directory: {source}")
        return source.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "huggingface_hub is required to download a checkpoint. Install it "
            "with `pip install huggingface_hub`."
        ) from exc

    kwargs = {
        "repo_id": model,
        "revision": revision,
        "local_dir": str(destination),
        "local_files_only": local_files_only,
    }
    if token is not None:
        kwargs["token"] = token
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    if ignore_patterns is not None:
        kwargs["ignore_patterns"] = list(ignore_patterns)
    snapshot_download(**kwargs)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODELS[0],
        help="Hugging Face repository id or local checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which to store the complete checkpoint",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None, help="Hugging Face access token")
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="File glob to include; may be supplied more than once",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        dest="ignore_patterns",
        help="File glob to exclude; may be supplied more than once",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    destination = download_model(
        args.model,
        args.output_dir,
        revision=args.revision,
        token=args.token,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        local_files_only=args.local_files_only,
    )
    print(f"MusicGen checkpoint is available at: {destination}")


if __name__ == "__main__":
    main()
