"""Compatibility entry point for :mod:`download_model`."""

try:
    from .download_model import DEFAULT_MODELS, build_parser, download_model, main
except ImportError:  # supports ``python scripts/utils/download_musicgen.py``
    from download_model import DEFAULT_MODELS, build_parser, download_model, main

__all__ = ["DEFAULT_MODELS", "build_parser", "download_model", "main"]


if __name__ == "__main__":
    main()
