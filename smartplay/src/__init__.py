"""Top-level package initializer to allow absolute imports like `from scripts...`.

This file ensures that when invoking `python -m src.scripts.main`, the `src` directory
is treated as a proper Python package root so submodules can be resolved.
"""

__all__ = []
