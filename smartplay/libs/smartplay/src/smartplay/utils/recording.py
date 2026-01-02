from __future__ import annotations

import os
from typing import List

import imageio.v2 as imageio


def save_gif_from_paths(paths: List[str], out_path: str, duration: float = 0.2, loop: int = 0) -> str:
    """
    Save a list of image file paths as an animated GIF.

    Args:
        paths: List of paths to image files (PNG, JPG, etc.) in desired order.
        out_path: Destination GIF filepath.
        duration: Duration per frame in seconds.
        loop: GIF loop count (0 = infinite).

    Returns:
        The absolute path to the saved GIF.
    """
    if not paths:
        raise ValueError("No frame paths provided to save_gif_from_paths().")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames = []
    for p in paths:
        if os.path.exists(p):
            frames.append(imageio.imread(p))

    if not frames:
        raise ValueError("None of the provided frame paths exist on disk.")

    imageio.mimsave(out_path, frames, duration=duration, loop=loop)
    return os.path.abspath(out_path)
