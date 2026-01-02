"""Utility helpers for multimodal (vision) support (clean version)."""

from __future__ import annotations

import os
import logging
import json
import shutil
from typing import List, Optional, Dict, Tuple, Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

try:  # Pillow optional
    from PIL import Image  # type: ignore
    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover
    _PIL_AVAILABLE = False
    Image = None  # type: ignore


def capture_vision_frame(
    env,
    env_name: str,
    state,
    episode_number: int | None,
    step_number: int,
    base_dir: str = "vision_frames",
    override_output_dir: str | None = None,
) -> Optional[List[str]]:
    """Obtain a vision frame for the current step.

    Order:
      1. Hanoi3Disk mapping (pre-rendered assets) for classic variants.
      2. MessengerL1 on-demand render (cache only for Image / ImageLanguage reps).
      3. Generic env.render / ndarray fallback persisted as PNG.
    """
    # 1. Hanoi3Disk mapping
    if env_name.startswith("Hanoi3Disk") and not env_name.endswith("Image"):
        try:
            internal_state = getattr(env, "current_state", None)
            if internal_state and isinstance(internal_state, tuple) and len(internal_state) == 3:
                mapped = _get_hanoi3_image_for_state(internal_state)  # type: ignore
                if mapped:
                    copied = _ensure_episode_copy(
                        mapped,
                        override_output_dir,
                        env_name,
                        step_number,
                    )
                    return [copied]
        except Exception as e:  # pragma: no cover
            logging.debug(f"Hanoi3Disk mapping error: {e}")

    # 2. MessengerL1 unified vision path
    if env_name.lower().startswith("messengerl1"):
        try:
            base_env = getattr(env, 'unwrapped', env)
            repr_name = getattr(base_env, 'representation', 'Default')
            last_obs = getattr(base_env, 'last_obs', None)
            if last_obs is not None and hasattr(base_env, '_render_image_grid'):
                try:
                    img = base_env._render_image_grid(last_obs)
                    if isinstance(img, str):
                        img_abs = os.path.abspath(img)
                        if os.path.exists(img_abs):
                            # Always update cache to allow external tools to reference latest
                            try:
                                setattr(base_env, '_last_image_path', img_abs)
                            except Exception:  # pragma: no cover
                                pass
                            copied_path = _ensure_episode_copy(
                                img_abs,
                                override_output_dir,
                                env_name,
                                step_number,
                            )
                            if step_number == 0:
                                logging.info(f"[MessengerVision] First frame captured: {copied_path} (repr={repr_name})")
                            else:
                                logging.debug(f"[MessengerVision] Step frame captured (step={step_number}, repr={repr_name}): {copied_path}")
                            return [copied_path]
                        else:
                            logging.debug(f"[MessengerVision] Render path missing on disk (step={step_number}): {img_abs}")
                except Exception as re:  # pragma: no cover
                    logging.debug(f"[MessengerVision] Render error (step={step_number}): {re}")
            # fallback: if we have a cached last image path from earlier step, return it (better than None)
            cached = getattr(base_env, '_last_image_path', None)
            if isinstance(cached, str) and os.path.exists(cached):
                copied_cached = _ensure_episode_copy(
                    cached,
                    override_output_dir,
                    env_name,
                    step_number,
                )
                logging.debug(f"[MessengerVision] Using cached last frame (step={step_number}): {copied_cached}")
                return [copied_cached]
            if step_number == 0:
                logging.info(f"[MessengerVision] No initial frame produced (env={env_name}, repr={repr_name}).")
        except Exception as e:  # pragma: no cover
            logging.debug(f"[MessengerVision] Messenger vision helper error: {e}")

    # 3. Generic fallback
    frame = None
    try:
        frame = env.render(mode="rgb_array")  # type: ignore[attr-defined]
    except Exception:
        frame = None

    if frame is None and isinstance(state, np.ndarray) and state.ndim in (2, 3):
        frame = state

    if frame is None:
        return None

    if not _PIL_AVAILABLE:
        logging.warning("PIL not available; cannot save vision frame. Using text-only.")
        return None

    try:
        arr = frame
        if not isinstance(arr, np.ndarray):
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 4:  # drop alpha
            arr = arr[:, :, :3]
        ep = episode_number or 0
        if override_output_dir:
            frame_dir = override_output_dir
        else:
            frame_dir = os.path.join(base_dir, env_name, f"episode_{ep}")
        os.makedirs(frame_dir, exist_ok=True)
        frame_path = os.path.join(frame_dir, f"step_{step_number}.png")
        Image.fromarray(arr).save(frame_path)
        return [frame_path]
    except Exception as e:  # pragma: no cover
        logging.warning(f"Vision frame save failed (env={env_name}, step={step_number}): {e}")
        return None


__all__ = ["capture_vision_frame"]


def _ensure_episode_copy(
    frame_path: str,
    override_output_dir: Optional[str],
    env_name: str,
    step_number: int,
) -> str:
    """Copy source sprites into the per-episode directory so pruning never deletes originals."""
    abs_src = os.path.abspath(frame_path)
    if not override_output_dir:
        return abs_src

    abs_dir = os.path.abspath(override_output_dir)
    try:
        os.makedirs(abs_dir, exist_ok=True)
    except Exception:  # pragma: no cover
        pass

    try:
        if os.path.commonpath([abs_src, abs_dir]) == abs_dir:
            return abs_src
    except ValueError:
        # Paths on different drives; treat as unique
        pass

    safe_name = f"{env_name.lower()}_step_{step_number:04d}_{Path(abs_src).name}"
    dest_path = os.path.join(abs_dir, safe_name)

    try:
        shutil.copy2(abs_src, dest_path)
        return dest_path
    except Exception as copy_err:  # pragma: no cover
        logging.warning(
            f"Failed to copy frame '{abs_src}' into episode directory '{abs_dir}': {copy_err}. Using source path."
        )
        return abs_src


# ---------------------------------------------------------------------------
# Episode-level helper lifecycle for vision (Messenger + generic envs)
# ---------------------------------------------------------------------------
def initialize_episode_vision(
    *,
    env,
    env_name: str,
    vision: bool,
    episode_number: int | None,
    initial_state,
) -> Tuple[Optional[str], List[str], Optional[List[str]]]:
    """Set up per-episode vision capture.

    Returns: (episode_render_dir, collected_frames_list, initial_image_paths)
    """
    if not vision:
        return None, [], None

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    episode_render_dir: Optional[str] = None

    if env_name.lower().startswith("messengerl1"):
        try:
            import messenger  # type: ignore
            messenger_images_dir = os.path.join(os.path.dirname(messenger.__file__), "images")
            base_renders_dir = os.path.join(messenger_images_dir, "renders")
            episode_render_dir = os.path.join(base_renders_dir, timestamp)
            os.makedirs(episode_render_dir, exist_ok=True)
            # inform internal renderer (wrapper and base env if present)
            for target in (env, getattr(env, 'unwrapped', None)):
                if target is None:
                    continue
                try:
                    setattr(target, "_frame_output_dir", Path(episode_render_dir))
                except Exception:
                    pass
            logging.info(f"[MessengerVision] Using timestamped render directory: {episode_render_dir}")
        except Exception as e:  # pragma: no cover
            logging.warning(f"Failed to initialize Messenger render directory: {e}")
            episode_render_dir = None
    else:
        # generic fallback
        episode_render_dir = os.path.join("process_results", "data", "renders", env_name, timestamp)
        try:
            os.makedirs(episode_render_dir, exist_ok=True)
        except Exception as e:  # pragma: no cover
            logging.warning(f"Could not create render directory {episode_render_dir}: {e}")
            episode_render_dir = None

    # Initial frame capture (step 0)
    initial_paths: Optional[List[str]] = None
    try:
        initial_paths = capture_vision_frame(
            env=env,
            env_name=env_name,
            state=initial_state,
            episode_number=episode_number,
            step_number=0,
            override_output_dir=episode_render_dir,
        )
    except Exception as e:  # pragma: no cover
        logging.debug(f"Initial vision frame capture failed: {e}")

    collected_frames: List[str] = []
    if initial_paths:
        collected_frames.extend(initial_paths)

    return episode_render_dir, collected_frames, initial_paths


def capture_post_step_frame(
    *,
    env,
    env_name: str,
    state,
    episode_number: int | None,
    step_number: int,
    episode_render_dir: Optional[str],
    collected_frames: List[str],
) -> Optional[List[str]]:
    """Capture a post-step frame (step_number already incremented for HUD)."""
    try:
        post_paths = capture_vision_frame(
            env=env,
            env_name=env_name,
            state=state,
            episode_number=episode_number,
            step_number=step_number,
            override_output_dir=episode_render_dir,
        )
        if post_paths:
            collected_frames.extend(post_paths)
            return post_paths
    except Exception as e:  # pragma: no cover
        logging.debug(f"Post-step vision frame capture failed (step={step_number}): {e}")
    return None


def finalize_episode_vision(
    *,
    env,
    env_name: str,
    episode_number: int | None,
    collected_vision_frames: List[str],
    vision: bool,
    episode_render_dir: Optional[str],
    tag: str = "vision",
    create_local_gif: bool = True,
) -> None:
    """Finalize vision for an episode: central GIF + optional local GIF in render dir."""
    if not vision:
        return
    # Central logging GIF (W&B)
    maybe_generate_env_gif(
        env=env,
        env_name=env_name,
        episode_number=episode_number,
        collected_vision_frames=collected_vision_frames,
        vision=vision,
        tag=tag,
    )
    # Local GIF
    if create_local_gif and episode_render_dir and collected_vision_frames:
        if not _PIL_AVAILABLE:
            logging.debug("PIL not available; skipping local GIF synthesis.")
            return
        try:
            frames = [Image.open(p) for p in sorted(collected_vision_frames) if os.path.exists(p)]
            if frames:
                gif_path = os.path.join(episode_render_dir, "episode.gif")
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=200,
                    loop=0,
                )
                logging.info(f"Saved episode GIF to {gif_path}")
        except Exception as e:  # pragma: no cover
            logging.debug(f"Local GIF synthesis failed: {e}")

__all__.extend([
    "initialize_episode_vision",
    "capture_post_step_frame",
    "finalize_episode_vision",
])


def maybe_generate_env_gif(
    *,
    env,
    env_name: str,
    episode_number: int | None,
    collected_vision_frames: list[str],
    vision: bool,
    tag: str = "vision",
) -> None:
    """Generate a GIF for Hanoi or Messenger classic runs if enabled."""
    if not vision or not collected_vision_frames:
        return
    try:
        base_env = getattr(env, 'unwrapped', env)
        record_gif = getattr(base_env, 'record_gif', False)
        if not record_gif:
            return
        gif_duration = getattr(base_env, 'gif_duration', 0.2)
        record_dir = getattr(base_env, 'record_dir', None)
        is_hanoi = env_name.startswith('Hanoi3Disk')
        is_messenger = (env_name.lower().startswith('messengerl1') and 'image' not in env_name.lower())
        if not (is_hanoi or is_messenger):
            return
        from scripts.utils.wandb_logging import generate_and_log_episode_gif  # type: ignore
        generate_and_log_episode_gif(
            episode_number=episode_number or 0,
            frame_paths=collected_vision_frames,
            env_name=env_name,
            gif_duration=gif_duration,
            record_dir=record_dir,
            tag=tag,
        )
    except Exception as e:  # pragma: no cover
        logging.debug(f"maybe_generate_env_gif failed (non-fatal): {e}")

__all__.extend(['maybe_generate_env_gif'])


# Hanoi3Disk helpers -------------------------------------------------
_HANOI3_METADATA_CACHE: Optional[Dict[Tuple[int, int, int], str]] = None


def _load_hanoi3_metadata() -> None:
    """Lazy load metadata.json building state->image mapping for 3-disk Hanoi."""
    global _HANOI3_METADATA_CACHE
    if _HANOI3_METADATA_CACHE is not None:
        return
    try:
        import smartplay.hanoi as hanoi_pkg  # type: ignore
        pkg_dir = os.path.dirname(hanoi_pkg.__file__)
        images_dir = os.path.join(pkg_dir, "images")
        meta_path = os.path.join(images_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping: Dict[Tuple[int, int, int], str] = {}
        for entry in data.get("states", []):
            vec = entry.get("vector")
            fname = entry.get("filename")
            if (
                isinstance(vec, list)
                and len(vec) == 3
                and all(isinstance(v, int) for v in vec)
                and isinstance(fname, str)
            ):
                abs_path = os.path.join(images_dir, fname)
                if os.path.exists(abs_path):
                    mapping[tuple(vec)] = abs_path
        _HANOI3_METADATA_CACHE = mapping if mapping else None
    except Exception as e:  # pragma: no cover
        logging.debug(f"Failed to load Hanoi3 metadata: {e}")


def _get_hanoi3_image_for_state(state: Tuple[int, int, int]) -> Optional[str]:
    _load_hanoi3_metadata()
    if _HANOI3_METADATA_CACHE is None:
        return None
    return _HANOI3_METADATA_CACHE.get(tuple(state))

__all__.extend(["_get_hanoi3_image_for_state"])
