"""Resolve dataset video references to local files.

Annotation records store extensionless references like
``kinetic600/land sailing/QszpApNTHuQ_000038_000048``; the actual file lives
under a local video root with a ``.mp4`` extension.
"""

import os
from pathlib import Path


def resolve_video_path(ref: str, video_root: str | Path | None = None) -> Path:
    """Try the reference as-is, with ``.mp4`` appended, and under ``video_root``."""
    candidates = [ref, ref + ".mp4"]
    if video_root is not None:
        candidates += [
            os.path.join(video_root, ref),
            os.path.join(video_root, ref) + ".mp4",
        ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return Path(candidate)
    raise FileNotFoundError(
        f"Video {ref!r} not found (tried: {', '.join(str(c) for c in candidates)}). "
        "Pass --video-root pointing at your local Kinetics-600 directory, "
        "or fetch the files first with `uv run fetch-videos`."
    )
