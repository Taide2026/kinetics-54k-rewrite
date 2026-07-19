"""End-to-end rewrite loop: annotations in, model captions out."""

import copy
from pathlib import Path

from tqdm import tqdm

from rewrite.annotations import (
    get_system_text,
    get_user_text,
    get_video_ref,
    load_annotations,
    save_annotations,
    set_assistant_text,
)
from rewrite.model import Captioner
from rewrite.videos import resolve_video_path, sample_frames


def caption_record(
    record: dict,
    captioner: Captioner,
    video_root: str | Path | None = None,
    num_frames: int = 8,
    max_new_tokens: int = 64,
) -> str:
    """Generate the replacement caption for one record."""
    ref = get_video_ref(record)
    if ref is None:
        raise ValueError("Record has no video reference")
    video_path = resolve_video_path(ref, video_root)
    frames = sample_frames(video_path, num_frames=num_frames)
    return captioner.caption(
        frames,
        user_text=get_user_text(record)
        or "Describe the main action happening in this video in one sentence.",
        system_text=get_system_text(record),
        max_new_tokens=max_new_tokens,
    )


def rewrite_annotations(
    annotation_file: str | Path,
    output_file: str | Path,
    captioner: Captioner,
    video_root: str | Path | None = None,
    limit: int | None = None,
    num_frames: int = 8,
    max_new_tokens: int = 64,
    save_every: int = 25,
) -> list[dict]:
    """Clone the annotation file with assistant captions rewritten by the model.

    Only the assistant ``content[].text`` of the first ``limit`` records is
    changed; every other field (and every record past ``limit``) is copied
    verbatim.
    """
    records = load_annotations(annotation_file)
    output = copy.deepcopy(records)
    todo = output if limit is None else output[:limit]

    for i, record in enumerate(tqdm(todo, desc="rewriting", unit="rec")):
        caption = caption_record(
            record,
            captioner,
            video_root=video_root,
            num_frames=num_frames,
            max_new_tokens=max_new_tokens,
        )
        set_assistant_text(record, caption)
        if save_every and (i + 1) % save_every == 0:
            save_annotations(output, output_file)

    save_annotations(output, output_file)
    return output
