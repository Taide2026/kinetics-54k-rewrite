from rewrite.annotations.download import ensure_annotation_file
from rewrite.annotations.io import load_annotations, save_annotations
from rewrite.annotations.records import (
    get_assistant_text,
    get_system_text,
    get_user_text,
    get_video_ref,
    set_assistant_text,
)

__all__ = [
    "ensure_annotation_file",
    "load_annotations",
    "save_annotations",
    "get_assistant_text",
    "get_system_text",
    "get_user_text",
    "get_video_ref",
    "set_assistant_text",
]
