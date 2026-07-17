"""Accessors for the kinetics_54K multimodal chat record format.

A record looks like::

    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": [
            {"type": "video", "video": "kinetic600/<label>/<video_id>"},
            {"type": "text", "text": "..."}
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
      ],
      "label": "...", "video_id": "...", ...
    }

``content`` may be a plain string or a list of typed items depending on the
role/variant, so every accessor handles both shapes.
"""


def _content_items(message: dict) -> list:
    content = message.get("content", [])
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content


def _first_text(message: dict) -> str | None:
    for item in _content_items(message):
        if isinstance(item, dict) and item.get("type") == "text":
            return item.get("text")
    return None


def _messages_with_role(record: dict, role: str) -> list[dict]:
    return [m for m in record.get("messages", []) if m.get("role") == role]


def get_system_text(record: dict) -> str | None:
    for msg in _messages_with_role(record, "system"):
        text = _first_text(msg)
        if text is not None:
            return text
    return None


def get_user_text(record: dict) -> str | None:
    for msg in _messages_with_role(record, "user"):
        text = _first_text(msg)
        if text is not None:
            return text
    return None


def get_video_ref(record: dict) -> str | None:
    """Return the video reference, e.g. ``kinetic600/land sailing/QszpApNTHuQ_000038_000048``."""
    for msg in record.get("messages", []):
        for item in _content_items(msg):
            if isinstance(item, dict) and item.get("type") == "video":
                for key in ("video", "path", "url"):
                    if key in item:
                        return item[key]
    return None


def get_assistant_text(record: dict) -> str | None:
    for msg in _messages_with_role(record, "assistant"):
        text = _first_text(msg)
        if text is not None:
            return text
    return None


def set_assistant_text(record: dict, new_text: str) -> None:
    """Replace the assistant caption in place. Touches nothing else."""
    for msg in _messages_with_role(record, "assistant"):
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = new_text
            return
        for item in content or []:
            if isinstance(item, dict) and item.get("type") == "text":
                item["text"] = new_text
                return
    raise ValueError("Record has no assistant text content to rewrite")
