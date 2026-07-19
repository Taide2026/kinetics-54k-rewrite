"""Orchestrate one rewrite across multiple model replicas (one per GPU).

Spawns ``python -m rewrite.worker`` subprocesses over disjoint contiguous
record slices, tracks their JSONL outputs for a combined progress bar, then
merges the captions into the cloned annotation file.
"""

import copy
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

from rewrite.annotations import load_annotations, save_annotations, set_assistant_text


def _split(total: int, parts: int) -> list[tuple[int, int]]:
    """Contiguous (offset, count) chunks covering range(total)."""
    base, extra = divmod(total, parts)
    chunks, offset = [], 0
    for p in range(parts):
        count = base + (1 if p < extra else 0)
        if count:
            chunks.append((offset, count))
        offset += count
    return chunks


def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def rewrite_parallel(
    annotation_file: str | Path,
    output_file: str | Path,
    model_id: str,
    devices: list[str],
    video_root: str | Path = "videos",
    limit: int | None = None,
    num_frames: int = 8,
    max_new_tokens: int = 64,
    dtype: str = "bfloat16",
) -> None:
    records = load_annotations(annotation_file)
    total = len(records) if limit is None else min(limit, len(records))
    chunks = _split(total, len(devices))

    with tempfile.TemporaryDirectory(prefix="rewrite-parts-") as tmp:
        procs: list[tuple[subprocess.Popen, Path, Path, str]] = []
        for device, (offset, count) in zip(devices, chunks):
            part = Path(tmp) / f"part-{offset}.jsonl"
            log = Path(tmp) / f"worker-{offset}.log"
            cmd = [
                sys.executable, "-m", "rewrite.worker", str(annotation_file),
                "--model", model_id,
                "--device", device,
                "--offset", str(offset),
                "--count", str(count),
                "--output", str(part),
                "--video-root", str(video_root),
                "--num-frames", str(num_frames),
                "--max-new-tokens", str(max_new_tokens),
                "--dtype", dtype,
            ]
            procs.append((subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT), part, log, device))
            print(f"worker on {device}: records {offset}..{offset + count - 1}")

        with tqdm(total=total, desc="rewriting", unit="rec") as bar:
            while True:
                done = sum(_count_lines(part) for _, part, _, _ in procs)
                bar.n = done
                bar.refresh()
                if all(p.poll() is not None for p, _, _, _ in procs):
                    break
                time.sleep(2)

        failed = [(p, log, device) for p, _, log, device in procs if p.returncode != 0]
        if failed:
            for p, log, device in failed:
                tail = "".join(log.read_text().splitlines(keepends=True)[-15:])
                print(f"\nworker on {device} exited with {p.returncode}:\n{tail}", file=sys.stderr)
            raise SystemExit("parallel rewrite failed; partial results were discarded")

        captions: dict[int, str] = {}
        for _, part, _, _ in procs:
            with open(part, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    captions[row["index"]] = row["caption"]

    missing = set(range(total)) - set(captions)
    if missing:
        raise SystemExit(f"workers exited cleanly but {len(missing)} records lack captions: {sorted(missing)[:5]}...")

    output = copy.deepcopy(records)
    for index, caption in captions.items():
        set_assistant_text(output[index], caption)
    save_annotations(output, output_file)
