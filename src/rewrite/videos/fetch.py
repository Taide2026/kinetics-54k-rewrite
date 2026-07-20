"""Fetch only the videos referenced by an annotation file from the HF dataset.

The dataset ships videos as ~4 GB uncompressed tar shards
(``videos/shards/k600_train_*.tar``, ~92 GB total). Downloading everything to
caption a handful of records is wasteful, so this module walks each shard's
tar headers with HTTP Range requests (512 bytes per member) and downloads only
the byte ranges of the members we actually need.

``videos/shards/manifest.json`` maps labels to shards, so shards containing no
wanted label are skipped entirely.

Members are stored as ``k600/train/<label>/<id>.mp4`` and written locally as
``<dest>/kinetic600/<label>/<id>.mp4`` to match annotation references.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

from rewrite.annotations import ensure_annotation_file, get_video_ref, load_annotations
from rewrite.annotations.download import DEFAULT_REPO_ID
TAR_BLOCK = 512


def _resolve_url(repo_id: str, path_in_repo: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}"


def _auth_headers(token: str | None) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _retry_after_seconds(resp: requests.Response, default: float) -> float:
    """How long to wait after a 429, honoring the ``Retry-After`` header if sane."""
    value = resp.headers.get("Retry-After")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            pass  # HTTP-date form is rare here; fall back to our backoff
    return default


def load_manifest(repo_id: str, token: str | None) -> dict:
    resp = requests.get(
        _resolve_url(repo_id, "videos/shards/manifest.json"),
        headers=_auth_headers(token),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def member_name_for_ref(ref: str) -> str:
    """``kinetic600/<label>/<id>`` -> ``k600/train/<label>/<id>.mp4``."""
    _, label, video_id = ref.split("/", 2)
    return f"k600/train/{label}/{video_id}.mp4"


def dest_path_for_ref(ref: str, dest: Path) -> Path:
    return dest / (ref + ".mp4")


def _parse_octal(field: bytes) -> int:
    text = field.rstrip(b"\x00 ").strip()
    return int(text, 8) if text else 0


def _member_name(block: bytes) -> str:
    name = block[0:100].rstrip(b"\x00").decode("utf-8", errors="replace")
    if block[257:262] == b"ustar":
        prefix = block[345:500].rstrip(b"\x00").decode("utf-8", errors="replace")
        if prefix:
            name = f"{prefix}/{name}"
    return name


class _ShardReader:
    """Ranged reads against one shard, reusing the resolved CDN URL."""

    # Retries cover two transient cases: an expired presigned CDN URL (re-resolve)
    # and HTTP 429 rate limiting (wait, then retry with exponential backoff).
    MAX_ATTEMPTS = 10
    BACKOFF_START = 1.0
    BACKOFF_CAP = 60.0

    def __init__(self, repo_id: str, repo_path: str, token: str | None):
        self.resolve_url = _resolve_url(repo_id, repo_path)
        self.token = token
        self.session = requests.Session()
        self.cdn_url: str | None = None

    def read(self, offset: int, length: int) -> bytes:
        headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
        backoff = self.BACKOFF_START
        for _attempt in range(self.MAX_ATTEMPTS):
            if self.cdn_url is None:
                resp = self.session.get(
                    self.resolve_url,
                    headers={**headers, **_auth_headers(self.token)},
                    timeout=120,
                )
                if resp.ok:
                    self.cdn_url = resp.url
            else:
                resp = self.session.get(self.cdn_url, headers=headers, timeout=120)
                if resp.status_code in (401, 403):  # presigned URL expired
                    self.cdn_url = None
                    continue
            if resp.status_code == 429:  # rate limited: wait, then retry
                time.sleep(_retry_after_seconds(resp, backoff))
                backoff = min(backoff * 2, self.BACKOFF_CAP)
                continue
            resp.raise_for_status()
            return resp.content
        raise RuntimeError(f"Could not read range from {self.resolve_url}")


def fetch_from_shard(
    repo_id: str,
    repo_path: str,
    wanted: dict[str, Path],
    token: str | None,
) -> list[str]:
    """Walk one shard's tar headers; download wanted members. Returns found names."""
    reader = _ShardReader(repo_id, repo_path, token)
    remaining = dict(wanted)
    found: list[str] = []
    offset = 0
    long_name: str | None = None

    while remaining:
        block = reader.read(offset, TAR_BLOCK)
        if len(block) < TAR_BLOCK or block == b"\x00" * TAR_BLOCK:
            break  # end of archive
        size = _parse_octal(block[124:136])
        typeflag = block[156:157]
        data_blocks = (size + TAR_BLOCK - 1) // TAR_BLOCK

        if typeflag == b"L":  # GNU long name: data holds the next member's name
            long_name = reader.read(offset + TAR_BLOCK, size).rstrip(b"\x00").decode("utf-8")
        elif typeflag in (b"0", b"\x00"):
            name = long_name or _member_name(block)
            long_name = None
            if name in remaining:
                data = reader.read(offset + TAR_BLOCK, size)
                out_path = remaining.pop(name)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(out_path)
                found.append(name)
        else:
            long_name = None

        offset += TAR_BLOCK + data_blocks * TAR_BLOCK

    return found


def fetch_videos(
    annotation_file: str | Path,
    dest: str | Path = "videos",
    repo_id: str = DEFAULT_REPO_ID,
    limit: int | None = None,
    workers: int = 6,
    token: str | None = None,
) -> None:
    dest = Path(dest)
    annotation_file = ensure_annotation_file(str(annotation_file), repo_id=repo_id, token=token)
    records = load_annotations(annotation_file)
    if limit is not None:
        records = records[:limit]

    refs = {ref for r in records if (ref := get_video_ref(r)) is not None}
    wanted: dict[str, Path] = {}
    for ref in sorted(refs):
        out_path = dest_path_for_ref(ref, dest)
        if out_path.is_file() and out_path.stat().st_size > 0:
            continue
        wanted[member_name_for_ref(ref)] = out_path

    print(f"{len(refs)} videos referenced, {len(wanted)} to fetch")
    if not wanted:
        return

    manifest = load_manifest(repo_id, token)
    wanted_labels = {name.split("/")[2] for name in wanted}
    jobs = []  # (repo_path, wanted-members-in-shard)
    for shard in manifest["shards"]:
        if not wanted_labels & set(shard["labels"]):
            continue
        in_shard = {
            name: path for name, path in wanted.items() if name.split("/")[2] in shard["labels"]
        }
        jobs.append((shard["repo_path"], in_shard))

    print(f"Scanning {len(jobs)}/{len(manifest['shards'])} shards with {workers} workers ...")
    total_found = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_from_shard, repo_id, repo_path, in_shard, token): repo_path
            for repo_path, in_shard in jobs
        }
        for future in as_completed(futures):
            repo_path = futures[future]
            found = future.result()
            total_found += len(found)
            print(f"  {repo_path}: {len(found)} files ({total_found}/{len(wanted)} total)")

    missing = len(wanted) - total_found
    if missing:
        sys.exit(f"ERROR: {missing} wanted videos were not found in any shard")
    print(f"Done. Videos are under {dest}/")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Download only the videos referenced by an annotation file "
        "from the HF dataset's tar shards (via HTTP range requests)."
    )
    parser.add_argument(
        "annotation_file",
        help="Annotation JSON path inside the dataset repo, e.g. "
        "annotations/splits-SQ/test.json; downloaded locally if not present",
    )
    parser.add_argument("--dest", default="videos", help="Local video root (default: videos)")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"HF dataset repo (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--limit", type=int, default=None, help="Only consider the first N records")
    parser.add_argument("--workers", type=int, default=6, help="Parallel shard scanners (default: 6)")
    args = parser.parse_args(argv)

    fetch_videos(
        annotation_file=args.annotation_file,
        dest=args.dest,
        repo_id=args.repo_id,
        limit=args.limit,
        workers=args.workers,
        token=os.environ.get("HF_TOKEN"),
    )


if __name__ == "__main__":
    main()
