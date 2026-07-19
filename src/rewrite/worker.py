"""Internal worker for parallel rewriting: caption one contiguous record slice.

Invoked by the parallel orchestrator as ``python -m rewrite.worker``. Emits one
JSON line ``{"index": ..., "caption": ...}`` per record, flushed immediately so
the parent can track progress and a crash loses nothing already captioned.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="rewrite.worker")
    parser.add_argument("annotation_file")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True, help='e.g. "cuda:0" or "auto"')
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", required=True, help="JSONL file for {index, caption} lines")
    parser.add_argument("--video-root", default="videos")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args(argv)

    from rewrite.annotations import load_annotations
    from rewrite.model import Captioner
    from rewrite.pipeline import caption_record

    records = load_annotations(args.annotation_file)
    todo = records[args.offset : args.offset + args.count]

    # Resume: skip indices already present in the output from a previous attempt.
    done: set[int] = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            done = {json.loads(line)["index"] for line in f if line.strip()}

    captioner = Captioner(
        model_id=args.model,
        dtype=args.dtype,
        device_map=args.device,
        token=os.environ.get("HF_TOKEN"),
    )

    with open(args.output, "a", encoding="utf-8") as out:
        for i, record in enumerate(todo):
            index = args.offset + i
            if index in done:
                continue
            caption = caption_record(
                record,
                captioner,
                video_root=args.video_root,
                num_frames=args.num_frames,
                max_new_tokens=args.max_new_tokens,
            )
            out.write(json.dumps({"index": index, "caption": caption}, ensure_ascii=False) + "\n")
            out.flush()

    print(f"worker[{args.device}] finished records {args.offset}..{args.offset + args.count - 1}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
