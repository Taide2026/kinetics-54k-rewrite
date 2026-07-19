"""CLI: rewrite the ground-truth captions of an annotation file with a VLM."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from rewrite.annotations import ensure_annotation_file
from rewrite.annotations.download import DEFAULT_REPO_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rewrite-captions",
        description="Rewrite the assistant ground-truth captions of a kinetics_54K "
        "annotation file using a video VLM. Only assistant content[].text is "
        "changed; everything else is copied verbatim.",
    )
    parser.add_argument(
        "annotation_file",
        help="Annotation JSON path inside the dataset repo, e.g. "
        "annotations/splits-SQ/test.json; it is downloaded to the same "
        "relative path locally if not already present",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo to download the annotation file from (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id used to rewrite captions, e.g. google/gemma-4-E4B-it",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <annotation_file stem>.rewritten.json next to the input)",
    )
    parser.add_argument(
        "--video-root",
        default="videos",
        help="Local directory that resolves the records' video references (default: videos)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Rewrite only the first N records")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames sampled per video (default: 8)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation budget per caption (default: 64)")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help='Model replicas to run in parallel: an integer, or "auto" (default) to '
        "fit one replica per GPU with enough free memory. Falls back to a single "
        "instance sharded across GPUs when one GPU cannot hold the model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Passed to from_pretrained; only used when running a single worker (default: "auto")',
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Checkpoint the output file every N records (default: 25)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)

    annotation_file = ensure_annotation_file(
        args.annotation_file,
        repo_id=args.repo_id,
        token=os.environ.get("HF_TOKEN"),
    )
    output_file = (
        Path(args.output)
        if args.output
        else annotation_file.with_name(annotation_file.stem + ".rewritten.json")
    )

    if args.workers == "auto":
        requested = None
    else:
        try:
            requested = int(args.workers)
        except ValueError:
            raise SystemExit(f'--workers must be an integer or "auto", got {args.workers!r}')
        if requested < 1:
            raise SystemExit("--workers must be >= 1")

    token = os.environ.get("HF_TOKEN")

    if requested != 1:
        # Import lazily so `--help` stays fast and dependency-light.
        from rewrite.hardware import plan_replicas

        plan = plan_replicas(args.model, args.dtype, token=token, requested=requested, max_useful=args.limit)
        print(f"worker plan: {plan.num_workers} ({plan.reason})")
        if plan.num_workers > 1:
            from rewrite.parallel import rewrite_parallel

            rewrite_parallel(
                annotation_file=annotation_file,
                output_file=output_file,
                model_id=args.model,
                devices=plan.devices,
                video_root=args.video_root,
                limit=args.limit,
                num_frames=args.num_frames,
                max_new_tokens=args.max_new_tokens,
                dtype=args.dtype,
            )
            print(f"Wrote {output_file}")
            return

    from rewrite.model import Captioner
    from rewrite.pipeline import rewrite_annotations

    captioner = Captioner(
        model_id=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        token=os.environ.get("HF_TOKEN"),
    )
    rewrite_annotations(
        annotation_file=annotation_file,
        output_file=output_file,
        captioner=captioner,
        video_root=args.video_root,
        limit=args.limit,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        save_every=args.save_every,
    )
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
