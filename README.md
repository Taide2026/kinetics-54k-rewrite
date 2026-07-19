# kinetics-54k-rewrite

Rewrite the ground-truth (assistant) captions of the
[`bear7011/gemma-4-e4b-kinetics_54K`](https://huggingface.co/datasets/bear7011/gemma-4-e4b-kinetics_54K)
dataset with a video VLM (e.g. `google/gemma-4-E4B-it`). The output is a clone
of the input annotation JSON where **only** the assistant `content[].text` of
each processed record is replaced by the model's caption; every other field is
copied verbatim.

## Setup

```bash
uv sync                      # installs torch (cu121), transformers, etc.
echo 'HF_TOKEN=hf_...' > .env
```

Both CLIs take the annotation path **as it appears inside the dataset repo**
(e.g. `annotations/splits-SQ/test.json`) and download it to the same relative
path locally on first use — no manual annotation download needed. Use
`--repo-id` to point at a different dataset repo.

## Usage

### 1. Fetch the videos you need

Videos are not part of the annotation files. `fetch-videos` downloads only the
clips referenced by an annotation file from the dataset's 92 GB tar shards,
using HTTP range requests (tar headers are walked remotely; only wanted
members are transferred):

```bash
uv run fetch-videos annotations/splits-SQ/test.json --limit 200
```

Files land under `videos/kinetic600/<label>/<video_id>.mp4`, matching the
records' video references. Skip this step if you already have Kinetics-600
locally — just point `--video-root` at it.

### 2. Rewrite the captions

```bash
uv run rewrite-captions annotations/splits-SQ/test.json \
    --model google/gemma-4-E4B-it \
    --limit 200
```

- `annotation_file` (required): annotation JSON path inside the dataset repo;
  auto-downloaded to the same local path if missing.
- `--model` (required): HF model id that generates the new captions.
- `--output`: output path (default: `<input>.rewritten.json` next to the input).
- `--video-root`: local video directory (default: `videos`).
- `--limit`: rewrite only the first N records (the rest are copied unchanged).
- `--repo-id`: dataset repo to download annotations from (default:
  `bear7011/gemma-4-e4b-kinetics_54K`).
- `--workers`: model replicas run in parallel (default `auto`: one replica on
  every GPU with enough free memory, estimated from the checkpoint's
  safetensors metadata — no manual tuning needed). Falls back to a single
  instance sharded across GPUs when one GPU cannot hold the model; `--workers 1`
  forces that mode.
- `--num-frames` (8), `--max-new-tokens` (64), `--dtype` (bfloat16),
  `--device-map` (auto; single-worker mode only), `--save-every` (checkpoint
  interval, 25; single-worker mode only).

Each video is decoded into `--num-frames` evenly spaced frames and fed to the
model together with the record's own system/user prompts; generation is greedy.
With multiple workers, records are split into contiguous slices, each worker
streams its captions to a JSONL part file (so a crash loses nothing already
done), and the parts are merged into the final output.

## Layout

```
src/rewrite/
├── cli.py            # rewrite-captions entry point
├── pipeline.py       # rewrite loop (load → caption → save)
├── hardware.py       # replica planner (fits one model copy per free GPU)
├── parallel.py       # multi-worker orchestrator (spawn, progress, merge)
├── worker.py         # per-GPU worker (python -m rewrite.worker)
├── annotations/      # JSON I/O + record accessors (only assistant text is mutated)
├── model/            # Gemma-4 captioner (frames-as-images chat inference)
└── videos/           # video path resolution, frame sampling, shard fetcher
```
