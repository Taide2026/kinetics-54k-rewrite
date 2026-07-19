"""Plan how many model replicas fit on the current hardware.

One replica per GPU at most — generation saturates a single GPU, so
co-locating replicas on one card only makes them queue behind each other.
If no single GPU can hold the model, fall back to one instance sharded
across all GPUs (``device_map="auto"``), which is the pre-parallel behavior.
"""

from dataclasses import dataclass

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}

# Margin over raw weight bytes for activations (the ~2K-token multimodal
# prefill), KV cache, and the CUDA context.
ACTIVATION_FACTOR = 1.15
FIXED_OVERHEAD_BYTES = 3 * 1024**3


@dataclass
class ReplicaPlan:
    devices: list[str]  # e.g. ["cuda:0", "cuda:1"], or ["auto"] for sharded fallback
    required_bytes: int
    reason: str

    @property
    def num_workers(self) -> int:
        return len(self.devices) if self.devices != ["auto"] else 1


def estimate_model_bytes(model_id: str, dtype: str, token: str | None = None) -> int:
    """Weight bytes at the target dtype, from the checkpoint's safetensors metadata
    (no weights are downloaded)."""
    from huggingface_hub import get_safetensors_metadata

    metadata = get_safetensors_metadata(model_id, token=token)
    total_params = sum(metadata.parameter_count.values())
    return total_params * DTYPE_BYTES[dtype]


def _gpu_free_bytes() -> list[int]:
    import torch

    if not torch.cuda.is_available():
        return []
    free = []
    for i in range(torch.cuda.device_count()):
        free_i, _total = torch.cuda.mem_get_info(i)
        free.append(free_i)
    return free


def plan_replicas(
    model_id: str,
    dtype: str,
    token: str | None = None,
    requested: int | None = None,
    max_useful: int | None = None,
) -> ReplicaPlan:
    """Decide replica devices.

    ``requested=None`` means auto: one replica on every GPU with enough free
    memory. An explicit ``requested`` is honored if that many GPUs fit the
    model, otherwise planning fails loudly rather than risking OOM.
    ``max_useful`` caps the count (e.g. at the number of records).
    """
    weight_bytes = estimate_model_bytes(model_id, dtype, token)
    required = int(weight_bytes * ACTIVATION_FACTOR) + FIXED_OVERHEAD_BYTES

    free = _gpu_free_bytes()
    fitting = sorted(
        (i for i, f in enumerate(free) if f >= required),
        key=lambda i: free[i],
        reverse=True,
    )

    if requested is not None and requested > 1:
        if len(fitting) < requested:
            raise SystemExit(
                f"--workers {requested} requested, but only {len(fitting)} GPU(s) have "
                f"the ~{required / 1024**3:.1f} GiB free needed for one replica of "
                f"{model_id} ({dtype}). Use fewer workers, or --workers 1 to shard "
                "one instance across all GPUs."
            )
        fitting = fitting[:requested]

    if max_useful is not None:
        fitting = fitting[: max(1, max_useful)]

    if requested == 1 or not free:
        return ReplicaPlan(["auto"], required, "single worker requested" if requested == 1 else "no CUDA GPUs")

    if not fitting:
        return ReplicaPlan(
            ["auto"],
            required,
            f"model needs ~{required / 1024**3:.1f} GiB but no single GPU has that free; "
            "using one instance sharded across GPUs",
        )

    devices = [f"cuda:{i}" for i in sorted(fitting)]
    return ReplicaPlan(
        devices,
        required,
        f"~{required / 1024**3:.1f} GiB per replica fits on {len(devices)} GPU(s)",
    )
