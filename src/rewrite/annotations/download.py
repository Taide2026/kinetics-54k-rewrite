"""Fetch annotation files from the HF dataset repo, mirroring the repo layout.

The CLIs take the annotation path *inside the dataset repo* (e.g.
``annotations/splits-SQ/test.json``) and materialize it at the identical
relative path locally before use.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "bear7011/gemma-4-e4b-kinetics_54K"


def ensure_annotation_file(
    path_in_repo: str,
    repo_id: str = DEFAULT_REPO_ID,
    token: str | None = None,
    local_root: str | Path = ".",
) -> Path:
    """Return the local path for ``path_in_repo``, downloading it if absent."""
    local_path = Path(local_root) / path_in_repo
    if local_path.is_file():
        return local_path
    print(f"Downloading {path_in_repo} from {repo_id} ...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=path_in_repo,
        local_dir=local_root,
        token=token,
    )
    return Path(downloaded)
