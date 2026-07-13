#!/usr/bin/env python3
"""Hash the exact official files used by the Ovi 720-checkpoint run."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILES = (
    "Ovi/model.safetensors",
    "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    "Wan2.2-TI2V-5B/google/umt5-xxl/spiece.model",
    "Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json",
    "Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer_config.json",
    "Wan2.2-TI2V-5B/google/umt5-xxl/special_tokens_map.json",
    "MMAudio/ext_weights/best_netG.pt",
    "MMAudio/ext_weights/v1-16.pth",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hf_download_metadata(checkpoint_root, relative_path):
    parts = Path(relative_path).parts
    metadata_path = (
        checkpoint_root
        / parts[0]
        / ".cache"
        / "huggingface"
        / "download"
        / Path(*parts[1:])
    ).with_name(parts[-1] + ".metadata")
    if not metadata_path.is_file():
        return {}
    lines = metadata_path.read_text().splitlines()
    return {
        "hf_revision": lines[0] if len(lines) >= 1 else None,
        "hf_etag": lines[1] if len(lines) >= 2 else None,
    }


def main():
    default_root = Path(os.environ.get("FASTA2V_CACHE_ROOT", "/cache/liluchen/FastA2V"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=default_root / "ckpts")
    parser.add_argument("--output", type=Path, default=default_root / "checkpoint_manifest.json")
    args = parser.parse_args()

    files = {}
    for relative_path in REQUIRED_FILES:
        path = args.checkpoint_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative_path] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            **hf_download_metadata(args.checkpoint_root, relative_path),
        }

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "transport_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "sources": {
            "Ovi": "chetwinlow1/Ovi",
            "Wan2.2-TI2V-5B": "Wan-AI/Wan2.2-TI2V-5B",
            "MMAudio": "hkchengrex/MMAudio",
        },
        "files": files,
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
