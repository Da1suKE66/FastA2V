#!/usr/bin/env python3
"""Record and enforce an idle physical GPU 0 before any CUDA work."""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ovi.gpu_process_monitor import (  # noqa: E402
    build_pre_run_gpu_report,
    query_gpu_compute_processes,
)


def main(output_path, device_index=0, sample_fn=query_gpu_compute_processes):
    if int(device_index) != 0:
        raise ValueError("FastA2V currently supports physical GPU index 0 only")
    snapshot = sample_fn(0)
    report = build_pre_run_gpu_report(snapshot)
    rendered = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    output_path = Path(output_path)
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["valid_for_run"]:
        print(
            "Refusing CUDA preflight/model loading because physical GPU 0 "
            "is not proven idle.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(main(args.output, args.device_index))
