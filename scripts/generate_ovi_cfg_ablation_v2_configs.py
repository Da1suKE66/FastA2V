#!/usr/bin/env python3
"""Validate and materialize the frozen Ovi CFG-cache ablation v2 matrix.

This is an extended repository integration of the authoritative bundle generator
(source SHA256 is retained in the protocol input binding).  It does not launch
GPU work.  It emits immutable flat YAML inputs plus a SHA-bound manifest for the
existing fail-closed runner.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ovi.cfg_ablation_v2_protocol import (  # noqa: E402
    FROZEN_CONFIG,
    PROMPT_SET_CONTRACTS,
    PROTOCOL_ID,
    STAGE3_ALLOWED_CONFIG_IDS,
    STAGE4_ALLOWED_CONFIG_IDS,
    STAGE_SEEDS,
    Cell,
    ProtocolError,
    cell_filename,
    filter_cells,
    load_and_validate_matrix,
    protocol_summary,
    validate_frozen_base_config,
    validate_seed_filter,
)


DEFAULT_MATRIX = REPO_ROOT / "configs/matrix/ovi_cfg_cache_ablation_v2_matrix.csv"
DEFAULT_PROTOCOL_DOC = REPO_ROOT / "docs/protocol/ovi_cfg_cache_ablation_v2.md"
DEFAULT_SOURCE_BINDING = (
    REPO_ROOT / "configs/matrix/ovi_cfg_cache_ablation_v2_inputs.json"
)
PROTOCOL_MODULE = REPO_ROOT / "ovi/cfg_ablation_v2_protocol.py"
FORBIDDEN_FORMAL8 = (REPO_ROOT / "prompts/ovi_formal8.csv").resolve()
PROMPT_HEADER = ("text_prompt",)
PROMPT_MANIFEST_HEADER = ("prompt_id", "category", "primary_stress")
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INT_RE = re.compile(r"^[+-]?(?:0|[1-9][0-9]*)$")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:(?:[0-9]+\.[0-9]*)|(?:[0-9]*\.[0-9]+))(?:[eE][+-]?[0-9]+)?$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_scalar(raw: str, *, path: Path, line_number: int) -> Any:
    value = raw.strip()
    if not value:
        raise ProtocolError(
            f"{path}:{line_number}: nested/empty YAML values are unsupported"
        )
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if _INT_RE.fullmatch(value):
        return int(value)
    if _FLOAT_RE.fullmatch(value):
        return float(value)
    if value[0] in "[{'\"":
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ProtocolError(
                f"{path}:{line_number}: invalid flat YAML scalar {value!r}"
            ) from exc
        if isinstance(parsed, (dict, tuple, set)):
            raise ProtocolError(
                f"{path}:{line_number}: only scalar/list flat YAML values are supported"
            )
        return parsed
    return value


def load_flat_yaml(path: Path) -> dict[str, Any]:
    """Parse the repository's flat Ovi configs without a PyYAML dependency."""

    path = Path(path)
    result: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ProtocolError(f"cannot read base config {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line[:1].isspace():
            raise ProtocolError(
                f"{path}:{line_number}: nested YAML is not supported by this frozen generator"
            )
        if ":" not in raw_line:
            raise ProtocolError(f"{path}:{line_number}: expected key: value")
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        if not _KEY_RE.fullmatch(key):
            raise ProtocolError(f"{path}:{line_number}: invalid YAML key {key!r}")
        if key in result:
            raise ProtocolError(f"{path}:{line_number}: duplicate YAML key {key!r}")
        result[key] = _parse_scalar(raw_value, path=path, line_number=line_number)
    if not result:
        raise ProtocolError("base YAML must contain a non-empty top-level mapping")
    return result


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise ProtocolError(f"cannot serialize YAML value of type {type(value).__name__}")


def dump_flat_yaml(config: Mapping[str, Any]) -> str:
    lines = []
    for key, value in config.items():
        if not _KEY_RE.fullmatch(key):
            raise ProtocolError(f"invalid output YAML key {key!r}")
        lines.append(f"{key}: {_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _csv_set(raw: str) -> set[str] | None:
    values = {value.strip() for value in raw.split(",") if value.strip()}
    return values or None


def _parse_seeds(raw: str) -> list[int]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise ProtocolError(f"invalid comma-separated seed filter {raw!r}") from exc


def load_prompt_csv(path: Path) -> list[str]:
    path = Path(path)
    if path.resolve() == FORBIDDEN_FORMAL8 or path.name == "ovi_formal8.csv":
        raise ProtocolError(
            "prompts/ovi_formal8.csv is development-overlapping and must not be "
            "used as the CFG ablation v2 held-out set"
        )
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PROMPT_HEADER:
                raise ProtocolError(
                    f"prompt CSV header {tuple(reader.fieldnames or ())!r} != {PROMPT_HEADER!r}"
                )
            rows = list(reader)
    except OSError as exc:
        raise ProtocolError(f"cannot read prompt CSV {path}: {exc}") from exc
    prompts = []
    for index, row in enumerate(rows, start=1):
        prompt = row.get("text_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ProtocolError(f"prompt row {index} must be non-empty")
        prompts.append(prompt)
    if not prompts:
        raise ProtocolError("prompt CSV must contain at least one prompt")
    return prompts


def load_prompt_manifest(path: Path, prompt_count: int) -> list[dict[str, str]]:
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PROMPT_MANIFEST_HEADER:
                raise ProtocolError(
                    "prompt manifest header differs from the frozen held-out schema"
                )
            rows = list(reader)
    except OSError as exc:
        raise ProtocolError(f"cannot read prompt manifest {path}: {exc}") from exc
    if len(rows) != prompt_count:
        raise ProtocolError(
            f"prompt manifest rows {len(rows)} != prompt count {prompt_count}"
        )
    expected_ids = [f"H{index:02d}" for index in range(1, prompt_count + 1)]
    ids = [row.get("prompt_id") for row in rows]
    if ids != expected_ids:
        raise ProtocolError(f"held-out prompt IDs {ids!r} != {expected_ids!r}")
    for row in rows:
        if not row.get("category") or not row.get("primary_stress"):
            raise ProtocolError("held-out prompt manifest fields must be non-empty")
    return rows


def validate_source_binding(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read source binding {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError("source binding has an invalid protocol_id")
    repo_files = payload.get("repo_files")
    if not isinstance(repo_files, dict) or not repo_files:
        raise ProtocolError("source binding must contain repo_files")
    for relative, binding in repo_files.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ProtocolError(f"invalid source-bound repository path {relative!r}")
        if not isinstance(binding, dict) or not isinstance(binding.get("sha256"), str):
            raise ProtocolError(f"invalid source binding for {relative}")
        actual_path = REPO_ROOT / relative
        if not actual_path.is_file():
            raise ProtocolError(f"source-bound repository input is missing: {actual_path}")
        actual = sha256_file(actual_path)
        if actual != binding["sha256"]:
            raise ProtocolError(
                f"source-bound SHA mismatch for {relative}: {actual} != {binding['sha256']}"
            )
    return payload


def configure(
    base: Mapping[str, Any],
    cell: Cell,
    seed: int,
    prompt_csv: Path,
    warmups: int,
    measurements: int,
    benchmark_eligible: bool,
    execution_stage: str | None = None,
) -> dict[str, Any]:
    config = dict(base)
    config.update(FROZEN_CONFIG)
    effective_stage = execution_stage if execution_stage is not None else cell.stage
    config.update(
        {
            "seed": seed,
            "text_prompt": str(prompt_csv),
            "each_example_n_times": 1,
            "warmup_runs": warmups,
            "measurement_runs": measurements,
            "run_kind": (
                f"cfg_cache_ablation_v2_s{effective_stage}_{cell.config_id}_seed{seed}"
            ),
            "benchmark_eligible": benchmark_eligible,
            "use_cfg_cache": cell.use_cfg_cache,
            "cfg_ablation_protocol_id": PROTOCOL_ID,
            "cfg_ablation_stage": int(effective_stage),
            "cfg_ablation_source_matrix_stage": int(cell.stage),
            "cfg_ablation_execution_stage": int(effective_stage),
            "cfg_ablation_config_id": cell.config_id,
            "cfg_cache_window_indexing": "zero_based_inclusive",
        }
    )
    if cell.use_cfg_cache:
        config.update(
            {
                "cfg_cache_start_step": cell.start_step,
                "cfg_cache_end_step": cell.end_step,
                "cfg_cache_refresh_interval": cell.refresh_interval,
            }
        )
    else:
        config.update(
            {
                "cfg_cache_start_step": 0,
                "cfg_cache_end_step": 0,
                "cfg_cache_refresh_interval": 1,
            }
        )
    validate_frozen_base_config(config)
    return config


def _input_binding(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("base_config", "matrix", "prompt_csv", "protocol_doc", "source_binding"):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise ProtocolError(f"missing required file: {path}")
    if args.prompt_manifest is not None and not Path(args.prompt_manifest).is_file():
        raise ProtocolError(f"missing required file: {args.prompt_manifest}")

    source_binding = validate_source_binding(args.source_binding)
    base = load_flat_yaml(args.base_config)
    validate_frozen_base_config(base)
    all_cells = load_and_validate_matrix(args.matrix)
    stages = _csv_set(args.stages)
    config_ids = _csv_set(args.config_ids)
    cells = filter_cells(all_cells, stages, config_ids)
    execution_stage = args.execution_stage
    if execution_stage in {"0", "1", "2"}:
        mismatched = [cell.config_id for cell in cells if cell.stage != execution_stage]
        if mismatched:
            raise ProtocolError(
                f"execution stage {execution_stage} cannot use source-matrix cells "
                f"from another stage: {mismatched}"
            )
    elif execution_stage == "3":
        invalid = sorted(
            cell.config_id
            for cell in cells
            if cell.config_id not in STAGE3_ALLOWED_CONFIG_IDS
        )
        if invalid:
            raise ProtocolError(f"stage 3 cannot materialize config IDs: {invalid}")
    elif execution_stage == "4":
        invalid = sorted(
            cell.config_id
            for cell in cells
            if cell.config_id not in STAGE4_ALLOWED_CONFIG_IDS
        )
        if invalid:
            raise ProtocolError(f"stage 4 cannot materialize config IDs: {invalid}")
    seeds = validate_seed_filter(_parse_seeds(args.seeds), cells, execution_stage)
    prompts = load_prompt_csv(args.prompt_csv)
    prompt_manifest = None
    if args.prompt_manifest is not None:
        prompt_manifest = load_prompt_manifest(args.prompt_manifest, len(prompts))
    if execution_stage == "3":
        expected_prompt = REPO_ROOT / PROMPT_SET_CONTRACTS["heldout8"]["path"]
        expected_manifest = REPO_ROOT / PROMPT_SET_CONTRACTS["heldout8"]["manifest"]
        if Path(args.prompt_csv).resolve() != expected_prompt.resolve():
            raise ProtocolError(
                "stage 3 requires the authoritative held-out prompt CSV; "
                "prompts/ovi_formal8.csv and other substitutes are forbidden"
            )
        if args.prompt_manifest is None or (
            Path(args.prompt_manifest).resolve() != expected_manifest.resolve()
        ):
            raise ProtocolError(
                "stage 3 requires the authoritative held-out prompt manifest"
            )
        if len(prompts) != PROMPT_SET_CONTRACTS["heldout8"]["prompt_count"]:
            raise ProtocolError("stage 3 held-out prompt count must be exactly 8")

    pairs = [
        (cell, seed)
        for cell in cells
        for seed in seeds
        if execution_stage is not None or seed in STAGE_SEEDS[cell.stage]
    ]
    if not pairs:
        raise ProtocolError("no stage/config/seed combinations matched the protocol")
    return {
        "base": base,
        "all_cells": all_cells,
        "cells": cells,
        "seeds": seeds,
        "pairs": pairs,
        "prompts": prompts,
        "prompt_manifest": prompt_manifest,
        "source_binding": source_binding,
        "stages": stages,
        "config_ids": config_ids,
        "execution_stage": execution_stage,
    }


def _validation_summary(args: argparse.Namespace, validated: Mapping[str, Any]) -> dict[str, Any]:
    inputs = {
        "base_config": _input_binding(args.base_config),
        "matrix": _input_binding(args.matrix),
        "prompt_csv": _input_binding(args.prompt_csv),
        "protocol_doc": _input_binding(args.protocol_doc),
        "source_binding": _input_binding(args.source_binding),
        "generator": _input_binding(Path(__file__)),
        "protocol_module": _input_binding(PROTOCOL_MODULE),
    }
    if args.prompt_manifest is not None:
        inputs["prompt_manifest"] = _input_binding(args.prompt_manifest)
    return {
        "schema_version": 2,
        "record_type": "ovi_cfg_ablation_v2_protocol_validation",
        "protocol_id": PROTOCOL_ID,
        "status": "ok",
        "filters": {
            "stages": sorted(validated["stages"]) if validated["stages"] else None,
            "config_ids": (
                sorted(validated["config_ids"]) if validated["config_ids"] else None
            ),
            "seeds": list(validated["seeds"]),
            "execution_stage": validated["execution_stage"],
        },
        "selected_config_ids": [cell.config_id for cell in validated["cells"]],
        "selected_materializations": len(validated["pairs"]),
        "prompt_count": len(validated["prompts"]),
        "input_files": inputs,
        "protocol": protocol_summary(),
    }


def validate_protocol_command(args: argparse.Namespace) -> int:
    validated = validate_inputs(args)
    print(json.dumps(_validation_summary(args, validated), ensure_ascii=False, indent=2))
    return 0


def materialize_command(args: argparse.Namespace) -> int:
    if args.warmup_runs < 0 or args.measurement_runs < 1:
        raise ProtocolError("warmup-runs must be >= 0 and measurement-runs must be >= 1")
    if args.benchmark_eligible and (
        args.warmup_runs < 3 or args.measurement_runs < 5
    ):
        raise ProtocolError(
            "benchmark materialization requires at least 3 warmups and 5 measurements"
        )
    if args.execution_stage == "4" and not args.benchmark_eligible:
        raise ProtocolError("execution stage 4 requires --benchmark-eligible")
    validated = validate_inputs(args)
    output_dir = Path(args.output_dir).resolve()
    if os.path.lexists(output_dir):
        raise ProtocolError(f"refusing to reuse existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs_dir = output_dir / "inputs"
    configs_dir = output_dir / "configs"
    inputs_dir.mkdir()
    configs_dir.mkdir()

    copied_prompt = inputs_dir / Path(args.prompt_csv).name
    shutil.copyfile(args.prompt_csv, copied_prompt)
    copied_prompt_manifest = None
    if args.prompt_manifest is not None:
        copied_prompt_manifest = inputs_dir / Path(args.prompt_manifest).name
        shutil.copyfile(args.prompt_manifest, copied_prompt_manifest)

    validation = _validation_summary(args, validated)
    materializations = []
    for cell, seed in validated["pairs"]:
        filename = cell_filename(cell, seed, validated["execution_stage"])
        output_path = configs_dir / filename
        config = configure(
            validated["base"],
            cell,
            seed,
            copied_prompt,
            args.warmup_runs,
            args.measurement_runs,
            args.benchmark_eligible,
            validated["execution_stage"],
        )
        effective_stage = (
            validated["execution_stage"]
            if validated["execution_stage"] is not None
            else cell.stage
        )
        output_path.write_text(dump_flat_yaml(config), encoding="utf-8")
        entry = {
            "stage": int(effective_stage),
            "source_matrix_stage": int(cell.stage),
            "execution_stage": int(effective_stage),
            "config_id": cell.config_id,
            "seed": seed,
            "indexing": "zero_based_inclusive",
            "use_cfg_cache": cell.use_cfg_cache,
            "start_step": cell.start_step,
            "end_step": cell.end_step,
            "refresh_interval": cell.refresh_interval,
            "eligible_steps": cell.eligible_steps,
            "refreshes": cell.refreshes,
            "cache_hits": cell.cache_hits,
            "negative_forwards": cell.negative_forwards,
            "expected_video_self_attention_calls": (
                cell.expected_video_self_attention_calls
            ),
            "max_cache_age": cell.max_cache_age,
            "config_path": str(output_path),
            "config_sha256": sha256_file(output_path),
            "suggested_run_tag": output_path.stem,
        }
        materializations.append(entry)

    copied_inputs = {
        "prompt_csv": _input_binding(copied_prompt),
    }
    if copied_prompt_manifest is not None:
        copied_inputs["prompt_manifest"] = _input_binding(copied_prompt_manifest)
    manifest = {
        **validation,
        "record_type": "ovi_cfg_ablation_v2_materialization_manifest",
        "output_dir": str(output_dir),
        "warmup_runs": args.warmup_runs,
        "measurement_runs": args.measurement_runs,
        "benchmark_eligible": args.benchmark_eligible,
        "copied_inputs": copied_inputs,
        "materializations": materializations,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(materializations)} configs and {manifest_path}")
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--prompt-csv", required=True, type=Path)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--protocol-doc", type=Path, default=DEFAULT_PROTOCOL_DOC)
    parser.add_argument("--source-binding", type=Path, default=DEFAULT_SOURCE_BINDING)
    parser.add_argument("--seeds", default="103", help="comma-separated seed filter")
    parser.add_argument("--stages", default="", help="comma-separated stage filter")
    parser.add_argument(
        "--config-ids", default="", help="comma-separated config ID filter"
    )
    parser.add_argument(
        "--execution-stage",
        choices=("0", "1", "2", "3", "4"),
        help="execute selected source-matrix cells in this protocol stage",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ovi CFG-cache ablation v2 protocol validator/materializer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate-protocol", help="validate frozen inputs without writing output"
    )
    _add_common_arguments(validate)
    validate.set_defaults(handler=validate_protocol_command)

    materialize = subparsers.add_parser(
        "materialize-config", help="write a new immutable config input directory"
    )
    _add_common_arguments(materialize)
    materialize.add_argument("--output-dir", required=True, type=Path)
    materialize.add_argument("--warmup-runs", type=int, default=1)
    materialize.add_argument("--measurement-runs", type=int, default=1)
    materialize.add_argument("--benchmark-eligible", action="store_true")
    materialize.set_defaults(handler=materialize_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ProtocolError as exc:
        parser.exit(2, f"protocol error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
