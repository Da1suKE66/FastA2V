#!/usr/bin/env python3
"""Build audited, artifact-identity-paired Ovi quality sidecars.

The command deliberately accepts one explicit formal dense run and one explicit
formal candidate run.  It reuses the performance-run validator before reading
media, pairs every measurement/prompt/sample identity, and hashes both
artifacts before and after metric execution.  Missing LPIPS packages, modules,
weights, or source receipt fields are fatal; this tool never substitutes a
zero score.
"""

from __future__ import annotations

import sys

# The fixed evaluator is installed with ``pip --no-compile`` and must never
# create unauthenticated bytecode while validating or scoring.  Set this before
# loading the local run validator or any third-party scoring dependency.
sys.dont_write_bytecode = True

import argparse
import base64
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import statistics
import subprocess
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping
import zipfile


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _load_fixed_module(module_name: str, path: Path) -> Any:
    """Compile one exact source path, bypassing all import and pyc lookup."""

    path = path.resolve()
    try:
        source = path.read_bytes()
        code = compile(source, str(path), "exec", dont_inherit=True)
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"cannot compile fixed evaluator module from {path}: {exc}") from exc
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(code, module.__dict__)
    return module


RUN_VALIDATOR_PATH = SCRIPT_DIR / "build_ovi_eval_csv.py"
ARCHIVE_URL_POLICY_PATH = SCRIPT_DIR / "quality_archive_urls.py"
_ARCHIVE_URL_POLICY = _load_fixed_module(
    "_fasta2v_quality_archive_urls",
    ARCHIVE_URL_POLICY_PATH,
)
_FIXED_RUN_VALIDATOR: Any | None = None


def _run_validator_module() -> Any:
    global _FIXED_RUN_VALIDATOR
    if _FIXED_RUN_VALIDATOR is None:
        _FIXED_RUN_VALIDATOR = _load_fixed_module(
            "_fasta2v_build_ovi_eval_csv",
            RUN_VALIDATOR_PATH,
        )
    return _FIXED_RUN_VALIDATOR


DEFAULT_PROTOCOL = REPO_ROOT / "configs" / "quality_protocol.json"
DEFAULT_MATRIX = REPO_ROOT / "configs" / "ovi_eval_matrix.json"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)
QUALITY_SCHEMA_VERSION = 2
EXPECTED_MEASUREMENT_INDICES = (0, 1, 2)
# Backwards-compatible name for callers that only need the measurement repeat
# dimension.  Artifact pairing itself always uses all three identity fields.
EXPECTED_INDICES = EXPECTED_MEASUREMENT_INDICES
IDENTITY_FIELDS = (
    "measurement_index",
    "prompt_index",
    "sample_index",
)
EXPECTED_SAME_FIELDS = (
    "git_commit",
    "checkpoint_fingerprint_sha256",
    "gpu_identity",
    "prompt_set_sha256",
    "prompt_count",
    "prompts",
    "base_seed",
    "sample_count",
    "sample_seeds",
    "requested_shape",
    "actual_shape",
    "generated_video_shape",
    "generated_audio_shape",
    "sample_steps",
)
MANUAL_FIELDS = (
    "measurement_index",
    "prompt_index",
    "sample_index",
    "dense_artifact_sha256",
    "candidate_artifact_sha256",
    "reviewer",
    "reviewed_at_utc",
    "sync_rating",
    "notes",
)
MEDIAN_METRICS = (
    "video_psnr_db",
    "video_ssim",
    "lpips_alex",
    "audio_rmse",
    "audio_max_abs_difference",
    "audio_snr_db",
    "audio_correlation",
)
PACKAGE_CONTRACT_FIELDS = (
    "distribution",
    "version",
    "module",
    "module_path",
    "source_index",
)
EXPECTED_PACKAGE_CONTRACTS = (
    (
        "torch",
        "2.6.0+cpu",
        "torch",
        "/cache/liluchen/FastA2V/envs/eval/lib/python3.11/site-packages/torch/__init__.py",
        "https://download.pytorch.org/whl/cpu",
    ),
    (
        "torchvision",
        "0.21.0+cpu",
        "torchvision",
        "/cache/liluchen/FastA2V/envs/eval/lib/python3.11/site-packages/torchvision/__init__.py",
        "https://download.pytorch.org/whl/cpu",
    ),
    (
        "lpips",
        "0.1.4",
        "lpips",
        "/cache/liluchen/FastA2V/envs/eval/lib/python3.11/site-packages/lpips/__init__.py",
        "https://pypi.org/simple",
    ),
    (
        "numpy",
        "1.26.4",
        "numpy",
        "/cache/liluchen/FastA2V/envs/eval/lib/python3.11/site-packages/numpy/__init__.py",
        "https://pypi.org/simple",
    ),
    (
        "scipy",
        "1.13.1",
        "scipy",
        "/cache/liluchen/FastA2V/envs/eval/lib/python3.11/site-packages/scipy/__init__.py",
        "https://pypi.org/simple",
    ),
    (
        "tqdm",
        "4.67.1",
        "tqdm",
        "/cache/liluchen/FastA2V/envs/eval/lib/python3.11/site-packages/tqdm/__init__.py",
        "https://pypi.org/simple",
    ),
    (
        "pillow",
        "11.1.0",
        "PIL",
        "/cache/liluchen/FastA2V/envs/eval/lib/python3.11/site-packages/PIL/__init__.py",
        "https://pypi.org/simple",
    ),
)
WEIGHT_CONTRACT_FIELDS = (
    "weight_id",
    "path",
    "source_type",
    "source",
    "source_distribution",
    "source_version",
    "sha256_prefix",
)
EXPECTED_WEIGHT_CONTRACTS = (
    (
        "lpips_alex_v0.1_linear",
        "/cache/liluchen/FastA2V/envs/eval/lib/python3.11/site-packages/lpips/weights/v0.1/alex.pth",
        "installed_package",
        "https://pypi.org/project/lpips/0.1.4/",
        "lpips",
        "0.1.4",
        None,
    ),
    (
        "torchvision_alexnet_owt",
        "/cache/liluchen/FastA2V/checkpoints/eval/torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
        "url",
        "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        None,
        None,
        "7be5be79",
    ),
)
METHOD_REQUIRED_ENVIRONMENT = {
    "dense": {
        "run_kind": "dense_baseline",
        "attention_method": "dense",
        "use_cfg_cache": False,
        "use_block_cache": False,
    },
    "dense_cfg_cache": {
        "run_kind": "cfg_cache_benchmark",
        "attention_method": "dense",
        "use_cfg_cache": True,
        "use_block_cache": False,
    },
    "sparge_topk50": {
        "run_kind": "sparge_baseline",
        "attention_method": "sparge",
        "sparge_topk": 0.5,
        "use_cfg_cache": False,
        "use_block_cache": False,
    },
    "sparge_topk75": {
        "run_kind": "sparge_topk75_baseline",
        "attention_method": "sparge",
        "sparge_topk": 0.75,
        "use_cfg_cache": False,
        "use_block_cache": False,
    },
    "radial_conservative": {
        "run_kind": "radial_conservative_baseline",
        "attention_method": "radial",
        "radial_profile": "conservative",
        "use_cfg_cache": False,
        "use_block_cache": False,
    },
    "radial_aggressive": {
        "run_kind": "radial_aggressive_baseline",
        "attention_method": "radial",
        "radial_profile": "aggressive",
        "use_cfg_cache": False,
        "use_block_cache": False,
    },
    "best_sparse_cfg": {
        "use_cfg_cache": True,
        "use_block_cache": False,
    },
    "block_cache": {
        "block_cache_policy": "fixed",
        "use_cfg_cache": False,
        "use_block_cache": True,
    },
}


class QualityError(ValueError):
    """Raised when a quality result would not be auditable or comparable."""


_FIXED_MEDIA_MODULE: Any | None = None


def _media_module() -> Any:
    global _FIXED_MEDIA_MODULE
    if _FIXED_MEDIA_MODULE is not None:
        return _FIXED_MEDIA_MODULE
    path = (REPO_ROOT / "scripts" / "compare_media.py").resolve()
    try:
        module = _load_fixed_module("_fasta2v_fixed_compare_media", path)
    except Exception as exc:
        _fail("media dependencies", f"cannot import scripts/compare_media.py: {exc}")
    module_file = getattr(module, "__file__", None)
    _require(isinstance(module_file, str) and Path(module_file).resolve() == path, "media dependencies", "loaded compare_media module path differs from fixed script")
    _FIXED_MEDIA_MODULE = module
    return module


def _fail(context: str, message: str) -> None:
    raise QualityError(f"{context}: {message}")


def _require(condition: bool, context: str, message: str) -> None:
    if not condition:
        _fail(context, message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _read_json(path: Path, context: str) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _fail(context, f"cannot read strict JSON from {path}: {exc}")


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _fail(context, f"cannot read {path}: {exc}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        _require(bool(line.strip()), context, f"blank record at {path}:{line_number}")
        try:
            record = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            _fail(context, f"invalid strict JSON at {path}:{line_number}: {exc}")
        _require(isinstance(record, dict), context, f"record at {path}:{line_number} is not an object")
        records.append(record)
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise QualityError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    try:
        rendered = (
            json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QualityError(f"cannot write strict JSON to {path}: {exc}") from exc
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o644)
        view = memoryview(rendered)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise QualityError(f"cannot atomically create strict JSON {path}: {exc}") from exc


def _full_sha(value: Any, context: str, field: str) -> str:
    _require(
        isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None,
        context,
        f"{field} must be a lowercase full SHA256, found {value!r}",
    )
    return value


def _canonical_distribution_name(value: Any, context: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), context, "distribution name is missing")
    canonical = re.sub(r"[-_.]+", "-", value.strip()).lower()
    _require(canonical == value, context, f"distribution name must be canonical: {value!r}")
    return canonical


def _dependency_lock_records(
    packages: Iterable[Mapping[str, Any]],
    *,
    context: str,
) -> list[dict[str, str]]:
    """Return the canonical, complete archive trust payload for one eval env."""

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for package in packages:
        _require(isinstance(package, Mapping), context, "dependency record is not an object")
        distribution = _canonical_distribution_name(package.get("distribution"), context)
        _require(distribution not in seen, context, f"duplicate dependency {distribution}")
        seen.add(distribution)
        version = package.get("version")
        _require(isinstance(version, str) and bool(version), context, f"{distribution} version is missing")
        source_index = package.get("source_index")
        _require(
            source_index in {
                "https://download.pytorch.org/whl/cpu",
                "https://pypi.org/simple",
            },
            context,
            f"{distribution} has an unapproved source index",
        )
        archive_url = package.get("archive_url")
        _require(isinstance(archive_url, str), context, f"{distribution} archive URL is missing")
        try:
            _ARCHIVE_URL_POLICY.validate_dependency_archive_url(
                archive_url,
                expected_source_index=source_index,
            )
        except ValueError as exc:
            _fail(
                context,
                f"{distribution} archive URL violates the fixed source policy: {exc}",
            )
        records.append(
            {
                "distribution": distribution,
                "version": version,
                "source_index": source_index,
                "archive_url": archive_url,
                "archive_sha256": _full_sha(
                    package.get("archive_sha256"),
                    context,
                    f"{distribution}.archive_sha256",
                ),
            }
        )
    _require(bool(records), context, "dependency set is empty")
    return sorted(records, key=lambda item: item["distribution"])


def dependency_environment_lock_sha256(
    packages: Iterable[Mapping[str, Any]],
    *,
    context: str = "LPIPS dependency environment lock",
) -> str:
    payload = _dependency_lock_records(packages, context=context)
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _positive_int(value: Any, context: str, field: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        context,
        f"{field} must be a positive integer, found {value!r}",
    )
    return value


def _shape(value: Any, context: str, field: str) -> tuple[int, ...]:
    _require(isinstance(value, (list, tuple)), context, f"{field} must be a sequence")
    result = tuple(value)
    _require(bool(result), context, f"{field} must not be empty")
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in result),
        context,
        f"{field} must contain positive integers",
    )
    return result


def _distribution_record_errors(record_path: Path, environment_root: Path) -> list[str]:
    """Verify every hashed file in an installed wheel RECORD."""

    errors: list[str] = []
    site_packages = record_path.parent.parent
    try:
        with record_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, csv.Error) as exc:
        return [f"cannot read wheel RECORD {record_path}: {exc}"]
    if not rows:
        return [f"wheel RECORD is empty: {record_path}"]
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 3:
            errors.append(f"RECORD row {row_number} does not have three fields")
            continue
        relative_path, encoded_hash, size_text = row
        installed_path = (site_packages / relative_path).resolve()
        try:
            installed_path.relative_to(environment_root)
        except ValueError:
            errors.append(f"RECORD row {row_number} escapes eval environment: {relative_path}")
            continue
        if not encoded_hash:
            # PEP 376 permits installers to append unhashed generated bytecode
            # rows.  This protocol installs with --no-compile, runs with -B,
            # and requires those files to be absent; Python must never execute
            # an unauthenticated pyc.  RECORD itself is the only other allowed
            # unhashed row.
            is_generated_pyc = (
                installed_path.suffix == ".pyc"
                and "__pycache__" in installed_path.parts
            )
            if is_generated_pyc:
                if installed_path.exists() or installed_path.is_symlink():
                    errors.append(
                        f"unhashed generated bytecode must be absent: {installed_path}"
                    )
            elif installed_path != record_path.resolve():
                errors.append(f"RECORD row {row_number} omits hash for {relative_path}")
            continue
        try:
            algorithm, encoded_digest = encoded_hash.split("=", 1)
        except ValueError:
            errors.append(f"RECORD row {row_number} has malformed hash")
            continue
        if algorithm != "sha256":
            errors.append(f"RECORD row {row_number} uses unsupported {algorithm!r}")
            continue
        if not installed_path.is_file():
            errors.append(f"RECORD file is missing: {installed_path}")
            continue
        try:
            expected_size = int(size_text)
        except ValueError:
            errors.append(f"RECORD row {row_number} has invalid byte count {size_text!r}")
            continue
        actual_bytes = installed_path.stat().st_size
        if actual_bytes != expected_size:
            errors.append(f"RECORD byte count drifted for {installed_path}")
            continue
        padding = "=" * (-len(encoded_digest) % 4)
        try:
            expected_digest = base64.urlsafe_b64decode(encoded_digest + padding).hex()
        except (ValueError, TypeError):
            errors.append(f"RECORD row {row_number} has invalid base64 digest")
            continue
        if sha256(installed_path) != expected_digest:
            errors.append(f"RECORD SHA256 drifted for {installed_path}")
    return errors


def _wheel_archive_errors(
    archive_path: Path,
    site_packages: Path,
    environment_root: Path,
) -> list[str]:
    """Verify installed files against the RECORD inside a trusted wheel."""

    errors: list[str] = []
    try:
        with zipfile.ZipFile(archive_path) as wheel:
            record_names = [
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/RECORD")
            ]
            if len(record_names) != 1:
                return [f"trusted wheel contains {len(record_names)} RECORD files"]
            record_text = wheel.read(record_names[0]).decode("utf-8")
    except (OSError, zipfile.BadZipFile, KeyError, UnicodeDecodeError) as exc:
        return [f"cannot read trusted wheel {archive_path}: {exc}"]
    try:
        rows = list(csv.reader(record_text.splitlines()))
    except csv.Error as exc:
        return [f"cannot parse trusted wheel RECORD: {exc}"]
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 3:
            errors.append(f"wheel RECORD row {row_number} does not have three fields")
            continue
        relative_path, encoded_hash, size_text = row
        if not encoded_hash:
            # The wheel's own RECORD is deliberately unhashed and need not be
            # byte-identical to pip's installed RECORD additions.
            continue
        components = Path(relative_path).parts
        if len(components) >= 3 and components[0].endswith(".data"):
            category = components[1]
            remainder = Path(*components[2:])
            if category in {"purelib", "platlib"}:
                installed_path = (site_packages / remainder).resolve()
            else:
                # scripts/headers/data do not influence importable metric code
                # and installer shebang rewriting would change script bytes.
                continue
        else:
            installed_path = (site_packages / relative_path).resolve()
        try:
            installed_path.relative_to(environment_root)
        except ValueError:
            errors.append(f"wheel RECORD row {row_number} escapes eval environment")
            continue
        try:
            algorithm, encoded_digest = encoded_hash.split("=", 1)
        except ValueError:
            errors.append(f"wheel RECORD row {row_number} has malformed hash")
            continue
        if algorithm != "sha256":
            errors.append(f"wheel RECORD row {row_number} uses unsupported {algorithm!r}")
            continue
        if not installed_path.is_file():
            errors.append(f"trusted wheel file is missing: {installed_path}")
            continue
        try:
            expected_size = int(size_text)
        except ValueError:
            errors.append(f"wheel RECORD row {row_number} has invalid byte count")
            continue
        if installed_path.stat().st_size != expected_size:
            errors.append(f"trusted wheel byte count drifted for {installed_path}")
            continue
        padding = "=" * (-len(encoded_digest) % 4)
        try:
            expected_digest = base64.urlsafe_b64decode(encoded_digest + padding).hex()
        except (ValueError, TypeError):
            errors.append(f"wheel RECORD row {row_number} has invalid base64 digest")
            continue
        if sha256(installed_path) != expected_digest:
            errors.append(f"trusted wheel SHA256 drifted for {installed_path}")
    return errors


def _installed_distribution_versions(site_packages: Path) -> dict[str, str]:
    """Read installed metadata from one explicit path without importing code."""

    versions: dict[str, str] = {}
    try:
        distributions = importlib.metadata.distributions(path=[str(site_packages)])
        for distribution in distributions:
            raw_name = distribution.metadata.get("Name")
            canonical = re.sub(r"[-_.]+", "-", str(raw_name).strip()).lower()
            if not canonical or canonical in versions:
                raise ValueError(f"missing or duplicate distribution metadata name {raw_name!r}")
            versions[canonical] = distribution.version
    except Exception as exc:
        raise QualityError(f"cannot enumerate fixed eval distributions: {exc}") from exc
    return versions


def _wheel_site_package_paths(
    archive_path: Path,
    site_packages: Path,
    environment_root: Path,
) -> tuple[set[Path], list[str]]:
    """Map a trusted wheel RECORD to the files it may own in site-packages."""

    allowed: set[Path] = set()
    errors: list[str] = []
    try:
        with zipfile.ZipFile(archive_path) as wheel:
            record_names = [
                name for name in wheel.namelist() if name.endswith(".dist-info/RECORD")
            ]
            if len(record_names) != 1:
                return set(), [f"trusted wheel contains {len(record_names)} RECORD files"]
            rows = list(csv.reader(wheel.read(record_names[0]).decode("utf-8").splitlines()))
    except (OSError, zipfile.BadZipFile, KeyError, UnicodeDecodeError, csv.Error) as exc:
        return set(), [f"cannot derive trusted wheel file set from {archive_path}: {exc}"]
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 3:
            errors.append(f"wheel RECORD row {row_number} does not have three fields")
            continue
        relative_path = Path(row[0])
        components = relative_path.parts
        if len(components) >= 3 and components[0].endswith(".data"):
            if components[1] not in {"purelib", "platlib"}:
                continue
            installed_path = (site_packages / Path(*components[2:])).resolve()
        else:
            installed_path = (site_packages / relative_path).resolve()
        try:
            installed_path.relative_to(environment_root)
        except ValueError:
            errors.append(f"wheel RECORD row {row_number} escapes eval environment")
            continue
        allowed.add(installed_path)
    return allowed, errors


def _site_packages_tree_errors(
    packages: Iterable[Mapping[str, Any]],
    environment_root: Path,
) -> list[str]:
    """Reject every file not owned by a trusted wheel or fixed pip metadata."""

    package_list = list(packages)
    errors: list[str] = []
    record_paths = [Path(str(item.get("record_path", ""))).resolve() for item in package_list]
    if not record_paths:
        return ["dependency receipt contains no installed RECORD paths"]
    site_roots = {path.parent.parent for path in record_paths}
    if len(site_roots) != 1:
        return ["dependency RECORD paths do not share one site-packages root"]
    site_packages = next(iter(site_roots))
    expected_site_packages = (
        environment_root / "lib" / "python3.11" / "site-packages"
    ).resolve()
    if site_packages != expected_site_packages:
        return [f"site-packages root differs from fixed path: {site_packages}"]
    allowed: set[Path] = set()
    generated_metadata: dict[Path, bytes] = {}
    for package, record_path in zip(package_list, record_paths):
        archive_path = Path(str(package.get("archive_path", ""))).resolve()
        wheel_allowed, wheel_errors = _wheel_site_package_paths(
            archive_path,
            site_packages,
            environment_root,
        )
        errors.extend(wheel_errors)
        allowed.update(wheel_allowed)
        generated_metadata[record_path.parent / "INSTALLER"] = b"pip\n"
        generated_metadata[record_path.parent / "REQUESTED"] = b""
    if errors:
        return errors
    try:
        entries = list(site_packages.rglob("*"))
    except OSError as exc:
        return [f"cannot scan fixed site-packages tree: {exc}"]
    for path in entries:
        if path.is_symlink():
            errors.append(f"site-packages symlink is forbidden: {path}")
            continue
        if path.is_dir():
            continue
        resolved = path.resolve()
        if resolved.suffix == ".pyc" or "__pycache__" in resolved.parts:
            errors.append(f"compiled bytecode is forbidden in fixed eval env: {resolved}")
            continue
        if resolved in allowed:
            continue
        expected_bytes = generated_metadata.get(resolved)
        if expected_bytes is None:
            errors.append(f"unowned site-packages file is forbidden: {resolved}")
            continue
        try:
            actual_bytes = resolved.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read generated pip metadata {resolved}: {exc}")
            continue
        if actual_bytes != expected_bytes:
            errors.append(f"generated pip metadata has unexpected content: {resolved}")
    return errors


def _activate_fixed_site_packages(site_packages: Path) -> None:
    """Expose the audited dependency tree only after the pre-import checks."""

    fixed = site_packages.resolve()
    for entry in sys.path:
        if not entry:
            continue
        resolved = Path(entry).resolve()
        if resolved.name in {"site-packages", "dist-packages"} and resolved != fixed:
            raise QualityError(f"foreign site-packages path is active: {resolved}")
    if str(fixed) not in sys.path:
        sys.path.insert(0, str(fixed))


def load_quality_protocol(path: Path = DEFAULT_PROTOCOL) -> tuple[dict[str, Any], str]:
    path = Path(path)
    protocol = _read_json(path, "quality protocol")
    _require(isinstance(protocol, dict), "quality protocol", "root must be an object")
    _require(protocol.get("schema_version") == QUALITY_SCHEMA_VERSION, "quality protocol", "unsupported schema_version")
    _require(
        protocol.get("protocol_id") == "ovi_720x720_5s_dense_pair_quality_v2",
        "quality protocol",
        "protocol_id is not the fixed Ovi quality protocol",
    )
    _require(protocol.get("reference_method_id") == "dense", "quality protocol", "reference method must be dense")
    _require(tuple(protocol.get("measurement_indices", ())) == EXPECTED_INDICES, "quality protocol", "measurement indices must be exactly 0,1,2")
    _require(tuple(protocol.get("pairing_key", ())) == IDENTITY_FIELDS, "quality protocol", "pairing key must be measurement_index,prompt_index,sample_index")
    _require(protocol.get("require_artifact_sha256_before_and_after_metrics") is True, "quality protocol", "pre/post artifact hashing must remain enabled")
    _require(tuple(protocol.get("required_same_across_runs", ())) == EXPECTED_SAME_FIELDS, "quality protocol", "cross-run comparison fields changed")

    media_protocol = protocol.get("media_metrics")
    _require(isinstance(media_protocol, dict), "quality protocol", "media_metrics is missing")
    _require(media_protocol.get("implementation") == "scripts/compare_media.py", "quality protocol", "media implementation changed")
    _require(media_protocol.get("frame_policy") == "exact_all_decoded_frames", "quality protocol", "video frame policy must be exact")
    _require(tuple(media_protocol.get("video_filters", ())) == ("psnr", "ssim"), "quality protocol", "video metrics changed")
    _require(
        tuple(media_protocol.get("audio_metrics", ()))
        == ("rmse", "max_abs_difference", "snr_db", "correlation"),
        "quality protocol",
        "audio metrics changed",
    )
    _require(media_protocol.get("automatic_acceptance_thresholds") is None, "quality protocol", "sparse acceptance thresholds must not be invented")
    audio_decode = media_protocol.get("audio_decode")
    _require(
        audio_decode == {
            "channels": 1,
            "sample_rate_hz": 16000,
            "sample_format": "f32le",
            "sample_count_policy": "exact",
        },
        "quality protocol",
        "audio decode contract changed",
    )

    lpips_protocol = protocol.get("lpips")
    _require(isinstance(lpips_protocol, dict), "quality protocol", "lpips section is missing")
    expected_lpips = {
        "implementation": "lpips.LPIPS",
        "network": "alex",
        "model_version": "0.1",
        "device": "cpu",
        "spatial": False,
        "batch_size": 1,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "torch_deterministic_algorithms": True,
        "torch_mkldnn_enabled": False,
        "input_range": "[-1,1]",
        "frame_policy": "exact_all_decoded_rgb24_frames",
        "python_executable": "/cache/liluchen/FastA2V/envs/eval/bin/python",
        "environment_root": "/cache/liluchen/FastA2V/envs/eval",
        "torch_home": "/cache/liluchen/FastA2V/checkpoints/eval/torch",
        "receipt_path": "/cache/liluchen/FastA2V/checkpoints/eval/lpips_alex_v0.1_receipt.json",
    }
    for field, expected in expected_lpips.items():
        _require(lpips_protocol.get(field) == expected, "quality protocol", f"LPIPS {field} must remain {expected!r}")
    packages = lpips_protocol.get("packages")
    weights = lpips_protocol.get("weights")
    _require(isinstance(packages, list) and len(packages) == 7, "quality protocol", "exactly seven LPIPS dependency packages are required")
    _require(isinstance(weights, list) and len(weights) == 2, "quality protocol", "exactly two LPIPS weight records are required")
    _require(all(isinstance(item, dict) for item in packages), "quality protocol", "every LPIPS package entry must be an object")
    _require(all(isinstance(item, dict) for item in weights), "quality protocol", "every LPIPS weight entry must be an object")
    package_contracts = tuple(
        tuple(item.get(field) for field in PACKAGE_CONTRACT_FIELDS)
        for item in packages
    )
    _require(package_contracts == EXPECTED_PACKAGE_CONTRACTS, "quality protocol", "LPIPS package version/module/path/source contract changed")
    weight_contracts = tuple(
        tuple(item.get(field) for field in WEIGHT_CONTRACT_FIELDS)
        for item in weights
    )
    _require(weight_contracts == EXPECTED_WEIGHT_CONTRACTS, "quality protocol", "LPIPS weight path/source contract changed")
    for item in packages:
        trusted_hash = item.get("trusted_archive_sha256")
        _require(trusted_hash is None or (isinstance(trusted_hash, str) and HEX_SHA256.fullmatch(trusted_hash) is not None), "quality protocol", f"{item['distribution']} trusted archive hash is invalid")
    for item in weights:
        trusted_hash = item.get("trusted_sha256")
        _require(trusted_hash is None or (isinstance(trusted_hash, str) and HEX_SHA256.fullmatch(trusted_hash) is not None), "quality protocol", f"{item['weight_id']} trusted weight hash is invalid")
    environment_lock = lpips_protocol.get("trusted_environment_lock_sha256")
    environment_packages = lpips_protocol.get("trusted_environment_packages")
    _require(
        environment_lock is None
        or (
            isinstance(environment_lock, str)
            and HEX_SHA256.fullmatch(environment_lock) is not None
        ),
        "quality protocol",
        "trusted environment lock SHA256 is invalid",
    )
    _require(
        environment_packages is None or isinstance(environment_packages, list),
        "quality protocol",
        "trusted environment package lock must be null or a list",
    )
    lock_values = [environment_lock] + [item.get("trusted_archive_sha256") for item in packages] + [
        item.get("trusted_sha256") for item in weights
    ]
    if all(value is None for value in lock_values) and environment_packages is None:
        _require(lpips_protocol.get("trusted_lock_status") == "bootstrap_unpinned", "quality protocol", "unpopulated trust lock must be marked bootstrap_unpinned")
    elif (
        all(isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None for value in lock_values)
        and isinstance(environment_packages, list)
    ):
        normalized_environment_packages = _dependency_lock_records(
            environment_packages,
            context="quality protocol dependency lock",
        )
        _require(
            environment_packages == normalized_environment_packages,
            "quality protocol",
            "trusted environment packages must use canonical sorted lock records",
        )
        _require(
            dependency_environment_lock_sha256(environment_packages)
            == environment_lock,
            "quality protocol",
            "trusted environment package payload does not match its SHA256",
        )
        locked_by_distribution = {
            item["distribution"]: item for item in environment_packages
        }
        _require(
            set(item["distribution"] for item in packages).issubset(locked_by_distribution),
            "quality protocol",
            "trusted environment package payload omits a direct dependency",
        )
        for item in packages:
            locked = locked_by_distribution[item["distribution"]]
            _require(
                locked["version"] == item["version"]
                and locked["source_index"] == item["source_index"]
                and locked["archive_sha256"] == item["trusted_archive_sha256"],
                "quality protocol",
                f"{item['distribution']} direct trust root differs from full environment lock",
            )
        _require(lpips_protocol.get("trusted_lock_status") == "pinned", "quality protocol", "complete trust lock must be marked pinned")
    else:
        _fail("quality protocol", "dependency trust lock is partially populated; pin every archive and weight together")

    manual = protocol.get("manual_reviews")
    _require(isinstance(manual, dict), "quality protocol", "manual review contract is missing")
    _require(manual.get("template") == "eval/manual_sync_reviews.csv", "quality protocol", "manual template path changed")
    _require(tuple(manual.get("fields", ())) == MANUAL_FIELDS, "quality protocol", "manual CSV fields changed")
    _require(tuple(manual.get("required_human_fields", ())) == ("reviewer", "reviewed_at_utc", "sync_rating"), "quality protocol", "required human review fields changed")
    _require(tuple(manual.get("allowed_sync_ratings", ())) == ("pass", "fail", "uncertain"), "quality protocol", "manual rating vocabulary changed")
    _require(manual.get("row_policy") == "zero_rows_or_all_artifacts", "quality protocol", "manual row policy changed")
    _require(manual.get("candidate_hash_binding_required") is True, "quality protocol", "candidate hash binding must stay required")
    _require(manual.get("automatic_population_forbidden") is True, "quality protocol", "manual fields must never be automatically populated")
    return protocol, sha256(path)


@dataclass(frozen=True)
class MeasurementArtifact:
    measurement_index: int
    prompt_index: int
    sample_index: int
    path: Path
    sha256: str
    metrics_sidecar_path: Path
    metrics_sidecar_sha256: str
    prompt: str
    seed: int
    requested_shape: tuple[int, ...]
    actual_shape: tuple[int, ...]
    generated_video_shape: tuple[int, ...]
    generated_audio_shape: tuple[int, ...]
    sample_steps: int

    @property
    def identity(self) -> tuple[int, int, int]:
        return (
            self.measurement_index,
            self.prompt_index,
            self.sample_index,
        )


@dataclass(frozen=True)
class AuditedRun:
    method_id: str
    run_dir: Path
    run_id: str
    verification_sha256: str
    timings_path: Path
    timings_bytes: int
    timings_sha256: str
    timings_record_count: int
    warmup_timings_path: Path
    warmup_timings_bytes: int
    warmup_timings_sha256: str
    warmup_record_count: int
    environment_sha256: str
    git_commit: str
    checkpoint_manifest_sha256: str
    checkpoint_fingerprint_sha256: str
    gpu_identity: tuple[int, str, str]
    prompt_set_sha256: str
    prompt_count: int
    prompts: tuple[str, ...]
    base_seed: int
    sample_count: int
    sample_seeds: tuple[int, ...]
    selected_sparse_profile: str
    requested_shape: tuple[int, ...]
    actual_shape: tuple[int, ...]
    generated_video_shape: tuple[int, ...]
    generated_audio_shape: tuple[int, ...]
    sample_steps: int
    acceleration_environment: Mapping[str, Any]
    artifacts: Mapping[Any, MeasurementArtifact]
    evidence_bindings: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )

    def comparison_values(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "checkpoint_fingerprint_sha256": self.checkpoint_fingerprint_sha256,
            "gpu_identity": self.gpu_identity,
            "prompt_set_sha256": self.prompt_set_sha256,
            "prompt_count": self.prompt_count,
            "prompts": self.prompts,
            "base_seed": self.base_seed,
            "sample_count": self.sample_count,
            "sample_seeds": self.sample_seeds,
            "requested_shape": self.requested_shape,
            "actual_shape": self.actual_shape,
            "generated_video_shape": self.generated_video_shape,
            "generated_audio_shape": self.generated_audio_shape,
            "sample_steps": self.sample_steps,
        }

    def sidecar_binding(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "verification_sha256": self.verification_sha256,
            "timings_path": str(self.timings_path),
            "timings_bytes": self.timings_bytes,
            "timings_sha256": self.timings_sha256,
            "timings_record_count": self.timings_record_count,
            "warmup_timings_path": str(self.warmup_timings_path),
            "warmup_timings_bytes": self.warmup_timings_bytes,
            "warmup_timings_sha256": self.warmup_timings_sha256,
            "warmup_record_count": self.warmup_record_count,
            "environment_sha256": self.environment_sha256,
            "git_commit": self.git_commit,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_fingerprint_sha256": self.checkpoint_fingerprint_sha256,
            "gpu_physical_index": self.gpu_identity[0],
            "gpu_uuid": self.gpu_identity[1],
            "gpu_name": self.gpu_identity[2],
            "prompt_set_sha256": self.prompt_set_sha256,
            "prompt_count": self.prompt_count,
            "prompts": list(self.prompts),
            "base_seed": self.base_seed,
            "sample_count": self.sample_count,
            "sample_seeds": list(self.sample_seeds),
            "selected_sparse_profile": self.selected_sparse_profile,
            "requested_shape": list(self.requested_shape),
            "actual_shape": list(self.actual_shape),
            "generated_video_shape": list(self.generated_video_shape),
            "generated_audio_shape": list(self.generated_audio_shape),
            "sample_steps": self.sample_steps,
            "acceleration_environment": dict(self.acceleration_environment),
            "evidence_bindings": {
                name: dict(binding)
                for name, binding in sorted(self.evidence_bindings.items())
            },
        }


def _find_method(matrix: Mapping[str, Any], method_id: str) -> dict[str, Any]:
    _require(method_id in METHOD_REQUIRED_ENVIRONMENT, "evaluation matrix", f"method_id {method_id!r} is outside the fixed quality slots")
    matches = [method for method in matrix["methods"] if method.get("method_id") == method_id]
    _require(len(matches) == 1, "evaluation matrix", f"method_id {method_id!r} is not unique and ready")
    method = matches[0]
    _require(
        method.get("implementation_status") == "ready",
        "evaluation matrix",
        f"method_id {method_id!r} is not implementation-ready",
    )
    actual_environment = method.get("expected_environment")
    _require(isinstance(actual_environment, Mapping), "evaluation matrix", f"{method_id} expected_environment is missing")
    for field, expected in METHOD_REQUIRED_ENVIRONMENT[method_id].items():
        _require(actual_environment.get(field) == expected, "evaluation matrix", f"{method_id} {field} was relabeled away from fixed method contract")
    validator = _run_validator_module()
    combo_run_kinds = validator.COMBO_METHOD_RUN_KINDS.get(method_id)
    if combo_run_kinds is not None:
        _require(
            method.get("selection_required") is True,
            "evaluation matrix",
            f"{method_id} no longer requires explicit sparse-profile selection",
        )
        _require(
            isinstance(method.get("allowed_run_kinds"), list)
            and tuple(method["allowed_run_kinds"]) == combo_run_kinds,
            "evaluation matrix",
            f"{method_id} allowed_run_kinds differ from the four fixed profiles",
        )
        _require(
            dict(actual_environment) == METHOD_REQUIRED_ENVIRONMENT[method_id],
            "evaluation matrix",
            f"{method_id} generic cache contract contains profile relabeling",
        )
    return method


def _selected_sparse_profile_for_environment(
    method: Mapping[str, Any],
    environment: Mapping[str, Any],
    context: str,
) -> str:
    """Revalidate a G/H run-kind profile and return its canonical selection."""

    validator = _run_validator_module()
    if method.get("method_id") not in validator.COMBO_METHOD_RUN_KINDS:
        return ""
    try:
        validator._validate_combo_run_environment(method, environment, context)
    except Exception as exc:
        _fail(context, f"sparse selection contract is invalid: {exc}")
    run_kind = environment.get("run_kind")
    profile = validator.SPARSE_PROFILE_BY_RUN_KIND.get(run_kind)
    _require(
        isinstance(profile, str) and profile,
        context,
        f"cannot derive selected sparse profile from run_kind={run_kind!r}",
    )
    return profile


def validate_selected_sparse_profile_consistency(
    runs: Iterable[AuditedRun],
    context: str = "sparse selection quality",
) -> str:
    """Require supplied G/H quality inputs to select the same sparse profile."""

    selected = {
        run.method_id: run.selected_sparse_profile
        for run in runs
        if run.method_id in {"best_sparse_cfg", "block_cache"}
    }
    for method_id, profile in selected.items():
        _require(
            isinstance(profile, str) and profile,
            context,
            f"{method_id} does not bind a selected sparse profile",
        )
    if {"best_sparse_cfg", "block_cache"}.issubset(selected):
        _require(
            selected["best_sparse_cfg"] == selected["block_cache"],
            context,
            "best_sparse_cfg and block_cache selected different sparse profiles",
        )
    return next(iter(selected.values()), "")


def _artifact_identity_map(
    run: AuditedRun,
    context: str,
) -> dict[tuple[int, int, int], MeasurementArtifact]:
    """Normalize artifact mappings while rejecting any relabelled identity."""

    normalized: dict[tuple[int, int, int], MeasurementArtifact] = {}
    for supplied_key, artifact in run.artifacts.items():
        _require(
            isinstance(artifact, MeasurementArtifact),
            context,
            "artifact mapping contains a non-artifact value",
        )
        identity = artifact.identity
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in identity),
            context,
            f"artifact identity is invalid: {identity!r}",
        )
        key_matches = supplied_key == identity or (
            supplied_key == identity[0]
            and identity[1:] == (0, 0)
        )
        _require(
            key_matches,
            context,
            f"artifact mapping key {supplied_key!r} differs from identity {identity!r}",
        )
        _require(
            identity not in normalized,
            context,
            f"duplicate artifact identity {identity!r}",
        )
        normalized[identity] = artifact
    return normalized


def load_audited_run(
    run_dir: Path,
    method_id: str,
    matrix: Mapping[str, Any],
) -> AuditedRun:
    """Validate a formal performance run, then expose hash-bound measurements."""

    run_dir = Path(run_dir).resolve()
    method = _find_method(matrix, method_id)
    summary = _run_validator_module().validate_run(
        method,
        run_dir,
        matrix["fixed_protocol"],
    )
    validator = _run_validator_module()
    environment_path = run_dir / "environment.json"
    environment_snapshot, environment = validator._snapshot_json(
        environment_path,
        f"{method_id} quality input",
    )
    _require(isinstance(environment, dict), f"{method_id} quality input", "environment.json must contain an object")
    selected_sparse_profile = _selected_sparse_profile_for_environment(
        method,
        environment,
        f"{method_id} quality input",
    )
    _require(
        summary.get("selected_sparse_profile", "") == selected_sparse_profile,
        f"{method_id} quality input",
        "selected sparse profile differs from performance validation",
    )
    timings_snapshot, records = validator._snapshot_jsonl(
        run_dir / "timings.jsonl",
        f"{method_id} quality input",
    )
    warmup_snapshot, _warmup_records = validator._snapshot_jsonl(
        run_dir / "warmup_timings.jsonl",
        f"{method_id} quality input",
    )
    verification_snapshot, _verification = validator._snapshot_json(
        run_dir / "verification.json",
        f"{method_id} quality input",
    )
    checkpoint_snapshot, _checkpoint = validator._snapshot_json(
        run_dir / "checkpoint_manifest.json",
        f"{method_id} quality input",
    )
    evidence_snapshots_by_name = {
        "environment.json": environment_snapshot,
        "verification.json": verification_snapshot,
        "timings.jsonl": timings_snapshot,
        "warmup_timings.jsonl": warmup_snapshot,
        "checkpoint_manifest.json": checkpoint_snapshot,
    }
    evidence_hashes = environment.get("evidence_file_sha256")
    _require(
        isinstance(evidence_hashes, Mapping) and evidence_hashes,
        f"{method_id} quality input",
        "environment evidence_file_sha256 is missing",
    )
    for filename, expected_hash in sorted(evidence_hashes.items()):
        _require(
            isinstance(filename, str)
            and filename
            and Path(filename).name == filename
            and filename not in {".", ".."},
            f"{method_id} quality input",
            f"invalid evidence filename {filename!r}",
        )
        expected_hash = _full_sha(
            expected_hash,
            f"{method_id} quality input",
            f"{filename} evidence SHA256",
        )
        snapshot = evidence_snapshots_by_name.get(filename)
        if snapshot is None:
            try:
                snapshot = validator._stable_file_snapshot(
                    run_dir / filename,
                    f"{method_id} quality input",
                )
            except Exception as exc:
                _fail(
                    f"{method_id} quality input",
                    f"cannot snapshot hash-bound evidence {filename}: {exc}",
                )
            evidence_snapshots_by_name[filename] = snapshot
        _require(
            snapshot.sha256 == expected_hash,
            f"{method_id} quality input",
            f"{filename} differs from environment evidence hash",
        )
    run_config_hash = _full_sha(
        environment.get("run_config_sha256"),
        f"{method_id} quality input",
        "run_config.yaml SHA256",
    )
    try:
        run_config_snapshot = validator._stable_file_snapshot(
            run_dir / "run_config.yaml",
            f"{method_id} quality input",
        )
    except Exception as exc:
        _fail(
            f"{method_id} quality input",
            f"cannot snapshot hash-bound evidence run_config.yaml: {exc}",
        )
    _require(
        run_config_snapshot.sha256 == run_config_hash,
        f"{method_id} quality input",
        "run_config.yaml differs from environment hash",
    )
    evidence_snapshots_by_name["run_config.yaml"] = run_config_snapshot
    _require(timings_snapshot.sha256 == summary["timings_sha256"], f"{method_id} quality input", "timings.jsonl drifted after performance validation")
    _require(verification_snapshot.sha256 == summary["verification_sha256"], f"{method_id} quality input", "verification.json drifted after performance validation")
    _require(checkpoint_snapshot.sha256 == summary["checkpoint_manifest_sha256"], f"{method_id} quality input", "checkpoint manifest drifted after performance validation")
    _require(
        str(timings_snapshot.path) == summary["timings_path"]
        and timings_snapshot.size == summary["timings_bytes"]
        and isinstance(summary.get("timings_record_count"), int)
        and not isinstance(summary.get("timings_record_count"), bool)
        and len(records) == summary["timings_record_count"],
        f"{method_id} quality input",
        "timings stable snapshot differs from performance validation",
    )
    _require(
        str(warmup_snapshot.path) == summary["warmup_timings_path"]
        and warmup_snapshot.size == summary["warmup_timings_bytes"]
        and warmup_snapshot.sha256 == summary["warmup_timings_sha256"]
        and isinstance(summary.get("warmup_record_count"), int)
        and not isinstance(summary.get("warmup_record_count"), bool)
        and len(_warmup_records) == summary["warmup_record_count"],
        f"{method_id} quality input",
        "warmup timings stable snapshot differs from performance validation",
    )

    prompt_count = summary["prompt_count"]
    sample_count = summary["comparison_values"]["sample_count"]
    expected_identities = {
        (measurement_index, prompt_index, sample_index)
        for measurement_index in EXPECTED_MEASUREMENT_INDICES
        for prompt_index in range(prompt_count)
        for sample_index in range(sample_count)
    }
    _require(
        len(records) == len(expected_identities),
        f"{method_id} quality input",
        "formal quality comparison requires the complete artifact identity matrix",
    )

    artifacts: dict[tuple[int, int, int], MeasurementArtifact] = {}
    artifact_snapshots = []
    for record in records:
        context = f"{method_id} measurement"
        index = record.get("measurement_index")
        prompt_index = record.get("prompt_index")
        sample_index = record.get("sample_index")
        identity = (index, prompt_index, sample_index)
        _require(identity in expected_identities, context, f"artifact identity is outside the fixed matrix: {identity!r}")
        _require(identity not in artifacts, context, f"duplicate artifact identity {identity!r}")
        output_value = record.get("output_path")
        _require(isinstance(output_value, str) and output_value, context, "output_path is missing")
        output = Path(output_value)
        expected_hash = _full_sha(record.get("output_sha256"), context, "output_sha256")
        _require(
            output.is_absolute()
            and output.parent == run_dir
            and output == run_dir / output.name,
            context,
            "artifact escaped the selected run or used a non-canonical path",
        )
        output_snapshot = validator._stable_file_snapshot(output, context)
        _require(output_snapshot.sha256 == expected_hash, context, "artifact hash drifted after run validation")
        metrics_sidecar = output.with_suffix(".metrics.json")
        metrics_snapshot, metrics_payload = validator._snapshot_json(
            metrics_sidecar,
            context,
        )
        _require(metrics_payload == record, context, "metrics sidecar differs from timings record")
        artifact_snapshots.extend((output_snapshot, metrics_snapshot))
        prompt = record.get("prompt")
        seed = record.get("seed")
        sample_steps = record.get("sample_steps")
        _require(isinstance(prompt, str) and prompt, context, "prompt is missing")
        _require(isinstance(seed, int) and not isinstance(seed, bool), context, "seed is invalid")
        _positive_int(sample_steps, context, "sample_steps")
        _require(isinstance(prompt_index, int) and not isinstance(prompt_index, bool) and prompt_index >= 0, context, "prompt_index is invalid")
        _require(isinstance(sample_index, int) and not isinstance(sample_index, bool) and sample_index >= 0, context, "sample_index is invalid")
        artifacts[identity] = MeasurementArtifact(
            measurement_index=index,
            prompt_index=prompt_index,
            sample_index=sample_index,
            path=output,
            sha256=expected_hash,
            metrics_sidecar_path=metrics_sidecar,
            metrics_sidecar_sha256=metrics_snapshot.sha256,
            prompt=prompt,
            seed=seed,
            requested_shape=_shape(record.get("requested_video_frame_height_width"), context, "requested shape"),
            actual_shape=_shape(record.get("actual_video_frame_height_width"), context, "actual shape"),
            generated_video_shape=_shape(record.get("generated_video_shape"), context, "generated video shape"),
            generated_audio_shape=_shape(record.get("generated_audio_shape"), context, "generated audio shape"),
            sample_steps=sample_steps,
        )

    _require(set(artifacts) == expected_identities, f"{method_id} quality input", "artifacts do not form the complete identity matrix")
    comparison = summary["comparison_values"]
    acceleration_environment = {
        key: value
        for key, value in sorted(environment.items())
        if key
        in {
            "run_kind",
            "attention_method",
            "use_cfg_cache",
            "use_block_cache",
        }
        or key.startswith(("sparge_", "radial_", "cfg_cache_", "block_cache_"))
    }
    audited = AuditedRun(
        method_id=method_id,
        run_dir=run_dir,
        run_id=summary["run_id"],
        verification_sha256=_full_sha(summary["verification_sha256"], method_id, "verification_sha256"),
        timings_path=timings_snapshot.path,
        timings_bytes=timings_snapshot.size,
        timings_sha256=_full_sha(summary["timings_sha256"], method_id, "timings_sha256"),
        timings_record_count=summary["timings_record_count"],
        warmup_timings_path=warmup_snapshot.path,
        warmup_timings_bytes=warmup_snapshot.size,
        warmup_timings_sha256=_full_sha(summary["warmup_timings_sha256"], method_id, "warmup_timings_sha256"),
        warmup_record_count=summary["warmup_record_count"],
        environment_sha256=environment_snapshot.sha256,
        git_commit=summary["git_commit"],
        checkpoint_manifest_sha256=_full_sha(summary["checkpoint_manifest_sha256"], method_id, "checkpoint_manifest_sha256"),
        checkpoint_fingerprint_sha256=_full_sha(summary["checkpoint_fingerprint_sha256"], method_id, "checkpoint_fingerprint_sha256"),
        gpu_identity=tuple(comparison["gpu_identity"]),
        prompt_set_sha256=_full_sha(summary["prompt_set_sha256"], method_id, "prompt_set_sha256"),
        prompt_count=summary["prompt_count"],
        prompts=tuple(comparison["prompts"]),
        base_seed=comparison["base_seed"],
        sample_count=comparison["sample_count"],
        sample_seeds=tuple(comparison["sample_seeds"]),
        selected_sparse_profile=selected_sparse_profile,
        requested_shape=tuple(comparison["requested_shape"]),
        actual_shape=tuple(comparison["actual_shape"]),
        generated_video_shape=tuple(comparison["generated_video_shape"]),
        generated_audio_shape=tuple(comparison["generated_audio_shape"]),
        sample_steps=summary["sample_steps"],
        acceleration_environment=acceleration_environment,
        artifacts=artifacts,
        evidence_bindings={
            name: {
                "path": str(snapshot.path),
                "bytes": snapshot.size,
                "sha256": snapshot.sha256,
            }
            for name, snapshot in sorted(evidence_snapshots_by_name.items())
        },
    )
    for snapshot in (
        *evidence_snapshots_by_name.values(),
        *artifact_snapshots,
    ):
        validator._revalidate_snapshot(snapshot, f"{method_id} quality input")
    return audited


def _artifact_binding(run: AuditedRun, artifact: MeasurementArtifact) -> dict[str, Any]:
    binding = run.sidecar_binding()
    binding.update(
        {
            "measurement_index": artifact.measurement_index,
            "prompt_index": artifact.prompt_index,
            "sample_index": artifact.sample_index,
            "artifact_path": str(artifact.path),
            "artifact_sha256": artifact.sha256,
            "metrics_sidecar_path": str(artifact.metrics_sidecar_path),
            "metrics_sidecar_sha256": artifact.metrics_sidecar_sha256,
        }
    )
    return binding


def _validate_pairing(dense: AuditedRun, candidate: AuditedRun, protocol: Mapping[str, Any]) -> None:
    _require(dense.method_id == protocol["reference_method_id"], "quality pairing", "the explicit reference run is not dense")
    _require(candidate.method_id != dense.method_id, "quality pairing", "candidate method must differ from dense")
    for field in protocol["required_same_across_runs"]:
        expected = dense.comparison_values()[field]
        actual = candidate.comparison_values()[field]
        _require(actual == expected, "quality pairing", f"candidate {field}={actual!r} differs from dense={expected!r}")
    dense_artifacts = _artifact_identity_map(dense, "quality pairing dense")
    candidate_artifacts = _artifact_identity_map(candidate, "quality pairing candidate")
    _require(
        set(dense_artifacts) == set(candidate_artifacts),
        "quality pairing",
        "candidate artifact identities differ from dense",
    )
    _require(
        set(dense_artifacts)
        == {
            (measurement_index, prompt_index, sample_index)
            for measurement_index in EXPECTED_MEASUREMENT_INDICES
            for prompt_index in range(dense.prompt_count)
            for sample_index in range(dense.sample_count)
        },
        "quality pairing",
        "dense identities do not form the complete artifact matrix",
    )
    for identity in sorted(dense_artifacts):
        left = dense_artifacts[identity]
        right = candidate_artifacts[identity]
        for field in (
            "measurement_index",
            "prompt_index",
            "sample_index",
            "prompt",
            "seed",
            "requested_shape",
            "actual_shape",
            "generated_video_shape",
            "generated_audio_shape",
            "sample_steps",
        ):
            _require(getattr(left, field) == getattr(right, field), f"quality pair {identity}", f"candidate {field} differs from dense")


def validate_lpips_receipt(
    lpips_protocol: Mapping[str, Any],
    *,
    receipt_path: Path | None = None,
    import_module: Callable[[str], Any] | None = None,
    distribution_version: Callable[[str], str] | None = None,
    executable: str | None = None,
    prefix: str | None = None,
    runtime_flags: Any | None = None,
    installed_distributions: Callable[[Path], Mapping[str, str]] = _installed_distribution_versions,
    installed_record_validator: Callable[[Path, Path], list[str]] = _distribution_record_errors,
    wheel_record_validator: Callable[[Path, Path, Path], list[str]] = _wheel_archive_errors,
    site_packages_validator: Callable[[Iterable[Mapping[str, Any]], Path], list[str]] = _site_packages_tree_errors,
    site_packages_activator: Callable[[Path], None] = _activate_fixed_site_packages,
) -> dict[str, Any]:
    """Validate the complete CPU scoring environment before importing it."""

    context = "LPIPS dependency receipt"
    trusted_environment_lock = lpips_protocol.get("trusted_environment_lock_sha256")
    trusted_environment_packages = lpips_protocol.get("trusted_environment_packages")
    _require(
        isinstance(trusted_environment_lock, str)
        and HEX_SHA256.fullmatch(trusted_environment_lock) is not None,
        context,
        "complete dependency environment lock is not pinned in the checked-in quality protocol",
    )
    _require(
        isinstance(trusted_environment_packages, list)
        and bool(trusted_environment_packages),
        context,
        "complete dependency package payload is not pinned in the checked-in quality protocol",
    )
    for package in lpips_protocol.get("packages", []):
        distribution = package.get("distribution", "unknown")
        trusted_hash = package.get("trusted_archive_sha256")
        _require(
            isinstance(trusted_hash, str)
            and HEX_SHA256.fullmatch(trusted_hash) is not None,
            context,
            f"{distribution} trusted archive SHA256 is not pinned in the checked-in quality protocol",
        )
    for weight in lpips_protocol.get("weights", []):
        weight_id = weight.get("weight_id", "unknown")
        trusted_hash = weight.get("trusted_sha256")
        _require(
            isinstance(trusted_hash, str)
            and HEX_SHA256.fullmatch(trusted_hash) is not None,
            context,
            f"{weight_id} trusted full SHA256 is not pinned in the checked-in quality protocol",
        )
    receipt_path = Path(receipt_path or lpips_protocol["receipt_path"]).resolve()
    receipt = _read_json(receipt_path, context)
    _require(isinstance(receipt, dict), context, "root must be an object")
    _require(receipt.get("schema_version") == 2, context, "unsupported schema_version")
    _require(receipt.get("environment_root") == lpips_protocol["environment_root"], context, "environment_root differs from protocol")
    expected_executable = os.path.abspath(lpips_protocol["python_executable"])
    actual_executable = os.path.abspath(executable or sys.executable)
    _require(actual_executable == expected_executable, context, f"must run with {expected_executable}, found {actual_executable}")
    _require(receipt.get("python_executable") == expected_executable, context, "receipt python_executable differs from protocol")
    expected_prefix = os.path.abspath(lpips_protocol["environment_root"])
    actual_prefix = os.path.abspath(prefix or str(Path(actual_executable).parent.parent))
    _require(actual_prefix == expected_prefix, context, f"python executable escaped fixed eval environment: {actual_prefix!r}")
    _require(receipt.get("sys_prefix") == expected_prefix, context, "receipt sys_prefix differs from protocol")
    python_version = receipt.get("python_version")
    _require(
        isinstance(python_version, str)
        and re.fullmatch(r"3\.11\.\d+", python_version) is not None,
        context,
        "receipt Python version is not 3.11.x",
    )
    fixed_site_packages = (
        Path(expected_prefix) / "lib" / "python3.11" / "site-packages"
    ).resolve()
    _require(
        receipt.get("runtime_contract")
        == {
            "python_arguments": ["-I", "-S", "-B"],
            "python_minor": "3.11",
            "site_packages": str(fixed_site_packages),
        },
        context,
        "receipt runtime contract differs from fixed isolated scorer",
    )
    flags = runtime_flags or sys.flags
    for field in (
        "isolated",
        "no_site",
        "dont_write_bytecode",
        "no_user_site",
        "ignore_environment",
        "safe_path",
    ):
        _require(getattr(flags, field, 0) == 1, context, f"python runtime flag {field} must be enabled; launch with -I -S -B")

    receipt_packages = receipt.get("packages")
    _require(isinstance(receipt_packages, list) and bool(receipt_packages), context, "packages must be a non-empty list")
    computed_environment_lock = dependency_environment_lock_sha256(
        receipt_packages,
        context=context,
    )
    computed_lock_records = _dependency_lock_records(receipt_packages, context=context)
    _require(receipt.get("environment_lock_sha256") == computed_environment_lock, context, "receipt environment lock does not match its package set")
    _require(computed_environment_lock == trusted_environment_lock, context, "complete dependency environment lock differs from checked-in trust root")
    _require(computed_lock_records == trusted_environment_packages, context, "complete dependency package payload differs from checked-in trust root")
    by_distribution = {
        item.get("distribution"): item
        for item in receipt_packages
        if isinstance(item, dict) and isinstance(item.get("distribution"), str)
    }
    _require(len(by_distribution) == len(receipt_packages), context, "receipt contains invalid or duplicate package records")
    expected_distributions = [item["distribution"] for item in lpips_protocol["packages"]]
    _require(set(expected_distributions).issubset(by_distribution), context, "receipt omits a fixed direct dependency")
    installed_versions = dict(installed_distributions(fixed_site_packages))
    _require(set(installed_versions) == set(by_distribution), context, "installed distribution set differs from the complete receipt")
    normalized_packages = []
    environment_root = Path(expected_prefix).resolve()
    for distribution in sorted(by_distribution):
        record = by_distribution[distribution]
        _require(record.get("distribution") == distribution, context, f"{distribution} receipt name is not canonical")
        version = record.get("version")
        _require(isinstance(version, str) and bool(version), context, f"{distribution} receipt version is missing")
        _require(installed_versions[distribution] == version, context, f"{distribution} installed version differs from receipt")
        archive_hash = _full_sha(record.get("archive_sha256"), context, f"{distribution}.archive_sha256")
        archive_path_value = record.get("archive_path")
        _require(isinstance(archive_path_value, str) and archive_path_value, context, f"{distribution} retained wheel path is missing")
        archive_path = Path(archive_path_value).resolve()
        wheelhouse = Path(lpips_protocol["environment_root"]).parent.parent / "checkpoints" / "eval" / "wheels"
        try:
            archive_path.relative_to(wheelhouse.resolve())
        except ValueError:
            _fail(context, f"{distribution} retained wheel escaped fixed wheelhouse")
        _require(archive_path.is_file(), context, f"{distribution} retained wheel is missing")
        _require(sha256(archive_path) == archive_hash, context, f"{distribution} retained wheel SHA256 differs from receipt")
        record_path_value = record.get("record_path")
        _require(isinstance(record_path_value, str) and record_path_value, context, f"{distribution} wheel RECORD path is missing")
        wheel_record_path = Path(record_path_value).resolve()
        try:
            wheel_record_path.relative_to(environment_root)
        except ValueError:
            _fail(context, f"{distribution} wheel RECORD escaped fixed eval environment")
        _require(wheel_record_path.is_file(), context, f"{distribution} wheel RECORD is missing")
        wheel_record_hash = _full_sha(record.get("record_sha256"), context, f"{distribution}.record_sha256")
        _require(sha256(wheel_record_path) == wheel_record_hash, context, f"{distribution} wheel RECORD SHA256 drifted")
        record_errors = installed_record_validator(
            wheel_record_path,
            environment_root,
        )
        _require(not record_errors, context, f"{distribution} installed files differ from wheel RECORD: {'; '.join(record_errors[:5])}")
        site_packages = wheel_record_path.parent.parent
        _require(site_packages == fixed_site_packages, context, f"{distribution} RECORD escaped fixed site-packages")
        trusted_wheel_errors = wheel_record_validator(
            archive_path,
            site_packages,
            environment_root,
        )
        _require(not trusted_wheel_errors, context, f"{distribution} installed files differ from trusted wheel: {'; '.join(trusted_wheel_errors[:5])}")
        normalized_packages.append(dict(record))

    tree_errors = site_packages_validator(normalized_packages, environment_root)
    _require(not tree_errors, context, f"fixed site-packages tree contains untrusted files: {'; '.join(tree_errors[:5])}")
    site_packages_activator(fixed_site_packages)
    module_importer = import_module or importlib.import_module
    version_reader = distribution_version or (lambda name: installed_versions[name])
    for expected in lpips_protocol["packages"]:
        distribution = expected["distribution"]
        record = by_distribution[distribution]
        trusted_archive_hash = expected["trusted_archive_sha256"]
        for field in PACKAGE_CONTRACT_FIELDS:
            _require(record.get(field) == expected[field], context, f"{distribution} receipt {field} differs from protocol")
        _require(record.get("archive_sha256") == trusted_archive_hash, context, f"{distribution} archive SHA256 differs from checked-in direct trust root")
        try:
            imported_version = version_reader(distribution)
            module = module_importer(expected["module"])
        except Exception as exc:
            _fail(context, f"cannot import {distribution} exactly as receipted: {exc}")
        module_file = getattr(module, "__file__", None)
        _require(isinstance(module_file, str) and module_file, context, f"{distribution} has no module file")
        actual_module_path = str(Path(module_file).resolve())
        expected_module_path = str(Path(expected["module_path"]).resolve())
        _require(imported_version == expected["version"], context, f"{distribution} version {imported_version!r} != {expected['version']!r}")
        _require(actual_module_path == expected_module_path, context, f"{distribution} module path {actual_module_path!r} != {expected_module_path!r}")
        module_hash = _full_sha(record.get("module_sha256"), context, f"{distribution}.module_sha256")
        _require(sha256(Path(actual_module_path)) == module_hash, context, f"{distribution} imported module SHA256 drifted")

    receipt_weights = receipt.get("weights")
    _require(isinstance(receipt_weights, list), context, "weights must be a list")
    _require(len(receipt_weights) == len(lpips_protocol["weights"]), context, "receipt weight count differs from protocol")
    by_weight_id = {
        item.get("weight_id"): item
        for item in receipt_weights
        if isinstance(item, dict) and isinstance(item.get("weight_id"), str)
    }
    expected_weight_ids = [item["weight_id"] for item in lpips_protocol["weights"]]
    _require(set(by_weight_id) == set(expected_weight_ids), context, "receipt weight set differs from protocol")
    normalized_weights = []
    for expected in lpips_protocol["weights"]:
        weight_id = expected["weight_id"]
        record = by_weight_id[weight_id]
        trusted_weight_hash = expected.get("trusted_sha256")
        _require(
            isinstance(trusted_weight_hash, str)
            and HEX_SHA256.fullmatch(trusted_weight_hash) is not None,
            context,
            f"{weight_id} trusted full SHA256 is not pinned in the checked-in quality protocol",
        )
        for field in ("weight_id", "path", "source_type", "source"):
            _require(record.get(field) == expected[field], context, f"{weight_id} receipt {field} differs from protocol")
        for field in ("source_distribution", "source_version"):
            if field in expected:
                _require(record.get(field) == expected[field], context, f"{weight_id} receipt {field} differs from protocol")
        weight_path = Path(expected["path"]).resolve()
        _require(weight_path.is_file(), context, f"weight file is missing: {weight_path}")
        recorded_hash = _full_sha(record.get("sha256"), context, f"{weight_id}.sha256")
        recorded_bytes = _positive_int(record.get("bytes"), context, f"{weight_id}.bytes")
        try:
            actual_bytes = weight_path.stat().st_size
        except OSError as exc:
            _fail(context, f"cannot stat weight {weight_path}: {exc}")
        actual_hash = sha256(weight_path)
        _require(actual_bytes == recorded_bytes, context, f"{weight_id} byte count drifted")
        _require(actual_hash == recorded_hash, context, f"{weight_id} SHA256 drifted")
        _require(actual_hash == trusted_weight_hash, context, f"{weight_id} SHA256 differs from checked-in trust root")
        prefix = expected.get("sha256_prefix")
        if prefix is not None:
            _require(isinstance(prefix, str) and re.fullmatch(r"[0-9a-f]{8,64}", prefix) is not None, context, f"{weight_id} protocol hash prefix is invalid")
            _require(actual_hash.startswith(prefix), context, f"{weight_id} SHA256 does not match official prefix {prefix}")
        normalized_weights.append(dict(record))

    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "environment_root": receipt["environment_root"],
        "python_executable": receipt["python_executable"],
        "sys_prefix": receipt["sys_prefix"],
        "python_version": python_version,
        "runtime_contract": dict(receipt["runtime_contract"]),
        "environment_lock_sha256": computed_environment_lock,
        "packages": normalized_packages,
        "weights": normalized_weights,
    }


def collect_media_tool_receipt() -> dict[str, Any]:
    tools = []
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        _require(executable is not None, "media tool receipt", f"{name} is not on PATH")
        executable_path = Path(executable).resolve()
        try:
            process = subprocess.run(
                [str(executable_path), "-version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            _fail("media tool receipt", f"cannot query {name}: {exc}")
        first_line = process.stdout.splitlines()[0] if process.stdout.splitlines() else ""
        _require(bool(first_line), "media tool receipt", f"{name} did not report a version")
        tools.append(
            {
                "name": name,
                "path": str(executable_path),
                "sha256": sha256(executable_path),
                "version_line": first_line,
            }
        )
    return {"tools": tools}


def capture_evaluator_source_receipt(
    protocol_path: Path,
    matrix_path: Path,
) -> dict[str, Any]:
    context = "evaluator source receipt"
    protocol_path = Path(protocol_path).resolve()
    matrix_path = Path(matrix_path).resolve()
    _require(protocol_path == DEFAULT_PROTOCOL.resolve(), context, "only the checked-in fixed quality protocol is accepted")
    _require(matrix_path == DEFAULT_MATRIX.resolve(), context, "only the checked-in Ovi evaluation matrix is accepted")
    source_paths = {
        "comparison_script": Path(__file__).resolve(),
        "compare_media_script": (REPO_ROOT / "scripts" / "compare_media.py").resolve(),
        "run_validator_script": RUN_VALIDATOR_PATH.resolve(),
        "archive_url_policy": ARCHIVE_URL_POLICY_PATH.resolve(),
        "quality_protocol": protocol_path,
        "evaluation_matrix": matrix_path,
    }
    relative_paths = [str(path.relative_to(REPO_ROOT)) for path in source_paths.values()]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", *relative_paths],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", *relative_paths],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(context, f"cannot establish clean tracked evaluator source: {exc}")
    _require(bool(tracked.stdout.strip()), context, "evaluator source files are not tracked")
    _require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, context, "evaluator git commit is invalid")
    _require(not dirty.strip(), context, "evaluator source files have uncommitted changes")
    return {
        "git_commit": commit,
        "files": {
            role: {"path": str(path), "sha256": sha256(path)}
            for role, path in source_paths.items()
        },
    }


def validate_evaluator_source_receipt(receipt: Mapping[str, Any]) -> None:
    context = "evaluator source receipt"
    git_commit = receipt.get("git_commit")
    _require(isinstance(git_commit, str) and re.fullmatch(r"[0-9a-f]{40}", git_commit) is not None, context, "git commit is invalid")
    files = receipt.get("files")
    _require(isinstance(files, Mapping), context, "files are missing")
    _require(
        set(files)
        == {
            "comparison_script",
            "compare_media_script",
            "run_validator_script",
            "archive_url_policy",
            "quality_protocol",
            "evaluation_matrix",
        },
        context,
        "source file set differs from fixed evaluator contract",
    )
    expected_paths = {
        "comparison_script": Path(__file__).resolve(),
        "compare_media_script": (REPO_ROOT / "scripts" / "compare_media.py").resolve(),
        "run_validator_script": RUN_VALIDATOR_PATH.resolve(),
        "archive_url_policy": ARCHIVE_URL_POLICY_PATH.resolve(),
        "quality_protocol": DEFAULT_PROTOCOL.resolve(),
        "evaluation_matrix": DEFAULT_MATRIX.resolve(),
    }
    for role, expected_path in expected_paths.items():
        record = files[role]
        _require(isinstance(record, Mapping), context, f"{role} receipt is not an object")
        _require(record.get("path") == str(expected_path), context, f"{role} path differs from fixed source")
        expected_hash = _full_sha(record.get("sha256"), context, f"{role}.sha256")
        _require(expected_path.is_file(), context, f"{role} source file is missing")
        _require(sha256(expected_path) == expected_hash, context, f"{role} source SHA256 drifted")


def validate_media_tool_receipt(receipt: Mapping[str, Any]) -> dict[str, Path]:
    context = "media tool receipt"
    tools = receipt.get("tools")
    _require(isinstance(tools, list) and len(tools) == 2, context, "receipt must contain ffmpeg and ffprobe")
    by_name = {
        item.get("name"): item
        for item in tools
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    _require(set(by_name) == {"ffmpeg", "ffprobe"}, context, "tool set differs from fixed contract")
    paths: dict[str, Path] = {}
    for name in ("ffmpeg", "ffprobe"):
        record = by_name[name]
        path_value = record.get("path")
        _require(isinstance(path_value, str) and path_value, context, f"{name} path is missing")
        path = Path(path_value).resolve()
        expected_hash = _full_sha(record.get("sha256"), context, f"{name}.sha256")
        _require(path.is_file(), context, f"{name} binary is missing: {path}")
        _require(sha256(path) == expected_hash, context, f"{name} binary SHA256 drifted")
        try:
            process = subprocess.run(
                [str(path), "-version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            _fail(context, f"cannot re-query {name}: {exc}")
        version_line = process.stdout.splitlines()[0] if process.stdout.splitlines() else ""
        _require(version_line == record.get("version_line"), context, f"{name} version receipt drifted")
        paths[name] = path
    return paths


def compute_reused_media_metrics(
    reference: Path,
    candidate: Path,
    tool_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Reuse compare_media's decoded PSNR, SSIM, and mono-16k audio logic."""

    media = _media_module()
    try:
        np = importlib.import_module("numpy")
    except Exception as exc:
        _fail("media dependencies", f"cannot import numpy: {exc}")
    reference_video = media.probe_video(reference, ffprobe=tool_paths["ffprobe"])
    candidate_video = media.probe_video(candidate, ffprobe=tool_paths["ffprobe"])
    for field in ("frames", "width", "height", "avg_frame_rate"):
        _require(reference_video[field] == candidate_video[field], "decoded media", f"candidate video {field} differs from dense")
    frame_count = reference_video["frames"]
    _positive_int(frame_count, "decoded media", "frame count")
    video_psnr = media.ffmpeg_metric(reference, candidate, frame_count, "psnr", r"average:([0-9.+-]+|inf)", ffmpeg=tool_paths["ffmpeg"])
    video_ssim = media.ffmpeg_metric(reference, candidate, frame_count, "ssim", r"All:([0-9.+-]+|inf)", ffmpeg=tool_paths["ffmpeg"])

    reference_audio = media.decode_audio(reference, ffmpeg=tool_paths["ffmpeg"])
    candidate_audio = media.decode_audio(candidate, ffmpeg=tool_paths["ffmpeg"])
    _require(reference_audio.size > 0 and candidate_audio.size > 0, "decoded media", "one or both artifacts have no audio samples")
    _require(reference_audio.size == candidate_audio.size, "decoded media", "candidate decoded audio sample count differs from dense")
    difference = reference_audio - candidate_audio
    audio_rmse = float(np.sqrt(np.mean(np.square(difference))))
    audio_max_abs = float(np.max(np.abs(difference)))
    reference_rms = float(np.sqrt(np.mean(np.square(reference_audio))))
    audio_snr_db = float(20.0 * math.log10(max(reference_rms, 1e-12) / max(audio_rmse, 1e-12)))
    if np.std(reference_audio) == 0 or np.std(candidate_audio) == 0:
        audio_correlation = 1.0 if np.array_equal(reference_audio, candidate_audio) else 0.0
    else:
        audio_correlation = float(np.corrcoef(reference_audio, candidate_audio)[0, 1])
    return {
        "compared_video_frames": frame_count,
        "video_psnr_db": video_psnr,
        "video_ssim": video_ssim,
        "reference_audio_samples": int(reference_audio.size),
        "candidate_audio_samples": int(candidate_audio.size),
        "audio_sample_count_compared": int(reference_audio.size),
        "audio_rmse": audio_rmse,
        "audio_max_abs_difference": audio_max_abs,
        "audio_snr_db": audio_snr_db,
        "audio_correlation": audio_correlation,
    }


class LpipsAlexCpu:
    """One fixed, reused LPIPS AlexNet model running only on CPU."""

    def __init__(
        self,
        lpips_protocol: Mapping[str, Any],
        tool_paths: Mapping[str, Path],
    ):
        self.protocol = lpips_protocol
        self.ffmpeg = str(tool_paths["ffmpeg"])
        self.ffprobe = str(tool_paths["ffprobe"])
        os.environ["TORCH_HOME"] = lpips_protocol["torch_home"]
        try:
            self.torch = importlib.import_module("torch")
            lpips_module = importlib.import_module("lpips")
            self.torch.set_num_threads(lpips_protocol["torch_num_threads"])
            self.torch.set_num_interop_threads(
                lpips_protocol["torch_num_interop_threads"]
            )
            self.torch.use_deterministic_algorithms(
                lpips_protocol["torch_deterministic_algorithms"]
            )
            self.torch.backends.mkldnn.enabled = lpips_protocol[
                "torch_mkldnn_enabled"
            ]
            self.model = lpips_module.LPIPS(
                net="alex",
                version="0.1",
                lpips=True,
                spatial=False,
                pnet_rand=False,
                pretrained=True,
                eval_mode=True,
                verbose=False,
            ).to("cpu").eval()
        except Exception as exc:
            _fail("LPIPS model", f"cannot construct fixed AlexNet model from receipted dependencies: {exc}")
        parameters = list(self.model.parameters())
        _require(parameters, "LPIPS model", "model has no parameters")
        _require(all(parameter.device.type == "cpu" for parameter in parameters), "LPIPS model", "model is not entirely on CPU")

    def _start_decoder(self, path: Path, frame_count: int) -> subprocess.Popen:
        try:
            return subprocess.Popen(
                [
                    self.ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-frames:v",
                    str(frame_count),
                    "-fps_mode",
                    "passthrough",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            _fail("LPIPS decode", f"cannot start ffmpeg for {path}: {exc}")

    @staticmethod
    def _read_exact(stream: Any, size: int, context: str) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = stream.read(size - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        _require(len(chunks) == size, "LPIPS decode", f"{context} ended after {len(chunks)} of {size} bytes")
        return bytes(chunks)

    @staticmethod
    def _stop_decoder(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

    def __call__(self, reference: Path, candidate: Path) -> tuple[float, int]:
        media = _media_module()
        try:
            np = importlib.import_module("numpy")
        except Exception as exc:
            _fail("LPIPS metric", f"cannot import numpy: {exc}")
        reference_video = media.probe_video(reference, ffprobe=self.ffprobe)
        candidate_video = media.probe_video(candidate, ffprobe=self.ffprobe)
        for field in ("frames", "width", "height"):
            _require(reference_video[field] == candidate_video[field], "LPIPS decode", f"candidate {field} differs from dense")
        frame_count = _positive_int(reference_video["frames"], "LPIPS decode", "frame count")
        width = _positive_int(reference_video["width"], "LPIPS decode", "width")
        height = _positive_int(reference_video["height"], "LPIPS decode", "height")
        frame_bytes = width * height * 3
        processes: list[subprocess.Popen] = []
        values: list[float] = []
        completed = False
        try:
            left = self._start_decoder(reference, frame_count)
            processes.append(left)
            right = self._start_decoder(candidate, frame_count)
            processes.append(right)
            _require(left.stdout is not None and left.stderr is not None, "LPIPS decode", "dense decoder pipes are unavailable")
            _require(right.stdout is not None and right.stderr is not None, "LPIPS decode", "candidate decoder pipes are unavailable")
            with self.torch.no_grad():
                for index in range(frame_count):
                    left_raw = self._read_exact(left.stdout, frame_bytes, f"dense frame {index}")
                    right_raw = self._read_exact(right.stdout, frame_bytes, f"candidate frame {index}")
                    left_array = np.frombuffer(left_raw, dtype=np.uint8).copy().reshape(height, width, 3)
                    right_array = np.frombuffer(right_raw, dtype=np.uint8).copy().reshape(height, width, 3)
                    left_tensor = self.torch.from_numpy(left_array).permute(2, 0, 1).unsqueeze(0).to(dtype=self.torch.float32)
                    right_tensor = self.torch.from_numpy(right_array).permute(2, 0, 1).unsqueeze(0).to(dtype=self.torch.float32)
                    left_tensor = left_tensor.div(127.5).sub(1.0)
                    right_tensor = right_tensor.div(127.5).sub(1.0)
                    value = float(self.model(left_tensor, right_tensor, normalize=False).reshape(-1).mean().item())
                    _require(math.isfinite(value) and value >= 0.0, "LPIPS metric", f"frame {index} produced invalid value {value!r}")
                    values.append(value)
            try:
                left_remainder, left_stderr = left.communicate(timeout=30)
                right_remainder, right_stderr = right.communicate(timeout=30)
            except subprocess.TimeoutExpired as exc:
                _fail("LPIPS decode", f"ffmpeg did not exit after fixed frame decode: {exc}")
            _require(left.returncode == 0, "LPIPS decode", f"dense ffmpeg failed: {left_stderr.decode('utf-8', errors='replace').strip()}")
            _require(right.returncode == 0, "LPIPS decode", f"candidate ffmpeg failed: {right_stderr.decode('utf-8', errors='replace').strip()}")
            _require(left_remainder == b"", "LPIPS decode", "dense decoder emitted more frames than fixed count")
            _require(right_remainder == b"", "LPIPS decode", "candidate decoder emitted more frames than fixed count")
            completed = True
        finally:
            if not completed:
                for process in processes:
                    self._stop_decoder(process)
        _require(len(values) == frame_count, "LPIPS metric", "not every decoded frame produced a score")
        return float(statistics.fmean(values)), frame_count


def make_metric_runner(
    lpips_runner: LpipsAlexCpu,
    tool_paths: Mapping[str, Path],
) -> Callable[[MeasurementArtifact, MeasurementArtifact], dict[str, Any]]:
    def run(dense_artifact: MeasurementArtifact, candidate_artifact: MeasurementArtifact) -> dict[str, Any]:
        result = compute_reused_media_metrics(
            dense_artifact.path,
            candidate_artifact.path,
            tool_paths,
        )
        lpips_value, lpips_frames = lpips_runner(dense_artifact.path, candidate_artifact.path)
        result["lpips_alex"] = lpips_value
        result["lpips_frame_count"] = lpips_frames
        return result

    return run


def _finite_metric(value: Any, context: str, field: str, *, nonnegative: bool = False) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), context, f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result), context, f"{field} must be finite, found {result!r}")
    if nonnegative:
        _require(result >= 0.0, context, f"{field} must be nonnegative")
    return result


def _normalize_metrics(payload: Mapping[str, Any], context: str) -> tuple[dict[str, Any], dict[str, float]]:
    _require(isinstance(payload, Mapping), context, "metric runner did not return an object")
    counts = {}
    for field in (
        "compared_video_frames",
        "reference_audio_samples",
        "candidate_audio_samples",
        "audio_sample_count_compared",
        "lpips_frame_count",
    ):
        counts[field] = _positive_int(payload.get(field), context, field)
    _require(counts["compared_video_frames"] == counts["lpips_frame_count"], context, "LPIPS frame count differs from PSNR/SSIM frame count")
    _require(counts["reference_audio_samples"] == counts["candidate_audio_samples"] == counts["audio_sample_count_compared"], context, "audio sample counts are not exact")

    psnr_value = payload.get("video_psnr_db")
    _require(isinstance(psnr_value, (int, float)) and not isinstance(psnr_value, bool), context, "video_psnr_db must be numeric")
    psnr = float(psnr_value)
    _require(not math.isnan(psnr) and psnr != -math.inf, context, f"video_psnr_db is invalid: {psnr!r}")
    numeric = {
        "video_psnr_db": psnr,
        "video_ssim": _finite_metric(payload.get("video_ssim"), context, "video_ssim"),
        "lpips_alex": _finite_metric(payload.get("lpips_alex"), context, "lpips_alex", nonnegative=True),
        "audio_rmse": _finite_metric(payload.get("audio_rmse"), context, "audio_rmse", nonnegative=True),
        "audio_max_abs_difference": _finite_metric(payload.get("audio_max_abs_difference"), context, "audio_max_abs_difference", nonnegative=True),
        "audio_snr_db": _finite_metric(payload.get("audio_snr_db"), context, "audio_snr_db"),
        "audio_correlation": _finite_metric(payload.get("audio_correlation"), context, "audio_correlation"),
    }
    _require(-1.0 <= numeric["video_ssim"] <= 1.0, context, "video_ssim is outside [-1,1]")
    _require(-1.0 <= numeric["audio_correlation"] <= 1.0, context, "audio_correlation is outside [-1,1]")
    rendered = {**counts, **numeric}
    if math.isinf(psnr):
        rendered["video_psnr_db"] = "inf"
    return rendered, numeric


def _normalize_persisted_metrics(
    payload: Mapping[str, Any],
    context: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Revalidate an immutable sidecar, including the explicit PSNR sentinel."""

    _require(isinstance(payload, Mapping), context, "persisted metrics are missing")
    expected_fields = {
        "compared_video_frames",
        "reference_audio_samples",
        "candidate_audio_samples",
        "audio_sample_count_compared",
        "lpips_frame_count",
        *MEDIAN_METRICS,
    }
    _require(set(payload) == expected_fields, context, "persisted metric field set differs from protocol")
    working = dict(payload)
    if working.get("video_psnr_db") == "inf":
        working["video_psnr_db"] = math.inf
    rendered, numeric = _normalize_metrics(working, context)
    _require(rendered == dict(payload), context, "persisted metrics are not in canonical form")
    return rendered, numeric


def _assert_artifact_hash(artifact: MeasurementArtifact, phase: str) -> None:
    context = f"artifact {phase}"
    validator = _run_validator_module()
    try:
        snapshot = validator._stable_file_snapshot(artifact.path, context)
        sidecar_snapshot = validator._stable_file_snapshot(
            artifact.metrics_sidecar_path,
            context,
        )
    except Exception as exc:
        _fail(context, f"cannot take stable no-follow artifact snapshot: {exc}")
    _require(snapshot.sha256 == artifact.sha256, context, f"SHA256 drift for identity {artifact.identity}: {snapshot.sha256} != {artifact.sha256}")
    _require(sidecar_snapshot.sha256 == artifact.metrics_sidecar_sha256, context, f"metrics sidecar SHA256 drift for identity {artifact.identity}")
    try:
        validator._revalidate_snapshot(snapshot, context)
        validator._revalidate_snapshot(sidecar_snapshot, context)
    except Exception as exc:
        _fail(context, f"artifact changed after stable snapshot: {exc}")


_CORE_RUN_EVIDENCE = frozenset(
    {
        "environment.json",
        "verification.json",
        "timings.jsonl",
        "warmup_timings.jsonl",
        "checkpoint_manifest.json",
    }
)


def _validate_bound_environment_originals(
    snapshots: Mapping[str, Any],
    context: str,
) -> None:
    """Cross-bind every original named by the stable environment snapshot."""

    environment_snapshot = snapshots["environment.json"]
    try:
        environment = json.loads(environment_snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(context, f"cannot decode bound environment.json: {exc}")
    _require(
        isinstance(environment, dict),
        context,
        "bound environment.json must contain an object",
    )
    evidence_hashes = environment.get("evidence_file_sha256")
    _require(
        isinstance(evidence_hashes, Mapping) and evidence_hashes,
        context,
        "bound environment evidence_file_sha256 is missing",
    )
    for filename, expected_hash_value in sorted(
        evidence_hashes.items(), key=lambda item: str(item[0])
    ):
        _require(
            isinstance(filename, str)
            and filename
            and Path(filename).name == filename
            and filename not in {".", ".."},
            context,
            f"invalid bound environment evidence filename {filename!r}",
        )
        _require(
            filename in snapshots,
            context,
            f"evidence_bindings omits environment evidence {filename}",
        )
        expected_hash = _full_sha(
            expected_hash_value,
            context,
            f"{filename} environment evidence SHA256",
        )
        _require(
            snapshots[filename].sha256 == expected_hash,
            context,
            f"{filename} differs from bound environment evidence hash",
        )

    _require(
        "run_config.yaml" in snapshots,
        context,
        "evidence_bindings omits run_config.yaml",
    )
    run_config_hash = _full_sha(
        environment.get("run_config_sha256"),
        context,
        "run_config.yaml environment SHA256",
    )
    _require(
        snapshots["run_config.yaml"].sha256 == run_config_hash,
        context,
        "run_config.yaml differs from bound environment hash",
    )


def _snapshot_bound_run_evidence(
    run_dir: Path,
    bindings: Any,
    context: str,
) -> dict[str, Any]:
    """Revalidate every persisted run-original binding without following links."""

    _require(isinstance(bindings, Mapping), context, "evidence_bindings is missing")
    _require(
        _CORE_RUN_EVIDENCE.issubset(bindings),
        context,
        "evidence_bindings omits a core run original",
    )
    validator = _run_validator_module()
    snapshots = {}
    for name, binding in sorted(bindings.items(), key=lambda item: str(item[0])):
        _require(
            isinstance(name, str)
            and name
            and Path(name).name == name
            and name not in {".", ".."},
            context,
            f"invalid evidence binding filename {name!r}",
        )
        _require(
            isinstance(binding, Mapping)
            and set(binding) == {"path", "bytes", "sha256"},
            context,
            f"{name} evidence binding fields are invalid",
        )
        expected_path = run_dir / name
        _require(
            binding.get("path") == str(expected_path),
            context,
            f"{name} evidence binding path is not canonical",
        )
        expected_bytes = binding.get("bytes")
        _require(
            isinstance(expected_bytes, int)
            and not isinstance(expected_bytes, bool)
            and expected_bytes >= 0,
            context,
            f"{name} evidence byte count is invalid",
        )
        expected_hash = _full_sha(
            binding.get("sha256"), context, f"{name} evidence SHA256"
        )
        try:
            snapshot = validator._stable_file_snapshot(expected_path, context)
        except Exception as exc:
            _fail(context, f"cannot take stable no-follow evidence snapshot: {exc}")
        _require(
            snapshot.size == expected_bytes and snapshot.sha256 == expected_hash,
            context,
            f"{name} evidence bytes or SHA256 drifted",
        )
        snapshots[name] = snapshot
    _validate_bound_environment_originals(snapshots, context)
    for snapshot in snapshots.values():
        try:
            validator._revalidate_snapshot(snapshot, context)
        except Exception as exc:
            _fail(context, f"evidence changed after stable snapshot: {exc}")
    return snapshots


def _assert_run_evidence(run: AuditedRun, phase: str) -> None:
    context = f"{run.method_id} run evidence {phase}"
    snapshots = _snapshot_bound_run_evidence(
        run.run_dir,
        run.evidence_bindings,
        context,
    )
    expected_hashes = {
        "environment.json": run.environment_sha256,
        "verification.json": run.verification_sha256,
        "timings.jsonl": run.timings_sha256,
        "warmup_timings.jsonl": run.warmup_timings_sha256,
        "checkpoint_manifest.json": run.checkpoint_manifest_sha256,
    }
    for name, expected_hash in expected_hashes.items():
        _require(
            snapshots[name].sha256 == expected_hash,
            context,
            f"{name} differs from the audited run summary",
        )
    validator = _run_validator_module()
    timings_snapshot, timings_records = validator._snapshot_jsonl(
        run.timings_path,
        context,
    )
    warmup_snapshot, warmup_records = validator._snapshot_jsonl(
        run.warmup_timings_path,
        context,
    )
    _require(
        timings_snapshot.path == run.timings_path
        and timings_snapshot.size == run.timings_bytes
        and timings_snapshot.sha256 == run.timings_sha256
        and len(timings_records) == run.timings_record_count,
        context,
        "timings path, bytes, hash, or record count drifted",
    )
    _require(
        warmup_snapshot.path == run.warmup_timings_path
        and warmup_snapshot.size == run.warmup_timings_bytes
        and warmup_snapshot.sha256 == run.warmup_timings_sha256
        and len(warmup_records) == run.warmup_record_count,
        context,
        "warmup timings path, bytes, hash, or record count drifted",
    )
    for snapshot in (timings_snapshot, warmup_snapshot):
        try:
            validator._revalidate_snapshot(snapshot, context)
        except Exception as exc:
            _fail(context, f"evidence changed after JSONL snapshot: {exc}")
    for artifact in run.artifacts.values():
        _assert_artifact_hash(artifact, phase)


def build_quality_report(
    dense: AuditedRun,
    candidate: AuditedRun,
    protocol: Mapping[str, Any],
    *,
    protocol_sha256: str,
    lpips_receipt: Mapping[str, Any],
    media_tool_receipt: Mapping[str, Any],
    evaluator_source_receipt: Mapping[str, Any],
    metric_runner: Callable[[MeasurementArtifact, MeasurementArtifact], Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the full identity-paired matrix; write nothing on failure."""

    _full_sha(protocol_sha256, "quality report", "protocol_sha256")
    _full_sha(lpips_receipt.get("receipt_sha256"), "quality report", "lpips receipt SHA256")
    validate_evaluator_source_receipt(evaluator_source_receipt)
    validate_selected_sparse_profile_consistency((dense, candidate))
    _validate_pairing(dense, candidate, protocol)
    _require(
        dense.git_commit == evaluator_source_receipt.get("git_commit"),
        "quality report",
        "selected runs were not generated by the hash-bound evaluator commit",
    )
    for run in (dense, candidate):
        _assert_run_evidence(run, "before all metrics")
    dense_artifacts = _artifact_identity_map(dense, "quality report dense")
    candidate_artifacts = _artifact_identity_map(candidate, "quality report candidate")
    pair_reports = []
    numeric_by_metric: dict[str, list[float]] = {field: [] for field in MEDIAN_METRICS}
    for identity in sorted(dense_artifacts):
        dense_artifact = dense_artifacts[identity]
        candidate_artifact = candidate_artifacts[identity]
        _assert_artifact_hash(dense_artifact, "before metrics")
        _assert_artifact_hash(candidate_artifact, "before metrics")
        metrics_payload = metric_runner(dense_artifact, candidate_artifact)
        _assert_artifact_hash(dense_artifact, "after metrics")
        _assert_artifact_hash(candidate_artifact, "after metrics")
        rendered_metrics, numeric_metrics = _normalize_metrics(metrics_payload, f"quality pair {identity}")
        for field in MEDIAN_METRICS:
            numeric_by_metric[field].append(numeric_metrics[field])
        pair_reports.append(
            {
                "schema_version": QUALITY_SCHEMA_VERSION,
                "record_type": "ovi_quality_pair",
                "quality_protocol_id": protocol["protocol_id"],
                "quality_protocol_sha256": protocol_sha256,
                "measurement_index": identity[0],
                "prompt_index": identity[1],
                "sample_index": identity[2],
                "dense": _artifact_binding(dense, dense_artifact),
                "candidate": _artifact_binding(candidate, candidate_artifact),
                "metrics": rendered_metrics,
                "automatic_acceptance": None,
            }
        )
    medians: dict[str, Any] = {}
    for field, values in numeric_by_metric.items():
        value = float(statistics.median(values))
        _require(not math.isnan(value) and value != -math.inf, "quality median", f"{field} median is invalid")
        medians[field] = "inf" if math.isinf(value) else value
    for run in (dense, candidate):
        _assert_run_evidence(run, "after all metrics")
    validate_evaluator_source_receipt(evaluator_source_receipt)
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "record_type": "ovi_quality_report",
        "quality_protocol_id": protocol["protocol_id"],
        "quality_protocol_sha256": protocol_sha256,
        "comparison_script_sha256": evaluator_source_receipt["files"]["comparison_script"]["sha256"],
        "compare_media_script_sha256": evaluator_source_receipt["files"]["compare_media_script"]["sha256"],
        "run_validator_script_sha256": evaluator_source_receipt["files"]["run_validator_script"]["sha256"],
        "evaluation_matrix_sha256": evaluator_source_receipt["files"]["evaluation_matrix"]["sha256"],
        "evaluator_source_receipt": dict(evaluator_source_receipt),
        "lpips_dependency_receipt": dict(lpips_receipt),
        "media_tool_receipt": dict(media_tool_receipt),
        "dense_run": dense.sidecar_binding(),
        "candidate_run": candidate.sidecar_binding(),
        "pairs": pair_reports,
        "pair_count": len(pair_reports),
        "metric_medians": medians,
        "automatic_acceptance": None,
        "manual_review": {
            "status": "not_provided",
            "row_count": 0,
            "csv_path": None,
            "csv_sha256": None,
        },
    }


def _validate_persisted_run_binding(binding: Mapping[str, Any], side: str) -> None:
    context = f"manual review {side} run binding"
    run_dir_value = binding.get("run_dir")
    _require(isinstance(run_dir_value, str) and run_dir_value, context, "run_dir is missing")
    run_dir = Path(run_dir_value).resolve()
    _require(run_dir.is_dir(), context, f"run directory is missing: {run_dir}")
    snapshots = _snapshot_bound_run_evidence(
        run_dir,
        binding.get("evidence_bindings"),
        context,
    )
    expected_hashes = {
        "environment.json": binding.get("environment_sha256"),
        "verification.json": binding.get("verification_sha256"),
        "timings.jsonl": binding.get("timings_sha256"),
        "warmup_timings.jsonl": binding.get("warmup_timings_sha256"),
        "checkpoint_manifest.json": binding.get("checkpoint_manifest_sha256"),
    }
    for name, expected_hash_value in expected_hashes.items():
        expected_hash = _full_sha(expected_hash_value, context, f"{name}.sha256")
        _require(
            snapshots[name].sha256 == expected_hash,
            context,
            f"{name} differs from the persisted run summary",
        )
    validator = _run_validator_module()
    timings_path = run_dir / "timings.jsonl"
    warmup_path = run_dir / "warmup_timings.jsonl"
    timings_snapshot, timings_records = validator._snapshot_jsonl(
        timings_path, context
    )
    warmup_snapshot, warmup_records = validator._snapshot_jsonl(
        warmup_path, context
    )
    _require(
        binding.get("timings_path") == str(timings_path)
        and binding.get("timings_bytes") == timings_snapshot.size
        and isinstance(binding.get("timings_record_count"), int)
        and not isinstance(binding.get("timings_record_count"), bool)
        and binding.get("timings_record_count") == len(timings_records),
        context,
        "timings snapshot metadata is invalid",
    )
    _require(
        binding.get("warmup_timings_path") == str(warmup_path)
        and binding.get("warmup_timings_bytes") == warmup_snapshot.size
        and isinstance(binding.get("warmup_record_count"), int)
        and not isinstance(binding.get("warmup_record_count"), bool)
        and binding.get("warmup_record_count") == len(warmup_records),
        context,
        "warmup timings snapshot metadata is invalid",
    )
    for snapshot in (timings_snapshot, warmup_snapshot):
        try:
            validator._revalidate_snapshot(snapshot, context)
        except Exception as exc:
            _fail(context, f"evidence changed after JSONL snapshot: {exc}")


def load_validated_quality_median(
    median_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[tuple[int, int, int], tuple[str, str]],
    str,
]:
    median_path = Path(median_path).resolve()
    report = _read_json(median_path, "quality median")
    _require(isinstance(report, dict), "quality median", "root must be an object")
    _require(report.get("schema_version") == QUALITY_SCHEMA_VERSION, "quality median", "unsupported schema")
    _require(report.get("record_type") == "ovi_quality_median", "quality median", "record_type must be ovi_quality_median")
    _require(
        set(report)
        == {
            "schema_version",
            "record_type",
            "quality_protocol_id",
            "quality_protocol_sha256",
            "comparison_script_sha256",
            "compare_media_script_sha256",
            "run_validator_script_sha256",
            "evaluation_matrix_sha256",
            "evaluator_source_receipt",
            "lpips_dependency_receipt",
            "media_tool_receipt",
            "dense_run",
            "candidate_run",
            "pairs",
            "pair_count",
            "metric_medians",
            "automatic_acceptance",
            "manual_review",
        },
        "quality median",
        "median field set differs from the fixed schema",
    )
    _require(report.get("automatic_acceptance") is None, "quality median", "automatic acceptance must remain null")
    _require(
        isinstance(report.get("pair_count"), int)
        and not isinstance(report.get("pair_count"), bool)
        and report.get("pair_count") > 0
        and isinstance(report.get("pairs"), list)
        and len(report["pairs"]) == report["pair_count"],
        "quality median",
        "pair_count does not match persisted pair bindings",
    )
    _require(
        report.get("manual_review")
        == {
            "status": "not_provided",
            "row_count": 0,
            "csv_path": None,
            "csv_sha256": None,
        },
        "quality median",
        "manual review status must remain separate and not provided",
    )
    _require(report.get("quality_protocol_id") == protocol["protocol_id"], "quality median", "protocol id differs from checked-in protocol")
    _require(report.get("quality_protocol_sha256") == protocol_sha256, "quality median", "protocol hash differs from checked-in protocol")
    evaluator_source = report.get("evaluator_source_receipt")
    _require(isinstance(evaluator_source, Mapping), "quality median", "evaluator source receipt is missing")
    validate_evaluator_source_receipt(evaluator_source)
    _require(report.get("comparison_script_sha256") == evaluator_source["files"]["comparison_script"]["sha256"], "quality median", "comparison script hash is not source-bound")
    _require(report.get("compare_media_script_sha256") == evaluator_source["files"]["compare_media_script"]["sha256"], "quality median", "media script hash is not source-bound")
    _require(report.get("run_validator_script_sha256") == evaluator_source["files"]["run_validator_script"]["sha256"], "quality median", "run validator script hash is not source-bound")
    _require(report.get("evaluation_matrix_sha256") == evaluator_source["files"]["evaluation_matrix"]["sha256"], "quality median", "matrix hash is not source-bound")
    media_receipt = report.get("media_tool_receipt")
    _require(isinstance(media_receipt, Mapping), "quality median", "media tool receipt is missing")
    validate_media_tool_receipt(media_receipt)
    lpips_receipt = report.get("lpips_dependency_receipt")
    _require(isinstance(lpips_receipt, Mapping), "quality median", "LPIPS receipt is missing")
    validated_lpips = validate_lpips_receipt(
        protocol["lpips"],
        receipt_path=Path(lpips_receipt.get("receipt_path", "")),
    )
    _require(validated_lpips == lpips_receipt, "quality median", "LPIPS dependency receipt differs from current validated receipt")
    dense_run = report.get("dense_run")
    candidate_run = report.get("candidate_run")
    _require(isinstance(dense_run, Mapping), "quality median", "dense_run binding is missing")
    _require(isinstance(candidate_run, Mapping), "quality median", "candidate_run binding is missing")
    _validate_persisted_run_binding(dense_run, "dense")
    _validate_persisted_run_binding(candidate_run, "candidate")
    for pair in report.get("pairs", []):
        if isinstance(pair, Mapping):
            pair_path_value = pair.get("pair_sidecar_path")
            _require(isinstance(pair_path_value, str) and Path(pair_path_value).resolve().parent == median_path.parent, "quality median", "pair sidecar is not beside median")
    bindings = _expected_manual_bindings_from_report(report)
    return report, bindings, sha256(median_path)


def _expected_manual_bindings_from_report(report: Mapping[str, Any]) -> dict[tuple[int, int, int], tuple[str, str]]:
    _require(report.get("schema_version") == QUALITY_SCHEMA_VERSION, "manual review bindings", "unsupported quality median schema")
    _require(report.get("record_type") == "ovi_quality_median", "manual review bindings", "only a persisted ovi_quality_median is accepted")
    pairs = report.get("pairs")
    _require(
        isinstance(pairs, list)
        and isinstance(report.get("pair_count"), int)
        and not isinstance(report.get("pair_count"), bool)
        and len(pairs) == report.get("pair_count"),
        "manual review bindings",
        "quality report pair count is invalid",
    )
    expected: dict[tuple[int, int, int], tuple[str, str]] = {}
    numeric_by_metric: dict[str, list[float]] = {
        field: [] for field in MEDIAN_METRICS
    }
    _require(isinstance(report.get("dense_run"), Mapping), "manual review bindings", "dense_run binding is missing")
    _require(isinstance(report.get("candidate_run"), Mapping), "manual review bindings", "candidate_run binding is missing")
    for pair in pairs:
        _require(isinstance(pair, Mapping), "manual review bindings", "pair must be an object")
        identity = tuple(pair.get(field) for field in IDENTITY_FIELDS)
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in identity)
            and identity[0] in EXPECTED_MEASUREMENT_INDICES
            and identity not in expected,
            "manual review bindings",
            f"invalid or duplicate identity {identity!r}",
        )
        dense_hash = _full_sha(pair.get("dense_artifact_sha256"), "manual review bindings", "dense artifact SHA256")
        candidate_hash = _full_sha(pair.get("candidate_artifact_sha256"), "manual review bindings", "candidate artifact SHA256")
        pair_path_value = pair.get("pair_sidecar_path")
        _require(isinstance(pair_path_value, str) and pair_path_value, "manual review bindings", "pair sidecar path is missing")
        pair_path = Path(pair_path_value).resolve()
        expected_sidecar_hash = _full_sha(pair.get("pair_sidecar_sha256"), "manual review bindings", "pair sidecar SHA256")
        _require(pair_path.is_file(), "manual review bindings", f"pair sidecar is missing: {pair_path}")
        _require(sha256(pair_path) == expected_sidecar_hash, "manual review bindings", f"pair sidecar hash drifted: {pair_path}")
        pair_sidecar = _read_json(pair_path, "manual review pair sidecar")
        _require(isinstance(pair_sidecar, Mapping), "manual review pair sidecar", "root must be an object")
        _require(pair_sidecar.get("schema_version") == QUALITY_SCHEMA_VERSION, "manual review pair sidecar", "unsupported schema")
        _require(pair_sidecar.get("record_type") == "ovi_quality_pair", "manual review pair sidecar", "record_type is invalid")
        _require(
            set(pair_sidecar)
            == {
                "schema_version",
                "record_type",
                "quality_protocol_id",
                "quality_protocol_sha256",
                "measurement_index",
                "prompt_index",
                "sample_index",
                "dense",
                "candidate",
                "metrics",
                "automatic_acceptance",
                "comparison_script_sha256",
                "compare_media_script_sha256",
                "run_validator_script_sha256",
                "evaluation_matrix_sha256",
                "evaluator_source_receipt",
                "lpips_dependency_receipt",
                "media_tool_receipt",
            },
            "manual review pair sidecar",
            "pair field set differs from the fixed schema",
        )
        _require(pair_sidecar.get("automatic_acceptance") is None, "manual review pair sidecar", "automatic acceptance must remain null")
        _rendered_metrics, numeric_metrics = _normalize_persisted_metrics(
            pair_sidecar.get("metrics"),
            f"manual review pair {identity} metrics",
        )
        for field in MEDIAN_METRICS:
            numeric_by_metric[field].append(numeric_metrics[field])
        _require(pair_sidecar.get("quality_protocol_id") == report.get("quality_protocol_id"), "manual review pair sidecar", "protocol id differs from median")
        _require(pair_sidecar.get("quality_protocol_sha256") == report.get("quality_protocol_sha256"), "manual review pair sidecar", "protocol hash differs from median")
        _require(pair_sidecar.get("comparison_script_sha256") == report.get("comparison_script_sha256"), "manual review pair sidecar", "comparison script hash differs from median")
        _require(pair_sidecar.get("compare_media_script_sha256") == report.get("compare_media_script_sha256"), "manual review pair sidecar", "media script hash differs from median")
        _require(pair_sidecar.get("run_validator_script_sha256") == report.get("run_validator_script_sha256"), "manual review pair sidecar", "run validator script hash differs from median")
        _require(pair_sidecar.get("evaluation_matrix_sha256") == report.get("evaluation_matrix_sha256"), "manual review pair sidecar", "matrix hash differs from median")
        _require(pair_sidecar.get("evaluator_source_receipt") == report.get("evaluator_source_receipt"), "manual review pair sidecar", "evaluator source receipt differs from median")
        _require(pair_sidecar.get("lpips_dependency_receipt") == report.get("lpips_dependency_receipt"), "manual review pair sidecar", "LPIPS receipt differs from median")
        _require(pair_sidecar.get("media_tool_receipt") == report.get("media_tool_receipt"), "manual review pair sidecar", "media tool receipt differs from median")
        _require(
            tuple(pair_sidecar.get(field) for field in IDENTITY_FIELDS)
            == identity,
            "manual review pair sidecar",
            "artifact identity differs from median binding",
        )
        persisted_dense = pair_sidecar.get("dense")
        persisted_candidate = pair_sidecar.get("candidate")
        _require(isinstance(persisted_dense, Mapping), "manual review pair sidecar", "dense binding is missing")
        _require(isinstance(persisted_candidate, Mapping), "manual review pair sidecar", "candidate binding is missing")
        for key, value in report.get("dense_run", {}).items():
            _require(persisted_dense.get(key) == value, "manual review pair sidecar", f"dense run field {key} differs from median")
        for key, value in report.get("candidate_run", {}).items():
            _require(persisted_candidate.get(key) == value, "manual review pair sidecar", f"candidate run field {key} differs from median")
        _require(persisted_dense.get("artifact_sha256") == dense_hash, "manual review pair sidecar", "dense hash differs from median binding")
        _require(persisted_candidate.get("artifact_sha256") == candidate_hash, "manual review pair sidecar", "candidate hash differs from median binding")
        for side_name, side in (("dense", persisted_dense), ("candidate", persisted_candidate)):
            artifact_path_value = side.get("artifact_path")
            _require(isinstance(artifact_path_value, str) and artifact_path_value, "manual review pair sidecar", f"{side_name} artifact path is missing")
            artifact_path = Path(artifact_path_value).resolve()
            run_dir_value = side.get("run_dir")
            _require(isinstance(run_dir_value, str) and artifact_path.parent == Path(run_dir_value).resolve(), "manual review pair sidecar", f"{side_name} artifact escaped selected run")
            _require(artifact_path.is_file(), "manual review pair sidecar", f"{side_name} artifact is missing")
            _require(sha256(artifact_path) == side.get("artifact_sha256"), "manual review pair sidecar", f"{side_name} artifact hash drifted")
            metrics_path_value = side.get("metrics_sidecar_path")
            _require(isinstance(metrics_path_value, str) and metrics_path_value, "manual review pair sidecar", f"{side_name} metrics sidecar path is missing")
            metrics_path = Path(metrics_path_value).resolve()
            _require(metrics_path.parent == artifact_path.parent, "manual review pair sidecar", f"{side_name} metrics sidecar escaped selected run")
            _require(metrics_path.is_file(), "manual review pair sidecar", f"{side_name} metrics sidecar is missing")
            _require(sha256(metrics_path) == side.get("metrics_sidecar_sha256"), "manual review pair sidecar", f"{side_name} metrics sidecar hash drifted")
        expected[identity] = (dense_hash, candidate_hash)
    dense_run = report["dense_run"]
    prompt_count = dense_run.get("prompt_count")
    sample_count = dense_run.get("sample_count")
    _require(
        isinstance(prompt_count, int)
        and not isinstance(prompt_count, bool)
        and prompt_count > 0
        and isinstance(sample_count, int)
        and not isinstance(sample_count, bool)
        and sample_count > 0,
        "manual review bindings",
        "run prompt/sample cardinality is invalid",
    )
    expected_identities = {
        (measurement_index, prompt_index, sample_index)
        for measurement_index in EXPECTED_MEASUREMENT_INDICES
        for prompt_index in range(prompt_count)
        for sample_index in range(sample_count)
    }
    _require(
        set(expected) == expected_identities,
        "manual review bindings",
        "pair identities do not form the complete artifact matrix",
    )
    _require(report.get("automatic_acceptance") is None, "manual review bindings", "automatic acceptance must remain null")
    recomputed_medians: dict[str, Any] = {}
    for field, values in numeric_by_metric.items():
        value = float(statistics.median(values))
        recomputed_medians[field] = "inf" if math.isinf(value) else value
    _require(
        report.get("metric_medians") == recomputed_medians,
        "manual review bindings",
        "median metrics differ from the hash-bound pair sidecars",
    )
    return expected


def validate_manual_reviews(
    csv_path: Path,
    expected_bindings: Mapping[Any, tuple[str, str]],
    manual_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate only user-authored rows; never manufacture human judgments."""

    csv_path = Path(csv_path).resolve()
    context = "manual sync reviews"
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(tuple(reader.fieldnames or ()) == MANUAL_FIELDS, context, "CSV header differs from the fixed template")
            rows = list(reader)
    except OSError as exc:
        _fail(context, f"cannot read {csv_path}: {exc}")
    if not rows:
        return {
            "status": "empty",
            "row_count": 0,
            "csv_path": str(csv_path),
            "csv_sha256": sha256(csv_path),
        }
    normalized_bindings: dict[tuple[int, int, int], tuple[str, str]] = {}
    for supplied_identity, hashes in expected_bindings.items():
        identity = (
            (supplied_identity, 0, 0)
            if isinstance(supplied_identity, int)
            and not isinstance(supplied_identity, bool)
            else supplied_identity
        )
        _require(
            isinstance(identity, tuple)
            and len(identity) == 3
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in identity)
            and identity not in normalized_bindings,
            context,
            f"invalid or duplicate expected artifact identity {identity!r}",
        )
        normalized_bindings[identity] = hashes
    _require(len(rows) == len(normalized_bindings), context, "manual CSV must contain zero rows or all artifact rows")
    seen = set()
    allowed = set(manual_protocol["allowed_sync_ratings"])
    for row_number, row in enumerate(rows, start=2):
        row_context = f"{context} row {row_number}"
        _require(None not in row, row_context, "row contains extra CSV fields")
        identity_values = []
        for field in IDENTITY_FIELDS:
            try:
                value = int(row[field])
            except (TypeError, ValueError):
                _fail(row_context, f"{field} must be an integer")
            _require(row[field] == str(value), row_context, f"{field} must use canonical decimal form")
            identity_values.append(value)
        identity = tuple(identity_values)
        _require(identity in normalized_bindings and identity not in seen, row_context, f"invalid or duplicate artifact identity {identity}")
        seen.add(identity)
        expected_dense, expected_candidate = normalized_bindings[identity]
        _require(row["dense_artifact_sha256"] == expected_dense, row_context, "dense artifact hash does not match the paired sidecar")
        _require(row["candidate_artifact_sha256"] == expected_candidate, row_context, "candidate artifact hash does not match the paired sidecar")
        for field in manual_protocol["required_human_fields"]:
            _require(isinstance(row.get(field), str) and bool(row[field].strip()), row_context, f"human field {field} is blank")
        _require(UTC_TIMESTAMP.fullmatch(row["reviewed_at_utc"]) is not None, row_context, "reviewed_at_utc must be an explicit UTC timestamp ending in Z")
        try:
            reviewed_at = datetime.fromisoformat(
                row["reviewed_at_utc"].replace("Z", "+00:00")
            )
        except ValueError:
            _fail(row_context, "reviewed_at_utc is not a real calendar timestamp")
        _require(reviewed_at.utcoffset() == timedelta(0), row_context, "reviewed_at_utc must be UTC")
        _require(row["sync_rating"] in allowed, row_context, f"sync_rating must be one of {sorted(allowed)}")
    _require(seen == set(normalized_bindings), context, "manual rows must bind the complete artifact identity matrix")
    return {
        "status": "complete",
        "row_count": len(rows),
        "csv_path": str(csv_path),
        "csv_sha256": sha256(csv_path),
    }


def write_manual_validation_receipt(
    output_path: Path,
    *,
    median_path: Path,
    median_sha256: str,
    manual_status: Mapping[str, Any],
    expected_bindings: Mapping[Any, tuple[str, str]],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> Path:
    context = "manual validation receipt"
    _require(manual_status.get("status") == "complete", context, "only a complete human review can produce a validation receipt")
    csv_path_value = manual_status.get("csv_path")
    _require(isinstance(csv_path_value, str) and csv_path_value, context, "manual CSV path is missing")
    csv_path = Path(csv_path_value).resolve()
    expected_csv_hash = _full_sha(manual_status.get("csv_sha256"), context, "manual CSV SHA256")
    _require(csv_path.is_file(), context, "manual CSV is missing")
    _require(sha256(csv_path) == expected_csv_hash, context, "manual CSV changed after validation")
    median_path = Path(median_path).resolve()
    _require(median_path.is_file(), context, "quality median is missing")
    _require(sha256(median_path) == _full_sha(median_sha256, context, "median SHA256"), context, "quality median changed after validation")
    output_path = Path(output_path).resolve()
    _require(output_path.parent == median_path.parent, context, "manual receipt must stay beside the quality median")
    normalized_bindings = {
        (
            (identity, 0, 0)
            if isinstance(identity, int) and not isinstance(identity, bool)
            else identity
        ): hashes
        for identity, hashes in expected_bindings.items()
    }
    _require(
        len(normalized_bindings) == len(expected_bindings)
        and all(
            isinstance(identity, tuple)
            and len(identity) == 3
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in identity)
            for identity in normalized_bindings
        ),
        context,
        "expected manual bindings contain invalid artifact identities",
    )
    payload = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "record_type": "ovi_manual_sync_review_validation",
        "quality_protocol_id": protocol["protocol_id"],
        "quality_protocol_sha256": protocol_sha256,
        "quality_median_path": str(median_path),
        "quality_median_sha256": median_sha256,
        "manual_reviews_csv_path": str(csv_path),
        "manual_reviews_csv_sha256": expected_csv_hash,
        "manual_review_status": "complete",
        "manual_review_row_count": manual_status.get("row_count"),
        "pairs": [
            {
                "measurement_index": identity[0],
                "prompt_index": identity[1],
                "sample_index": identity[2],
                "dense_artifact_sha256": normalized_bindings[identity][0],
                "candidate_artifact_sha256": normalized_bindings[identity][1],
            }
            for identity in sorted(normalized_bindings)
        ],
    }
    _write_json(output_path, payload)
    return output_path


def write_quality_sidecars(report: Mapping[str, Any], output_dir: Path) -> Path:
    """Write immutable identity pair sidecars and one hash-bound median."""

    output_dir = Path(output_dir).resolve()
    median_path = output_dir / "median.quality.json"
    pairs = report.get("pairs")
    _require(
        isinstance(pairs, list)
        and isinstance(report.get("pair_count"), int)
        and not isinstance(report.get("pair_count"), bool)
        and len(pairs) == report.get("pair_count")
        and len(pairs) > 0,
        "quality output",
        "report pair_count is invalid",
    )
    _require(not output_dir.exists(), "quality output", f"refusing to overwrite or reuse output directory {output_dir}")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        _fail("quality output", f"cannot create {output_dir}: {exc}")

    created: list[Path] = []
    try:
        pair_bindings = []
        previous_identity = None
        for pair_offset, pair in enumerate(pairs):
            _require(isinstance(pair, Mapping), "quality output", f"pair {pair_offset} must be an object")
            identity = tuple(pair.get(field) for field in IDENTITY_FIELDS)
            _require(
                all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in identity)
                and (previous_identity is None or identity > previous_identity),
                "quality output",
                f"pair identity order is invalid at {identity!r}",
            )
            previous_identity = identity
            path = output_dir / (
                f"measurement_{identity[0]:02d}_prompt_{identity[1]:03d}_"
                f"sample_{identity[2]:03d}.quality.json"
            )
            pair_payload = {
                **pair,
                "comparison_script_sha256": report["comparison_script_sha256"],
                "compare_media_script_sha256": report["compare_media_script_sha256"],
                "run_validator_script_sha256": report["run_validator_script_sha256"],
                "evaluation_matrix_sha256": report["evaluation_matrix_sha256"],
                "evaluator_source_receipt": report["evaluator_source_receipt"],
                "lpips_dependency_receipt": report["lpips_dependency_receipt"],
                "media_tool_receipt": report["media_tool_receipt"],
            }
            _write_json(path, pair_payload)
            created.append(path)
            pair_bindings.append(
                {
                    "measurement_index": identity[0],
                    "prompt_index": identity[1],
                    "sample_index": identity[2],
                    "pair_sidecar_path": str(path),
                    "pair_sidecar_sha256": sha256(path),
                    "dense_artifact_sha256": pair["dense"]["artifact_sha256"],
                    "candidate_artifact_sha256": pair["candidate"]["artifact_sha256"],
                }
            )
        median_payload = {
            key: value
            for key, value in report.items()
            if key not in {"pairs"}
        }
        median_payload["record_type"] = "ovi_quality_median"
        median_payload["pairs"] = pair_bindings
        _write_json(median_path, median_payload)
        created.append(median_path)
    except Exception:
        for path in created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise
    return median_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audited Ovi dense-to-candidate quality comparison")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="compute the complete identity-paired quality matrix")
    compare.add_argument("--dense-run", type=Path, required=True)
    compare.add_argument("--candidate-run", type=Path, required=True)
    compare.add_argument("--candidate-method-id", required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    compare.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    compare.add_argument("--lpips-receipt", type=Path)

    validate_manual = subparsers.add_parser("validate-manual", help="validate a user-authored manual sync CSV against a quality report")
    validate_manual.add_argument("--quality-report", type=Path, required=True)
    validate_manual.add_argument("--manual-reviews", type=Path, required=True)
    validate_manual.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    validate_manual.add_argument(
        "--output",
        type=Path,
        help="exclusive validation receipt path (default: beside median)",
    )

    validate_receipt = subparsers.add_parser("validate-receipt", help="validate fixed LPIPS modules and weights without computing metrics")
    validate_receipt.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    validate_receipt.add_argument("--lpips-receipt", type=Path)
    return parser


def _run_compare(args: argparse.Namespace) -> Path:
    evaluator_source = capture_evaluator_source_receipt(args.protocol, args.matrix)
    protocol, protocol_hash = load_quality_protocol(args.protocol)
    matrix = _run_validator_module().load_manifest(args.matrix)
    _require(args.candidate_method_id != "dense", "quality command", "candidate-method-id cannot be dense")
    dense = load_audited_run(args.dense_run, "dense", matrix)
    candidate = load_audited_run(args.candidate_run, args.candidate_method_id, matrix)
    receipt = validate_lpips_receipt(protocol["lpips"], receipt_path=args.lpips_receipt)
    tools_receipt = collect_media_tool_receipt()
    tool_paths = validate_media_tool_receipt(tools_receipt)
    lpips_runner = LpipsAlexCpu(protocol["lpips"], tool_paths)
    report = build_quality_report(
        dense,
        candidate,
        protocol,
        protocol_sha256=protocol_hash,
        lpips_receipt=receipt,
        media_tool_receipt=tools_receipt,
        evaluator_source_receipt=evaluator_source,
        metric_runner=make_metric_runner(lpips_runner, tool_paths),
    )
    post_metric_receipt = validate_lpips_receipt(
        protocol["lpips"], receipt_path=args.lpips_receipt
    )
    _require(post_metric_receipt == receipt, "LPIPS dependency receipt", "dependencies or weights changed while metrics were running")
    validate_media_tool_receipt(tools_receipt)
    validate_evaluator_source_receipt(evaluator_source)
    for run in (dense, candidate):
        _assert_run_evidence(run, "immediately before sidecar write")
    return write_quality_sidecars(report, args.output_dir)


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "compare":
            output = _run_compare(args)
            print(output)
        elif args.command == "validate-manual":
            _require(Path(args.protocol).resolve() == DEFAULT_PROTOCOL.resolve(), "manual validation", "only the checked-in fixed protocol is accepted")
            capture_evaluator_source_receipt(args.protocol, DEFAULT_MATRIX)
            protocol, protocol_hash = load_quality_protocol(args.protocol)
            _report, bindings, median_hash = load_validated_quality_median(
                args.quality_report,
                protocol,
                protocol_hash,
            )
            status = validate_manual_reviews(
                args.manual_reviews,
                bindings,
                protocol["manual_reviews"],
            )
            if status["status"] == "complete":
                output = args.output or (
                    Path(args.quality_report).resolve().parent
                    / "manual-review.validation.json"
                )
                receipt_path = write_manual_validation_receipt(
                    output,
                    median_path=args.quality_report,
                    median_sha256=median_hash,
                    manual_status=status,
                    expected_bindings=bindings,
                    protocol=protocol,
                    protocol_sha256=protocol_hash,
                )
                print(receipt_path)
            else:
                print(json.dumps(status, sort_keys=True, allow_nan=False))
        elif args.command == "validate-receipt":
            capture_evaluator_source_receipt(args.protocol, DEFAULT_MATRIX)
            protocol, _ = load_quality_protocol(args.protocol)
            receipt = validate_lpips_receipt(protocol["lpips"], receipt_path=args.lpips_receipt)
            print(json.dumps(receipt, sort_keys=True, allow_nan=False))
        else:
            raise AssertionError(f"unexpected command {args.command}")
    except (QualityError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
