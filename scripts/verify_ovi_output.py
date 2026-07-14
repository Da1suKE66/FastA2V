#!/usr/bin/env python3
"""Fail unless every generated MP4 has valid Ovi video and non-silent audio."""

import argparse
import hashlib
import json
import math
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovi.block_cache import fixed_block_cache_metric_errors
from ovi.eval_protocol import prompt_sequence_sha256, validate_run_protocol
from ovi.gpu_process_monitor import (
    GPU_EVIDENCE_SCHEMA_VERSION,
    GPU_PROCESS_MONITOR_SCHEMA_VERSION,
    GPU_QUERY_CADENCE_TOLERANCE_SECONDS,
    gpu_compute_snapshot_maximum_gap_seconds,
    gpu_compute_snapshot_observation_span_seconds,
    gpu_compute_snapshot_sequence_errors,
    gpu_compute_snapshot_errors,
    trusted_nvidia_smi_metadata_errors,
    validate_pre_run_gpu_report,
)
from ovi.sparge_evidence import (
    sparge_microtest_evidence_errors,
    sparge_receipt_evidence_errors,
)
from ovi.radial_evidence import (
    FLASHINFER_VERSION,
    RADIAL_BLOCK_SIZE,
    RADIAL_COMMIT,
    RADIAL_EMPTY_ROWS,
    RADIAL_GRID,
    RADIAL_MASK_API,
    RADIAL_MODEL_TYPE,
    RADIAL_PREFIX_SEQUENCE,
    RADIAL_PROFILE_AUDITS,
    RADIAL_REPOSITORY,
    RADIAL_SEQUENCE,
    RADIAL_TAIL_SEQUENCE,
    flashinfer_manifest_evidence_errors,
    radial_microtest_evidence_errors,
    radial_receipt_evidence_errors,
)


SPARGE_PROVENANCE = {
    "backend": "official_spargeattn",
    "repository": "https://github.com/thu-ml/SpargeAttn.git",
    "clone_url": "ssh://git@ssh.github.com:443/thu-ml/SpargeAttn.git",
    "pinned_commit": "ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a",
    "api": "spas_sage2_attn_meansim_topk_cuda",
    "tensor_layout": "NHD",
    "return_sparsity": False,
}


class MediaCommandError(subprocess.SubprocessError):
    """Fail-closed ffmpeg/ffprobe failure with stderr-only diagnostics."""

    def __init__(self, command, returncode, stderr):
        executable = Path(str(command[0])).name if command else "media command"
        stderr = bytes(stderr or b"")
        rendered_stderr = stderr.decode("utf-8", errors="replace").strip()
        if not rendered_stderr and stderr:
            rendered_stderr = repr(stderr)
        if returncode:
            message = f"{executable} exited with status {returncode}"
        else:
            message = (
                f"{executable} emitted error-level stderr despite exit status 0"
            )
        if rendered_stderr:
            message += f": {rendered_stderr}"
        else:
            message += " without stderr diagnostics"
        super().__init__(message)
        self.command = tuple(command)
        self.returncode = returncode
        self.stderr = stderr


class EvidenceSnapshotError(ValueError):
    """Raised when a persisted evidence file cannot be read immutably."""


class _StableFileSnapshot:
    """One no-follow regular-file read plus identity for final revalidation."""

    __slots__ = (
        "path",
        "data",
        "sha256",
        "device",
        "inode",
        "size",
        "mtime_ns",
        "ctime_ns",
    )

    def __init__(self, path, data, metadata):
        self.path = Path(path)
        self.data = data
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.device = metadata.st_dev
        self.inode = metadata.st_ino
        self.size = metadata.st_size
        self.mtime_ns = metadata.st_mtime_ns
        self.ctime_ns = metadata.st_ctime_ns


class _AbsentPathGuard:
    """Publication guard requiring a path to remain absent."""

    __slots__ = ("path",)

    def __init__(self, path):
        self.path = _canonical_leaf_path(path)


def _snapshot_identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_file_snapshot(path):
    """Read one regular file without following symlinks or accepting replacement."""

    path = Path(path)
    try:
        initial_entry = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise EvidenceSnapshotError(
            f"cannot stat evidence file {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(initial_entry.st_mode):
        raise EvidenceSnapshotError(f"evidence file must not be a symlink: {path}")
    if not stat.S_ISREG(initial_entry.st_mode):
        raise EvidenceSnapshotError(f"evidence file must be a regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceSnapshotError(
            f"cannot open no-follow evidence file {path}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceSnapshotError(
                f"opened evidence is not a regular file: {path}"
            )
        if _snapshot_identity(before) != _snapshot_identity(initial_entry):
            raise EvidenceSnapshotError(f"evidence file changed before read: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    if _snapshot_identity(after) != _snapshot_identity(before):
        raise EvidenceSnapshotError(f"evidence file changed while being read: {path}")
    if len(data) != after.st_size:
        raise EvidenceSnapshotError(
            f"evidence byte count changed while being read: {path}"
        )
    try:
        final_entry = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise EvidenceSnapshotError(
            f"cannot re-stat evidence file {path}: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(final_entry.st_mode)
        or _snapshot_identity(final_entry) != _snapshot_identity(after)
    ):
        raise EvidenceSnapshotError(
            f"evidence file was replaced while being read: {path}"
        )
    return _StableFileSnapshot(path, data, after)


def _revalidate_snapshot(snapshot):
    try:
        current = os.stat(snapshot.path, follow_symlinks=False)
    except OSError as exc:
        raise EvidenceSnapshotError(
            f"cannot revalidate evidence file {snapshot.path}: {exc}"
        ) from exc
    expected = (
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )
    if not stat.S_ISREG(current.st_mode) or _snapshot_identity(current) != expected:
        raise EvidenceSnapshotError(
            "evidence file changed after its stable byte snapshot: "
            f"{snapshot.path}"
        )


def _revalidate_publication_guard(guard):
    if isinstance(guard, _AbsentPathGuard):
        if os.path.lexists(guard.path):
            raise EvidenceSnapshotError(
                f"guarded absent evidence path appeared: {guard.path}"
            )
        return
    _revalidate_snapshot(guard)


def _snapshot_jsonl(path):
    snapshot = _stable_file_snapshot(path)
    try:
        lines = snapshot.data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvidenceSnapshotError(
            f"cannot decode UTF-8 JSONL from {path}: {exc}"
        ) from exc
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvidenceSnapshotError(
                f"blank JSONL record at {path}:{line_number}"
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceSnapshotError(
                f"invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise EvidenceSnapshotError(
                f"record at {path}:{line_number} is not an object"
            )
        records.append(record)
    return snapshot, records


def _canonical_leaf_path(path):
    """Canonicalize the parent while deliberately not following the leaf."""

    path = Path(path)
    return path.parent.resolve(strict=True) / path.name


def _json_from_snapshot(snapshot, context=None):
    context = context or str(snapshot.path)
    try:
        payload = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceSnapshotError(
            f"invalid JSON evidence file {context}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvidenceSnapshotError(
            f"JSON evidence file is not an object: {context}"
        )
    return payload


def _snapshot_json(path):
    snapshot = _stable_file_snapshot(path)
    payload = _json_from_snapshot(snapshot, path)
    return snapshot, payload


def _file_binding(snapshot):
    return {
        "path": str(_canonical_leaf_path(snapshot.path)),
        "bytes": snapshot.size,
        "sha256": snapshot.sha256,
    }


def _jsonl_binding(snapshot, records):
    binding = _file_binding(snapshot)
    binding["record_count"] = len(records)
    return binding


def _warmup_timings_binding(snapshot, records):
    # Kept as a named compatibility helper for callers and tests.
    return _jsonl_binding(snapshot, records)


def _materialize_snapshot(snapshot, suffix):
    """Materialize immutable bytes for external decoders without reopening source."""

    descriptor, name = tempfile.mkstemp(prefix="fasta2v-verify-", suffix=suffix)
    try:
        offset = 0
        while offset < len(snapshot.data):
            offset += os.write(descriptor, snapshot.data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return Path(name)


def _binding_shape_errors(binding, expected_keys, context):
    errors = []
    if not isinstance(binding, dict):
        return [f"{context} must be a JSON object"]
    if set(binding) != set(expected_keys):
        errors.append(
            f"{context} fields {sorted(binding)} != {sorted(expected_keys)}"
        )
    if not isinstance(binding.get("path"), str) or not binding.get("path"):
        errors.append(f"{context} path must be a non-empty string")
    if not is_json_int(binding.get("bytes")) or binding.get("bytes") < 0:
        errors.append(f"{context} bytes must be a nonnegative JSON integer")
    digest = binding.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        errors.append(f"{context} sha256 must be a lowercase SHA256 digest")
    if "record_count" in expected_keys and (
        not is_json_int(binding.get("record_count"))
        or binding.get("record_count") < 0
    ):
        errors.append(
            f"{context} record_count must be a nonnegative JSON integer"
        )
    return errors


def run(command):
    """Run a media command without a controlling stdin or mixed output pipes."""

    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout = bytes(completed.stdout or b"")
    stderr = bytes(completed.stderr or b"")
    if completed.returncode != 0 or stderr:
        raise MediaCommandError(command, completed.returncode, stderr)
    return stdout


def probe(path):
    payload = run([
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ])
    return json.loads(payload)


def decode_audio(path):
    raw = run([
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ])
    return np.frombuffer(raw, dtype="<f4")


def decode_video_gray(path, expected_frames):
    if type(expected_frames) is not int or expected_frames < 1:
        raise ValueError("expected decoded video frame count must be a positive integer")
    raw = run([
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        "scale=64:64,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ])
    bytes_per_frame = 64 * 64
    expected_bytes = expected_frames * bytes_per_frame
    actual_bytes = len(raw)
    if actual_bytes != expected_bytes:
        complete_frames, trailing_bytes = divmod(actual_bytes, bytes_per_frame)
        raise ValueError(
            "decoded 64x64 gray video byte length mismatch: "
            f"expected {expected_bytes} bytes for {expected_frames} frames, "
            f"found {actual_bytes} bytes ({complete_frames} complete frames, "
            f"{trailing_bytes} trailing bytes)"
        )
    return np.frombuffer(raw, dtype=np.uint8)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def as_int(value):
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_json_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def strict_json_equal(actual, expected):
    """Compare persisted JSON without Python's bool/int equivalence."""

    if expected is None or isinstance(expected, (bool, str)):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, int):
        return is_json_int(actual) and actual == expected
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        try:
            return float(actual) == expected
        except (OverflowError, ValueError):
            return False
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                strict_json_equal(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                strict_json_equal(actual[key], expected_value)
                for key, expected_value in expected.items()
            )
        )
    return type(actual) is type(expected) and actual == expected


def _nonnegative_json_int(value, default=-1):
    return value if is_json_int(value) and value >= 0 else default


def measurement_record_protocol_errors(timings, environment):
    """Validate the exact measurement/prompt/sample Cartesian product.

    Inference writes records in measurement-major, prompt-major, sample-major
    order.  The first measurement's sample-zero records bind the ordered prompt
    sequence to ``environment.prompts_sha256``; every repeat must then retain the
    same prompt at each prompt index.  Seeds are fixed to ``base + sample_index``.
    """

    errors = []
    if not isinstance(timings, list):
        return ["timings.jsonl records must be a list"], []
    if not isinstance(environment, dict):
        return ["measurement environment must be a JSON object"], []

    measurement_runs = _nonnegative_json_int(environment.get("measurement_runs"))
    prompt_count = _nonnegative_json_int(environment.get("prompt_count"))
    sample_count = _nonnegative_json_int(environment.get("each_example_n_times"))
    expected_record_count = _nonnegative_json_int(
        environment.get("expected_measurement_records")
    )
    base_seed = environment.get("seed")
    if measurement_runs < 1:
        errors.append("measurement_runs must be a positive JSON integer")
    if prompt_count < 1:
        errors.append("prompt_count must be a positive JSON integer")
    if sample_count < 1:
        errors.append("each_example_n_times must be a positive JSON integer")
    if expected_record_count < 0:
        errors.append(
            "expected_measurement_records must be a non-negative JSON integer"
        )
    if not is_json_int(base_seed):
        errors.append("seed must be a JSON integer")

    if errors:
        return errors, []

    expected_keys = [
        (measurement_index, prompt_index, sample_index)
        for measurement_index in range(measurement_runs)
        for prompt_index in range(prompt_count)
        for sample_index in range(sample_count)
    ]
    if expected_record_count != len(expected_keys):
        errors.append(
            "expected_measurement_records does not equal measurement_runs * "
            "prompt_count * each_example_n_times"
        )
    actual_keys = []
    records_by_key = {}
    for record_offset, record in enumerate(timings):
        context = f"measurement record[{record_offset}]"
        if not isinstance(record, dict):
            errors.append(f"{context} must be a JSON object")
            continue
        values = []
        for field in ("measurement_index", "prompt_index", "sample_index"):
            value = record.get(field)
            if not is_json_int(value):
                errors.append(f"{context} {field} must be a JSON integer")
                values.append(None)
            else:
                values.append(value)
        key = tuple(values)
        actual_keys.append(key)
        if None not in key:
            if key in records_by_key:
                errors.append(f"duplicate measurement/prompt/sample key {key}")
            else:
                records_by_key[key] = record

        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            errors.append(f"{context} prompt must be a non-empty string")
        seed = record.get("seed")
        if not is_json_int(seed):
            errors.append(f"{context} seed must be a JSON integer")
        elif is_json_int(record.get("sample_index")):
            expected_seed = base_seed + record["sample_index"]
            if seed != expected_seed:
                errors.append(
                    f"{context} seed={seed!r} != fixed seed {expected_seed}"
                )

    expected_key_set = set(expected_keys)
    actual_valid_keys = {key for key in actual_keys if None not in key}
    if actual_valid_keys != expected_key_set or len(actual_keys) != len(expected_keys):
        missing = sorted(expected_key_set - actual_valid_keys)
        unexpected = sorted(actual_valid_keys - expected_key_set)
        errors.append(
            "measurement/prompt/sample Cartesian product is incomplete or invalid: "
            f"missing={missing} unexpected={unexpected} "
            f"records={len(actual_keys)} expected={len(expected_keys)}"
        )
    if actual_keys != expected_keys:
        errors.append(
            "timings.jsonl record order must be measurement-major, prompt-major, "
            "sample-major"
        )

    ordered_prompts = []
    for prompt_index in range(prompt_count):
        record = records_by_key.get((0, prompt_index, 0))
        prompt = record.get("prompt") if isinstance(record, dict) else None
        if not isinstance(prompt, str) or not prompt:
            errors.append(
                "cannot bind ordered prompt sequence from measurement 0, "
                f"prompt {prompt_index}, sample 0"
            )
            ordered_prompts.append(None)
        else:
            ordered_prompts.append(prompt)

    if all(isinstance(prompt, str) for prompt in ordered_prompts):
        actual_prompt_hash = prompt_sequence_sha256(ordered_prompts)
        expected_prompt_hash = environment.get("prompts_sha256")
        if not isinstance(expected_prompt_hash, str) or (
            actual_prompt_hash != expected_prompt_hash
        ):
            errors.append(
                "ordered measurement prompts do not match the fixed environment "
                "prompt hash"
            )
        for key, record in records_by_key.items():
            prompt_index = key[1]
            if 0 <= prompt_index < prompt_count and (
                record.get("prompt") != ordered_prompts[prompt_index]
            ):
                errors.append(
                    f"measurement/prompt/sample key {key} changed prompt text"
                )

        for measurement_index in range(measurement_runs):
            measurement_prompts = []
            for prompt_index in range(prompt_count):
                record = records_by_key.get((measurement_index, prompt_index, 0))
                measurement_prompts.append(
                    record.get("prompt") if isinstance(record, dict) else None
                )
            if measurement_prompts != ordered_prompts:
                errors.append(
                    f"measurement index {measurement_index} changed prompt order"
                )
            elif prompt_sequence_sha256(measurement_prompts) != actual_prompt_hash:
                errors.append(
                    f"measurement index {measurement_index} prompt hash changed"
                )

    return errors, ordered_prompts


def warmup_record_protocol_errors(warmups, environment, ordered_prompts):
    """Require every excluded warm-up to use only the first fixed prompt."""

    errors = []
    if not isinstance(warmups, list):
        return ["warmup_timings.jsonl records must be a list"]
    if not isinstance(environment, dict):
        return ["warmup environment must be a JSON object"]

    expected_warmups = _nonnegative_json_int(
        environment.get("expected_warmup_records")
    )
    warmup_runs = _nonnegative_json_int(environment.get("warmup_runs"))
    base_seed = environment.get("seed")
    if expected_warmups < 0 or warmup_runs < 0:
        errors.append("warmup counts must be non-negative JSON integers")
        return errors
    if expected_warmups != warmup_runs:
        errors.append(
            "expected_warmup_records must equal warmup_runs for the fixed protocol"
        )
    if len(warmups) != expected_warmups:
        errors.append(
            f"warmup count {len(warmups)} != expected {expected_warmups}"
        )
    if not is_json_int(base_seed):
        errors.append("warmup seed must be a JSON integer")

    first_prompt = ordered_prompts[0] if ordered_prompts else None
    if expected_warmups and (not isinstance(first_prompt, str) or not first_prompt):
        errors.append("warm-up cannot bind the first fixed prompt")

    for record_offset, record in enumerate(warmups):
        context = f"warmup[{record_offset}]"
        if not isinstance(record, dict):
            errors.append(f"{context} must be a JSON object")
            continue
        if record.get("status") != "ok" or record.get("record_type") != "warmup":
            errors.append(f"{context} status/type is not ok/warmup")
        warmup_index = record.get("warmup_index")
        if not is_json_int(warmup_index) or warmup_index != record_offset:
            errors.append(
                f"{context} warmup_index must be integer {record_offset}"
            )
        if record.get("benchmark_valid") is not False:
            errors.append(f"{context} benchmark_valid must be false")
        if record.get("prompt") != first_prompt:
            errors.append(f"{context} must use only the first fixed prompt")
        if not is_json_int(record.get("seed")) or record.get("seed") != base_seed:
            errors.append(f"{context} seed must equal the fixed base seed")
    return errors


def warmup_timings_binding_errors(path, binding, expected_warmups):
    """Revalidate a persisted warm-up binding with strict JSON field types."""

    errors = []
    try:
        path = _canonical_leaf_path(path)
    except OSError as exc:
        return [f"warmup timings path cannot be canonicalized: {exc}"]
    if not is_json_int(expected_warmups) or expected_warmups < 0:
        return ["expected warm-up count must be a non-negative JSON integer"]
    if expected_warmups == 0:
        if binding is not None:
            errors.append("zero-warmup protocol must persist a null binding")
        if os.path.lexists(path):
            errors.append("zero-warmup protocol must not create warmup_timings.jsonl")
        return errors

    if not isinstance(binding, dict):
        return ["warmup_timings_binding must be a JSON object"]
    expected_fields = {"path", "bytes", "sha256", "record_count"}
    if set(binding) != expected_fields:
        errors.append("warmup_timings_binding fields are invalid")
    if binding.get("path") != str(path):
        errors.append("warmup_timings_binding path is not the canonical run path")
    for field in ("bytes", "record_count"):
        value = binding.get(field)
        if not is_json_int(value) or value < 0:
            errors.append(
                f"warmup_timings_binding {field} must be a non-negative JSON integer"
            )
    digest = binding.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        errors.append("warmup_timings_binding sha256 is invalid")

    try:
        snapshot, records = _snapshot_jsonl(path)
    except EvidenceSnapshotError as exc:
        errors.append(f"warmup_timings_binding cannot be revalidated: {exc}")
        return errors
    actual = _warmup_timings_binding(snapshot, records)
    if not strict_json_equal(binding, actual):
        errors.append(
            "warmup_timings.jsonl bytes, hash, or record count changed"
        )
    if len(records) != expected_warmups:
        errors.append(
            "warmup_timings.jsonl record count differs from the fixed protocol"
        )
    return errors


def expected_radial_runtime_dependency_evidence(receipt):
    """Summarize the exact runtime ELF inventory bound by a Radial receipt."""

    if not isinstance(receipt, dict):
        return None
    inventory = receipt.get("runtime_loaded_dependencies")
    if not isinstance(inventory, dict) or not inventory:
        return None
    mapped_paths = set()
    for fingerprints in inventory.values():
        if not isinstance(fingerprints, list) or not fingerprints:
            return None
        for metadata in fingerprints:
            if not isinstance(metadata, dict):
                return None
            path = metadata.get("path")
            if not isinstance(path, str) or not path:
                return None
            mapped_paths.add(path)
    try:
        canonical = json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return {
        "status": "ok",
        "aliases": len(inventory),
        "mapped_files": len(mapped_paths),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def validate_radial_runtime_dependency_evidence(
    evidence,
    expected,
    context,
    errors,
):
    """Require one runtime mapping summary to equal the copied receipt."""

    if not isinstance(evidence, dict):
        errors.append(f"{context}: runtime dependency evidence is missing")
        return
    if set(evidence) != {
        "status",
        "aliases",
        "mapped_files",
        "inventory_sha256",
    }:
        errors.append(f"{context}: runtime dependency fields are invalid")
    if evidence.get("status") != "ok":
        errors.append(f"{context}: runtime dependency status is not ok")
    for field in ("aliases", "mapped_files"):
        value = evidence.get(field)
        if not is_json_int(value) or value <= 0:
            errors.append(
                f"{context}: runtime dependency {field} must be a positive "
                "JSON integer"
            )
    inventory_sha256 = evidence.get("inventory_sha256")
    if (
        not isinstance(inventory_sha256, str)
        or len(inventory_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in inventory_sha256
        )
    ):
        errors.append(
            f"{context}: runtime dependency inventory_sha256 is invalid"
        )
    if expected is None:
        errors.append(f"{context}: copied receipt has no runtime inventory")
    elif not strict_json_equal(evidence, expected):
        errors.append(
            f"{context}: runtime dependency evidence differs from copied receipt"
        )


def validate_radial_optional_import_loader_evidence(
    evidence,
    expected_runtime,
    expected_removed_path,
    context,
    errors,
):
    """Validate the only loader mutation permitted for the fixed OpenCV build."""

    if not isinstance(evidence, dict):
        errors.append(f"{context}: optional-import loader evidence is missing")
        return
    if set(evidence) != {
        "status",
        "restored",
        "removed_prepend_paths",
        "runtime_dependencies",
    }:
        errors.append(f"{context}: optional-import loader fields are invalid")
    if evidence.get("status") != "ok" or evidence.get("restored") is not True:
        errors.append(f"{context}: audited loader environment was not restored")
    validate_radial_runtime_dependency_evidence(
        evidence.get("runtime_dependencies"),
        expected_runtime,
        f"{context}.runtime_dependencies",
        errors,
    )
    removed = evidence.get("removed_prepend_paths")
    if not isinstance(removed, list) or len(removed) != 1:
        errors.append(
            f"{context}: expected exactly one OpenCV loader prepend path"
        )
        return
    removed_path = removed[0]
    if not isinstance(removed_path, str) or not removed_path:
        errors.append(f"{context}: removed loader prepend path is invalid")
        return
    if str(Path(removed_path).resolve()) != str(Path(expected_removed_path).resolve()):
        errors.append(
            f"{context}: removed loader prepend path is not the fixed env lib64"
        )


def validate_radial_preflight_static_evidence(
    evidence,
    errors,
    *,
    context="preflight Radial",
):
    """Validate the fixed Radial preflight values with exact JSON types."""

    if not isinstance(evidence, dict):
        errors.append(f"{context}: evidence is missing")
        return
    expected = {
        "pinned_commit": RADIAL_COMMIT,
        "mask_api": RADIAL_MASK_API,
        "source_files_verified": True,
        "flashinfer_files_verified": True,
        "flashinfer_manifest_verified": True,
        "runtime_loader_environment_verified": True,
        "cpu_mask_audits_verified": True,
        "flashinfer_version": FLASHINFER_VERSION,
        "flashinfer_apis": {
            "BlockSparseAttentionWrapper": True,
            "single_prefill_with_kv_cache": True,
            "merge_state": True,
        },
        "derived_mask_api_callable": True,
        "install_cuda_kernel_launched": False,
        "preflight_cuda_microtest_required": True,
    }
    for field, expected_value in expected.items():
        actual_value = evidence.get(field)
        if not strict_json_equal(actual_value, expected_value):
            errors.append(
                f"{context} {field}={actual_value!r} != {expected_value!r}"
            )


def validated_radial_preflight_claims(microtest, validation_errors):
    """Expose claim limits only after canonical Radial evidence validation."""

    if validation_errors or not isinstance(microtest, dict):
        return None
    binding = microtest.get("gpu_process_binding")
    if not isinstance(binding, dict):
        return None
    mps = binding.get("mps")
    if not isinstance(mps, dict):
        return None
    pmon = mps.get("pmon")
    if not isinstance(pmon, dict):
        return None
    return {
        "status": "validated",
        "source": "preflight.json.radialattn_microtest",
        "pmon_status": pmon.get("status"),
        "pmon_collection_status": pmon.get("collection_status"),
        "pmon_observation_mode": microtest.get("pmon_observation_mode"),
        "mps_status": microtest.get("mps_status"),
        "binding_method": microtest.get("pid_binding_method"),
        "claim_scope": microtest.get("gpu_process_claim_scope"),
        "host_pid_ownership": microtest.get("host_pid_ownership"),
        "direct_compute_type_observed": pmon.get(
            "direct_compute_type_observed"
        ),
        "host_pid_observed_by_pmon": mps.get("host_pid_observed_by_pmon"),
        "continuous_exclusivity_proven": pmon.get(
            "continuous_exclusivity_proven"
        ),
    }


def validate_pre_run_gpu(report, environment, errors):
    """Validate idle evidence and return its physical-GPU identity tuple."""
    if not isinstance(report, dict):
        errors.append("pre_run_gpu.json must be a JSON object")
        return None
    errors.extend(
        f"pre-run GPU {error}"
        for error in validate_pre_run_gpu_report(
            report,
            cuda_visible_devices=report.get("cuda_visible_devices"),
        )
    )
    device_uuid = report.get("device_uuid")
    device_name = report.get("device_name")
    if not isinstance(device_uuid, str) or not device_uuid:
        errors.append("pre-run GPU UUID is missing")
    if not isinstance(device_name, str) or not device_name:
        errors.append("pre-run GPU name is missing")
    if not is_json_int(environment.get("gpu_physical_index")):
        errors.append("environment gpu_physical_index must be an integer")

    expected_environment = {
        "gpu_physical_index": 0,
        "gpu_uuid": device_uuid,
        "gpu_name": device_name,
        "gpu": device_name,
        "pre_run_gpu_valid": True,
        "cuda_visible_devices": report.get("cuda_visible_devices"),
    }
    for field, expected in expected_environment.items():
        if environment.get(field) != expected:
            errors.append(
                f"environment {field}={environment.get(field)!r} does not "
                f"match pre-run GPU evidence {expected!r}"
            )
    if (
        not device_uuid
        or not device_name
        or not is_json_int(report.get("physical_device_index"))
        or not is_json_int(report.get("device_index"))
    ):
        return None
    return (0, device_uuid, device_name)


def validate_gpu_monitor(
    monitor,
    expected_identity,
    expected_nvidia_smi_binary,
    expected_boot_id,
    expected_interval_seconds,
    minimum_coverage_seconds,
    candidate,
    context,
    errors,
):
    """Cross-bind every nvidia-smi sample to pre-run physical GPU 0."""
    if not isinstance(monitor, dict):
        errors.append(f"{context}: gpu_process_monitor must be a JSON object")
        return
    if (
        not is_json_int(monitor.get("schema_version"))
        or monitor.get("schema_version")
        != GPU_PROCESS_MONITOR_SCHEMA_VERSION
    ):
        errors.append(
            f"{context}: unsupported GPU process monitor evidence schema"
        )
    if expected_identity is not None:
        expected_index, expected_uuid, expected_name = expected_identity
        summary_identity = (
            monitor.get("device_index"),
            monitor.get("device_uuid"),
            monitor.get("device_name"),
        )
        if summary_identity != expected_identity:
            errors.append(
                f"{context}: monitor GPU identity {summary_identity!r} does not "
                f"match pre-run identity {expected_identity!r}"
            )
        if not is_json_int(monitor.get("device_index")):
            errors.append(f"{context}: monitor device_index must be an integer")
    else:
        expected_index = expected_uuid = expected_name = None

    samples = monitor.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append(f"{context}: monitor must retain every raw sample")
        return
    if (
        not is_json_int(monitor.get("sample_count"))
        or monitor.get("sample_count") != len(samples)
    ):
        errors.append(f"{context}: monitor sample_count does not match samples")
    interval_seconds = monitor.get("interval_seconds")
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, (int, float))
        or not math.isfinite(float(interval_seconds))
        or float(interval_seconds) <= 0.0
        or interval_seconds != expected_interval_seconds
    ):
        errors.append(f"{context}: monitor interval does not match environment")

    counts = []
    distinct_pids = set()
    first_nvidia_smi_binary = (
        samples[0].get("nvidia_smi_binary")
        if isinstance(samples[0], dict)
        else None
    )
    available_nvidia_smi_binaries = []
    nvidia_smi_binary_validation_errors = []
    sample_validation_errors = []
    snapshot_validation_errors = []
    for sample_index, sample in enumerate(samples):
        sample_context = f"{context}.samples[{sample_index}]"
        if not isinstance(sample, dict):
            errors.append(f"{sample_context}: sample must be a JSON object")
            continue
        if sample.get("available") is not True or sample.get("error") is not None:
            errors.append(f"{sample_context}: nvidia-smi query was unavailable")
        else:
            available_nvidia_smi_binaries.append(
                sample.get("nvidia_smi_binary")
            )
        sample_snapshot_errors = gpu_compute_snapshot_errors(sample)
        snapshot_validation_errors.extend(
            f"samples[{sample_index}]: {error}"
            for error in sample_snapshot_errors
        )
        errors.extend(
            f"{sample_context}: {error}"
            for error in sample_snapshot_errors
        )
        if sample.get("boot_id") != expected_boot_id:
            errors.append(
                f"{sample_context}: boot ID does not match pre-run evidence"
            )
        if (
            not is_json_int(sample.get("device_index"))
            or sample.get("device_index") != 0
        ):
            sample_validation_errors.append(
                f"samples[{sample_index}]: device_index must be integer 0"
            )
        sample_binary_errors = trusted_nvidia_smi_metadata_errors(
            sample.get("nvidia_smi_binary")
        )
        nvidia_smi_binary_validation_errors.extend(
            f"samples[{sample_index}]: {error}"
            for error in sample_binary_errors
        )
        errors.extend(
            f"{sample_context}: {error}"
            for error in sample_binary_errors
        )
        if sample.get("nvidia_smi_binary") != expected_nvidia_smi_binary:
            errors.append(
                f"{sample_context}: nvidia-smi binary metadata does not "
                "exactly match pre-run evidence"
            )
        if expected_identity is not None:
            sample_identity = (
                sample.get("device_index"),
                sample.get("device_uuid"),
                sample.get("device_name"),
            )
            if sample_identity != expected_identity:
                errors.append(
                    f"{sample_context}: GPU identity {sample_identity!r} does "
                    f"not match pre-run identity {expected_identity!r}"
                )
        count = sample.get("process_count")
        processes = sample.get("processes")
        if not is_json_int(count) or count < 0:
            errors.append(f"{sample_context}: invalid process_count {count!r}")
            sample_validation_errors.append(
                f"samples[{sample_index}]: process_count is invalid"
            )
            continue
        if not isinstance(processes, list) or len(processes) != count:
            errors.append(
                f"{sample_context}: process list does not match process_count"
            )
            sample_validation_errors.append(
                f"samples[{sample_index}]: processes do not match process_count"
            )
            continue
        counts.append(count)
        for process_index, process in enumerate(processes):
            pid = process.get("host_pid") if isinstance(process, dict) else None
            used_memory = (
                process.get("used_memory_mib")
                if isinstance(process, dict)
                else None
            )
            if not isinstance(process, dict):
                sample_validation_errors.append(
                    f"samples[{sample_index}].processes[{process_index}]: "
                    "process must be an object"
                )
            if not is_json_int(pid) or pid <= 0:
                errors.append(f"{sample_context}: invalid host PID evidence")
                if isinstance(process, dict):
                    sample_validation_errors.append(
                        f"samples[{sample_index}].processes[{process_index}]: "
                        "host_pid is invalid"
                    )
            else:
                distinct_pids.add(pid)
            if not is_json_int(used_memory) or used_memory <= 0:
                errors.append(f"{sample_context}: invalid used-memory evidence")
                if isinstance(process, dict):
                    sample_validation_errors.append(
                        f"samples[{sample_index}].processes[{process_index}]: "
                        "used_memory_mib is invalid"
                    )

    incomplete_samples = len(counts) != len(samples)
    if incomplete_samples:
        errors.append(f"{context}: one or more monitor samples are incomplete")
    if (
        not is_json_int(monitor.get("available_sample_count"))
        or monitor.get("available_sample_count") != len(samples)
    ):
        errors.append(f"{context}: not every monitor sample was available")
    if (
        not is_json_int(monitor.get("unavailable_sample_count"))
        or monitor.get("unavailable_sample_count") != 0
    ):
        errors.append(f"{context}: unavailable monitor samples were recorded")
    if monitor.get("identity_consistent") is not True:
        errors.append(f"{context}: monitor GPU identity was not consistent")
    expected_min_process_count = min(counts, default=None)
    expected_max_process_count = max(counts, default=None)
    if (
        not is_json_int(monitor.get("min_process_count"))
        or monitor.get("min_process_count") != expected_min_process_count
    ):
        errors.append(f"{context}: min_process_count disagrees with raw samples")
    if (
        not is_json_int(monitor.get("max_process_count"))
        or monitor.get("max_process_count") != expected_max_process_count
    ):
        errors.append(f"{context}: max_process_count disagrees with raw samples")
    summary_pids = monitor.get("distinct_host_pids")
    if (
        not isinstance(summary_pids, list)
        or any(not is_json_int(pid) or pid <= 0 for pid in summary_pids)
        or summary_pids != sorted(distinct_pids)
    ):
        errors.append(f"{context}: distinct_host_pids disagrees with raw samples")
    if monitor.get("collection_errors") != []:
        errors.append(f"{context}: monitor recorded collection errors")
    if monitor.get("sample_validation_errors") != sample_validation_errors:
        errors.append(
            f"{context}: sample-validation summary disagrees with raw samples"
        )
    if monitor.get("sample_validation_errors") != []:
        errors.append(f"{context}: monitor samples contain validation errors")
    if monitor.get("snapshot_validation_errors") != snapshot_validation_errors:
        errors.append(
            f"{context}: raw-snapshot validation summary disagrees with samples"
        )
    if monitor.get("snapshot_validation_errors") != []:
        errors.append(f"{context}: monitor raw snapshots contain validation errors")
    sample_sequence_validation_errors = (
        gpu_compute_snapshot_sequence_errors(
            samples,
            float(expected_interval_seconds)
            + GPU_QUERY_CADENCE_TOLERANCE_SECONDS
            if isinstance(expected_interval_seconds, (int, float))
            and not isinstance(expected_interval_seconds, bool)
            and math.isfinite(float(expected_interval_seconds))
            else None,
        )
    )
    if (
        monitor.get("sample_sequence_validation_errors")
        != sample_sequence_validation_errors
    ):
        errors.append(
            f"{context}: snapshot-sequence summary disagrees with raw samples"
        )
    if monitor.get("sample_sequence_validation_errors") != []:
        errors.append(f"{context}: monitor snapshot sequence is invalid")
    observation_span_seconds = (
        gpu_compute_snapshot_observation_span_seconds(samples)
    )
    maximum_sample_gap_seconds = (
        gpu_compute_snapshot_maximum_gap_seconds(samples)
    )
    if (
        monitor.get("cadence_tolerance_seconds")
        != GPU_QUERY_CADENCE_TOLERANCE_SECONDS
        or monitor.get("maximum_sample_gap_seconds")
        != maximum_sample_gap_seconds
    ):
        errors.append(f"{context}: monitor cadence summary disagrees")
    if monitor.get("observation_span_seconds") != observation_span_seconds:
        errors.append(f"{context}: observation span disagrees with raw samples")
    if (
        isinstance(minimum_coverage_seconds, bool)
        or not isinstance(minimum_coverage_seconds, (int, float))
        or not math.isfinite(float(minimum_coverage_seconds))
        or float(minimum_coverage_seconds) <= 0.0
        or observation_span_seconds is None
        or observation_span_seconds < float(minimum_coverage_seconds)
    ):
        errors.append(
            f"{context}: GPU observations do not cover total generation time"
        )
    first_boot_id = (
        samples[0].get("boot_id")
        if isinstance(samples[0], dict)
        else None
    )
    boot_id_consistent = (
        bool(samples)
        and first_boot_id is not None
        and all(
            isinstance(sample, dict)
            and sample.get("boot_id") == first_boot_id
            for sample in samples
        )
    )
    if monitor.get("boot_id") != first_boot_id:
        errors.append(f"{context}: boot ID summary differs from first sample")
    if monitor.get("boot_id") != expected_boot_id:
        errors.append(f"{context}: boot ID summary differs from pre-run evidence")
    if monitor.get("boot_id_consistent") is not boot_id_consistent:
        errors.append(f"{context}: boot ID consistency summary disagrees")
    if monitor.get("boot_id_consistent") is not True:
        errors.append(f"{context}: monitor samples cross boot boundaries")

    nvidia_smi_binary_fixed_valid = (
        bool(available_nvidia_smi_binaries)
        and not nvidia_smi_binary_validation_errors
    )
    nvidia_smi_binary_consistent = (
        bool(available_nvidia_smi_binaries)
        and first_nvidia_smi_binary is not None
        and all(
            metadata == first_nvidia_smi_binary
            for metadata in available_nvidia_smi_binaries
        )
    )
    if monitor.get("nvidia_smi_binary") != first_nvidia_smi_binary:
        errors.append(
            f"{context}: nvidia-smi binary summary differs from first sample"
        )
    if monitor.get("nvidia_smi_binary") != expected_nvidia_smi_binary:
        errors.append(
            f"{context}: nvidia-smi binary summary does not exactly match "
            "pre-run evidence"
        )
    if (
        monitor.get("nvidia_smi_binary_fixed_valid")
        is not nvidia_smi_binary_fixed_valid
    ):
        errors.append(
            f"{context}: nvidia-smi fixed-metadata summary disagrees with samples"
        )
    if monitor.get("nvidia_smi_binary_fixed_valid") is not True:
        errors.append(
            f"{context}: nvidia-smi binary metadata is not fixed-valid"
        )
    if (
        monitor.get("nvidia_smi_binary_consistent")
        is not nvidia_smi_binary_consistent
    ):
        errors.append(
            f"{context}: nvidia-smi binary-consistency summary disagrees with samples"
        )
    if monitor.get("nvidia_smi_binary_consistent") is not True:
        errors.append(
            f"{context}: nvidia-smi binary metadata changed between samples"
        )
    if (
        monitor.get("nvidia_smi_binary_validation_errors")
        != nvidia_smi_binary_validation_errors
    ):
        errors.append(
            f"{context}: nvidia-smi binary validation-errors summary "
            "disagrees with samples"
        )

    exact_singleton = (
        not incomplete_samples
        and not sample_validation_errors
        and not snapshot_validation_errors
        and not sample_sequence_validation_errors
        and boot_id_consistent
        and all(count == 1 for count in counts)
    )
    single_distinct_pid = exact_singleton and len(distinct_pids) == 1
    contention_detected = any(count > 1 for count in counts)
    no_process_detected = any(count == 0 for count in counts)
    if monitor.get("exact_singleton_process_per_sample") is not exact_singleton:
        errors.append(f"{context}: exact-singleton summary disagrees with samples")
    if monitor.get("single_distinct_host_pid") is not single_distinct_pid:
        errors.append(f"{context}: single-distinct-PID summary disagrees with samples")
    if monitor.get("contention_detected") is not contention_detected:
        errors.append(f"{context}: contention summary disagrees with samples")
    if monitor.get("no_process_detected") is not no_process_detected:
        errors.append(f"{context}: no-process summary disagrees with samples")
    if candidate:
        if len(samples) < 2:
            errors.append(
                f"{context}: benchmark monitor requires at least entry and exit samples"
            )
        if not exact_singleton:
            errors.append(
                f"{context}: benchmark samples must each contain exactly one "
                "compute process"
            )
        if not single_distinct_pid:
            errors.append(
                f"{context}: benchmark generation changed compute-process PID"
            )
        if monitor.get("valid_for_benchmark") is not True:
            errors.append(f"{context}: monitor is not valid for benchmark use")


def validate_sparge_dispatcher(
    dispatcher,
    errors,
    *,
    expected_receipt=None,
    expected_settings=None,
    expected_gpu_uuid=None,
    context="metrics",
):
    """Close the formal provenance loop for an official SpargeAttn record."""

    details = dispatcher.get("backend_details")
    if not isinstance(details, dict):
        errors.append(f"{context}: Sparge dispatcher is missing backend_details")
        return
    for field, expected in SPARGE_PROVENANCE.items():
        if details.get(field) != expected:
            errors.append(
                f"{context}: Sparge backend_details {field}="
                f"{details.get(field)!r} != {expected!r}"
            )
    if details.get("calls") != dispatcher.get("calls_total"):
        errors.append(
            f"{context}: Sparge backend calls={details.get('calls')} != "
            f"dispatcher calls_total={dispatcher.get('calls_total')}"
        )
    expected_calls_by_method = {
        "dense": 0,
        "sparge": dispatcher.get("calls_total"),
        "radial": 0,
        "svg": 0,
    }
    if dispatcher.get("calls_by_method") != expected_calls_by_method:
        errors.append(
            f"{context}: Sparge calls_by_method="
            f"{dispatcher.get('calls_by_method')!r} != "
            f"{expected_calls_by_method!r}"
        )
    if details.get("last_nhd_shape") != [1, 15004, 24, 128]:
        errors.append(
            f"{context}: Sparge last_nhd_shape="
            f"{details.get('last_nhd_shape')!r} != [1, 15004, 24, 128]"
        )
    if details.get("last_dtype") != "torch.bfloat16":
        errors.append(
            f"{context}: Sparge last_dtype={details.get('last_dtype')!r} "
            "!= 'torch.bfloat16'"
        )
    if details.get("last_device") != "cuda:0":
        errors.append(
            f"{context}: Sparge last_device={details.get('last_device')!r} "
            "!= 'cuda:0'"
        )

    receipt = details.get("install_receipt")
    if not isinstance(receipt, dict):
        errors.append(f"{context}: Sparge backend install_receipt is missing")
    else:
        for error in sparge_receipt_evidence_errors(
            receipt, expected_gpu_uuid=expected_gpu_uuid
        ):
            errors.append(f"{context}: Sparge backend receipt: {error}")
        if expected_receipt is not None and receipt != expected_receipt:
            errors.append(
                f"{context}: Sparge backend receipt differs from copied run receipt"
            )

    if expected_settings is not None:
        for field, expected in expected_settings.items():
            if details.get(field) != expected:
                errors.append(
                    f"{context}: Sparge setting {field}={details.get(field)!r} "
                    f"!= environment {expected!r}"
                )


def validate_radial_dispatcher(
    dispatcher,
    errors,
    *,
    expected_receipt=None,
    expected_settings=None,
    context="metrics",
):
    """Require real Radial calls, exact tail handling, and audited mask data."""

    details = dispatcher.get("backend_details")
    if not isinstance(details, dict):
        errors.append(f"{context}: Radial dispatcher is missing backend_details")
        return
    expected_provenance = {
        "backend": "official_radial_attention_flashinfer",
        "repository": RADIAL_REPOSITORY,
        "pinned_commit": RADIAL_COMMIT,
        "mask_api": RADIAL_MASK_API,
        "model_type": RADIAL_MODEL_TYPE,
        "block_size": RADIAL_BLOCK_SIZE,
        "sequence": RADIAL_SEQUENCE,
        "prefix_sequence": RADIAL_PREFIX_SEQUENCE,
        "tail_sequence": RADIAL_TAIL_SEQUENCE,
        "tail_strategy": "dense_lse_merge_no_padding",
        "empty_row_policy": "dense_row",
        "empty_rows": list(RADIAL_EMPTY_ROWS),
        "fallback_allowed": False,
    }
    for field, expected in expected_provenance.items():
        if not strict_json_equal(details.get(field), expected):
            errors.append(
                f"{context}: Radial backend_details {field}="
                f"{details.get(field)!r} != {expected!r}"
            )
    calls = dispatcher.get("calls_total")
    if not is_json_int(calls) or calls <= 0:
        errors.append(
            f"{context}: Radial dispatcher calls_total must be a positive "
            "JSON integer"
        )
    backend_calls = details.get("calls")
    if not is_json_int(backend_calls) or backend_calls <= 0:
        errors.append(
            f"{context}: Radial backend calls must be a positive JSON integer"
        )
    elif not is_json_int(calls) or backend_calls != calls:
        errors.append(
            f"{context}: Radial backend calls={backend_calls} != "
            f"dispatcher calls_total={calls}"
        )
    expected_calls_by_method = {
        "dense": 0,
        "sparge": 0,
        "radial": calls,
        "svg": 0,
    }
    calls_by_method = dispatcher.get("calls_by_method")
    calls_by_method_types_valid = (
        isinstance(calls_by_method, dict)
        and set(calls_by_method) == set(expected_calls_by_method)
        and all(
            is_json_int(value) and value >= 0
            for value in calls_by_method.values()
        )
    )
    if not calls_by_method_types_valid:
        errors.append(
            f"{context}: Radial calls_by_method must contain non-negative "
            "JSON integer counters"
        )
    elif not strict_json_equal(calls_by_method, expected_calls_by_method):
        errors.append(
            f"{context}: Radial calls_by_method="
            f"{calls_by_method!r} != "
            f"{expected_calls_by_method!r}"
        )
    if not strict_json_equal(details.get("last_shape"), [1, 15004, 24, 128]):
        errors.append(f"{context}: Radial last_shape is not fixed Ovi NHD")
    if not strict_json_equal(details.get("last_grid"), list(RADIAL_GRID)):
        errors.append(f"{context}: Radial last_grid is not [31, 22, 22]")
    if details.get("last_dtype") != "torch.bfloat16":
        errors.append(f"{context}: Radial last_dtype is not torch.bfloat16")
    if details.get("last_device") != "cuda:0":
        errors.append(f"{context}: Radial last_device is not cuda:0")
    plan_cache_entries = details.get("plan_cache_entries")
    if not is_json_int(plan_cache_entries) or plan_cache_entries != 1:
        errors.append(f"{context}: Radial must use exactly one keyed plan")
    misses = details.get("plan_cache_misses")
    hits = details.get("plan_cache_hits")
    if (
        not is_json_int(misses)
        or misses not in (0, 1)
        or not is_json_int(hits)
        or hits < 0
    ):
        errors.append(f"{context}: Radial plan-cache counters are invalid")
    elif not is_json_int(calls) or hits + misses != calls:
        errors.append(
            f"{context}: Radial plan cache hits+misses != backend calls"
        )

    profile = details.get("profile")
    expected_audit = (
        RADIAL_PROFILE_AUDITS.get(profile)
        if isinstance(profile, str)
        else None
    )
    observed_audit = details.get("last_mask_audit")
    if expected_audit is None:
        errors.append(f"{context}: unknown Radial profile {profile!r}")
    elif not strict_json_equal(observed_audit, expected_audit):
        errors.append(
            f"{context}: Radial mask audit differs from fixed {profile} audit"
        )

    receipt_summary = details.get("install_receipt")
    if not isinstance(receipt_summary, dict):
        errors.append(f"{context}: Radial backend receipt summary is missing")
    else:
        receipt_runtime = receipt_summary.get("runtime_dependencies")
        validate_radial_runtime_dependency_evidence(
            receipt_runtime,
            receipt_runtime,
            f"{context}: Radial backend receipt runtime dependencies",
            errors,
        )
        runtime_after_cuda = details.get(
            "runtime_dependencies_after_first_cuda"
        )
        validate_radial_runtime_dependency_evidence(
            runtime_after_cuda,
            receipt_runtime,
            f"{context}: Radial runtime dependencies after first CUDA call",
            errors,
        )
    if isinstance(receipt_summary, dict) and expected_receipt is not None:
        expected_derived_module = expected_receipt.get("derived_module")
        expected_summary = {
            "path": str(Path(expected_receipt["_copied_path"]).resolve()),
            "commit": expected_receipt.get("commit"),
            "derived_module_sha256": (
                expected_derived_module.get("sha256")
                if isinstance(expected_derived_module, dict)
                else None
            ),
            "flashinfer_version": expected_receipt.get("flashinfer_version"),
            "runtime_dependencies": (
                expected_radial_runtime_dependency_evidence(expected_receipt)
            ),
        }
        # The backend reads the cache receipt rather than the copied evidence;
        # bind immutable contents but allow its original cache path.
        expected_summary["path"] = expected_receipt.get("_original_path")
        if not strict_json_equal(receipt_summary, expected_summary):
            errors.append(
                f"{context}: Radial backend receipt summary differs from run evidence"
            )

    if expected_settings is not None:
        for field, expected in expected_settings.items():
            if not strict_json_equal(details.get(field), expected):
                errors.append(
                    f"{context}: Radial setting {field}="
                    f"{details.get(field)!r} != environment {expected!r}"
                )


def verify(
    path,
    require_metrics=True,
    expected_video_frames=121,
    run_dir=None,
    evidence_snapshots=None,
    metrics_payloads=None,
):
    run_dir = Path(run_dir or Path(path).parent).resolve(strict=True)
    path = _canonical_leaf_path(path)
    if path.parent != run_dir:
        raise EvidenceSnapshotError(
            f"artifact must be a direct child of run directory {run_dir}: {path}"
        )
    artifact_snapshot = _stable_file_snapshot(path)
    artifact_sha256 = artifact_snapshot.sha256
    materialized_path = _materialize_snapshot(artifact_snapshot, ".mp4")
    try:
        info = probe(materialized_path)
        media_videos = [
            stream
            for stream in info.get("streams", [])
            if stream.get("codec_type") == "video"
        ]
        media_audios = [
            stream
            for stream in info.get("streams", [])
            if stream.get("codec_type") == "audio"
        ]
        samples = (
            decode_audio(materialized_path)
            if media_audios
            else np.empty(0, dtype=np.float32)
        )
        gray = (
            decode_video_gray(materialized_path, expected_video_frames)
            if media_videos
            else np.empty(0, dtype=np.uint8)
        )
    finally:
        try:
            materialized_path.unlink()
        except FileNotFoundError:
            pass
    videos = media_videos
    audios = media_audios
    errors = []
    if len(videos) != 1:
        errors.append(f"expected exactly one video stream, found {len(videos)}")
    if len(audios) != 1:
        errors.append(f"expected exactly one audio stream, found {len(audios)}")

    video = videos[0] if videos else {}
    width = as_int(video.get("width"))
    height = as_int(video.get("height"))
    frames = as_int(video.get("nb_read_frames")) or as_int(video.get("nb_frames"))
    duration = float(info.get("format", {}).get("duration") or 0.0)
    if width is None or height is None or width <= 0 or height <= 0:
        errors.append(f"invalid video dimensions: {width}x{height}")
    elif width % 32 or height % 32:
        errors.append(f"video dimensions are not multiples of 32: {width}x{height}")
    if frames != expected_video_frames:
        errors.append(
            f"expected {expected_video_frames} video frames, found {frames}"
        )
    if not 4.5 <= duration <= 5.5:
        errors.append(f"expected about 5 seconds, found {duration:.6f}")

    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    active_ratio = float(np.mean(np.abs(samples) > 1e-3)) if samples.size else 0.0
    dbfs = float(20.0 * math.log10(max(rms, 1e-12)))
    if samples.size < 4 * 16000:
        errors.append(f"decoded audio is too short: {samples.size} samples")
    if not np.isfinite(samples).all():
        errors.append("decoded audio contains NaN or Inf")
    if not math.isfinite(rms) or rms <= 1e-3:
        errors.append(f"audio RMS is silent/invalid: {rms}")
    if not math.isfinite(peak) or peak <= 1e-2:
        errors.append(f"audio peak is silent/invalid: {peak}")
    if active_ratio <= 0.01:
        errors.append(f"audio active-sample ratio is too low: {active_ratio}")

    video_std = float(gray.std()) if gray.size else 0.0
    if gray.size == 0 or video_std <= 2.0:
        errors.append(f"decoded video is blank or nearly constant: std={video_std}")

    metrics_path = path.with_suffix(".metrics.json")
    if metrics_path.parent != run_dir:
        raise EvidenceSnapshotError(
            f"metrics must be a direct child of run directory {run_dir}: {metrics_path}"
        )
    metrics_snapshot = None
    metrics = None
    if os.path.lexists(metrics_path):
        metrics_snapshot, metrics = _snapshot_json(metrics_path)
        if metrics_payloads is not None:
            metrics_payloads[str(path)] = metrics
    if metrics is None and require_metrics:
        errors.append(f"missing metrics sidecar: {metrics_path}")
    elif metrics is not None:
        required_fields = (
            "status",
            "record_type",
            "denoise_seconds",
            "total_generation_seconds",
            "peak_memory_allocated_bytes",
            "peak_memory_reserved_bytes",
            "generated_video_shape",
            "generated_audio_shape",
            "actual_video_frame_height_width",
            "output_sha256",
            "save_video_seconds",
            "artifact_ready_seconds",
            "output_hash_seconds",
            "measurement_index",
            "prompt_index",
            "sample_index",
            "prompt",
            "seed",
            "output_path",
            "benchmark_candidate",
            "benchmark_valid",
            "attention_method",
            "use_cfg_cache",
            "cfg_cache_hits",
            "cfg_cache_refreshes",
            "cfg_negative_forwards",
            "expected_cfg_cache_metrics",
            "use_block_cache",
            "video_self_attention_dispatcher",
            "gpu_process_monitor",
        )
        missing = [field for field in required_fields if field not in metrics]
        if missing:
            errors.append(f"metrics sidecar missing required fields: {missing}")
        if metrics.get("status") != "ok" or metrics.get("record_type") != "measurement":
            errors.append(
                f"invalid metrics status/type: {metrics.get('status')}/{metrics.get('record_type')}"
            )
        if metrics.get("benchmark_valid") is not False:
            errors.append("per-artifact benchmark_valid must remain false until run verification")
        for field in ("benchmark_candidate", "use_cfg_cache", "use_block_cache"):
            if type(metrics.get(field)) is not bool:
                errors.append(f"metrics {field} must be a JSON boolean")
        for field in ("measurement_index", "prompt_index", "sample_index"):
            if not is_json_int(metrics.get(field)) or metrics.get(field) < 0:
                errors.append(f"metrics {field} must be a nonnegative JSON integer")
        if not isinstance(metrics.get("prompt"), str) or not metrics.get("prompt"):
            errors.append("metrics prompt must be a non-empty string")
        if not is_json_int(metrics.get("seed")):
            errors.append("metrics seed must be a JSON integer")
        actual_hw = metrics.get("actual_video_frame_height_width")
        generated_shape = metrics.get("generated_video_shape")
        if actual_hw != [height, width]:
            errors.append(f"metrics actual size {actual_hw} != stream size {[height, width]}")
        if generated_shape and generated_shape[1:] != [frames, height, width]:
            errors.append(
                f"metrics generated shape {generated_shape} != stream shape "
                f"[channels,{frames},{height},{width}]"
            )
        if metrics.get("output_sha256") != artifact_sha256:
            errors.append(
                f"output SHA256 mismatch: metrics={metrics.get('output_sha256')} actual={artifact_sha256}"
            )
        try:
            recorded_output_path = _canonical_leaf_path(metrics.get("output_path", ""))
        except (OSError, TypeError, ValueError):
            recorded_output_path = None
        if recorded_output_path != path:
            errors.append(f"metrics output_path does not match artifact: {metrics.get('output_path')}")

        expected_cfg = metrics.get("expected_cfg_cache_metrics")
        if isinstance(expected_cfg, dict):
            for field in (
                "cfg_cache_hits",
                "cfg_cache_refreshes",
                "cfg_negative_forwards",
            ):
                if metrics.get(field) != expected_cfg.get(field):
                    errors.append(
                        f"{field}={metrics.get(field)} != expected "
                        f"{expected_cfg.get(field)}"
                    )
        elif expected_cfg is not None:
            errors.append("expected_cfg_cache_metrics must be a JSON object")

        block_cache_enabled = metrics.get("use_block_cache") is True
        block_metric_fields = (
            "block_cache_start_block",
            "block_cache_end_block",
            "block_cache_window_inclusive",
            "block_cache_policy",
            "block_cache_cosine_threshold",
            "block_cache_max_consecutive_reuses",
            "block_cache_hits",
            "block_cache_refreshes",
            "block_cache_saved_video_self_attention_calls",
            "block_cache_branch_metrics",
        )
        block_metrics_present = any(
            field in metrics for field in block_metric_fields
        )
        block_hits = 0
        block_refreshes = 0
        block_saved_calls = 0
        if block_cache_enabled or block_metrics_present:
            missing_block_fields = [
                field for field in block_metric_fields if field not in metrics
            ]
            if missing_block_fields:
                errors.append(
                    "block-cache metrics missing required fields: "
                    f"{missing_block_fields}"
                )
            block_hits = as_int(metrics.get("block_cache_hits"))
            block_refreshes = as_int(metrics.get("block_cache_refreshes"))
            block_saved_calls = as_int(
                metrics.get("block_cache_saved_video_self_attention_calls")
            )
            block_branches = metrics.get("block_cache_branch_metrics")
            if metrics.get("block_cache_policy") not in ("fixed", "cosine"):
                errors.append("invalid block_cache_policy")
            try:
                block_cosine_threshold = float(
                    metrics.get("block_cache_cosine_threshold")
                )
            except (TypeError, ValueError):
                block_cosine_threshold = float("nan")
            if (
                not math.isfinite(block_cosine_threshold)
                or not 0.0 <= block_cosine_threshold <= 1.0
            ):
                errors.append("invalid block_cache_cosine_threshold")
            if metrics.get("block_cache_max_consecutive_reuses") != 1:
                errors.append(
                    "block cache must cap consecutive reuses at exactly 1"
                )
            block_start = as_int(metrics.get("block_cache_start_block"))
            block_end = as_int(metrics.get("block_cache_end_block"))
            if (
                block_start is None
                or block_end is None
                or not 0 <= block_start <= block_end
            ):
                errors.append(
                    f"invalid block-cache window: {block_start}..{block_end}"
                )
            if metrics.get("block_cache_window_inclusive") is not True:
                errors.append("block-cache window must be recorded as inclusive")
            if not block_cache_enabled:
                if (block_hits, block_refreshes, block_saved_calls) != (0, 0, 0):
                    errors.append(
                        "disabled block cache recorded non-zero activity"
                    )
                if block_branches not in ({}, None):
                    errors.append(
                        "disabled block cache recorded branch payload metrics"
                    )
            elif isinstance(block_branches, dict):
                expected_branches = {"conditional", "unconditional"}
                if set(block_branches) != expected_branches:
                    errors.append(
                        f"block-cache branches {sorted(block_branches)} != "
                        f"{sorted(expected_branches)}"
                    )
                branch_hits = sum(
                    as_int(item.get("hits")) or 0
                    for item in block_branches.values()
                    if isinstance(item, dict)
                )
                branch_refreshes = sum(
                    as_int(item.get("refreshes")) or 0
                    for item in block_branches.values()
                    if isinstance(item, dict)
                )
                branch_saved_calls = sum(
                    as_int(item.get("saved_video_self_attention_calls")) or 0
                    for item in block_branches.values()
                    if isinstance(item, dict)
                )
                if branch_hits != block_hits:
                    errors.append(
                        f"block_cache_hits={block_hits} != branch sum "
                        f"{branch_hits}"
                    )
                if branch_refreshes != block_refreshes:
                    errors.append(
                        "block_cache_refreshes="
                        f"{block_refreshes} != branch sum {branch_refreshes}"
                    )
                if branch_saved_calls != block_saved_calls:
                    errors.append(
                        "block_cache_saved_video_self_attention_calls="
                        f"{block_saved_calls} != branch sum {branch_saved_calls}"
                    )
                if metrics.get("block_cache_policy") == "fixed":
                    errors.extend(
                        f"fixed block-cache schedule: {error}"
                        for error in fixed_block_cache_metric_errors(metrics)
                    )
            else:
                errors.append("enabled block cache requires branch metrics")

        dispatcher = metrics.get("video_self_attention_dispatcher")
        if isinstance(dispatcher, dict):
            configured_method = metrics.get("attention_method")
            if dispatcher.get("configured_method") != configured_method:
                errors.append(
                    "dispatcher configured_method disagrees with attention_method"
                )
            if dispatcher.get("active_method") != configured_method:
                errors.append("dispatcher active_method disagrees with attention_method")
            if dispatcher.get("fallback_allowed") is not False:
                errors.append("dispatcher must not allow fallback")
            if dispatcher.get("fallback_used") is not False:
                errors.append("dispatcher unexpectedly used fallback")
            if dispatcher.get("fallback_count") != 0:
                errors.append("dispatcher fallback_count must be zero")
            if dispatcher.get("calls_total") != dispatcher.get("expected_calls"):
                errors.append(
                    f"dispatcher calls_total={dispatcher.get('calls_total')} != "
                    f"expected_calls={dispatcher.get('expected_calls')}"
                )
            expected_without_block_cache = dispatcher.get(
                "expected_calls_without_block_cache"
            )
            if expected_without_block_cache is not None:
                adjusted_expected = (
                    as_int(expected_without_block_cache) or 0
                ) - (block_saved_calls or 0)
                if dispatcher.get("expected_calls") != adjusted_expected:
                    errors.append(
                        "dispatcher expected_calls does not subtract the "
                        "recorded block-cache savings"
                    )
            if dispatcher.get("calls_match_expected") is not True:
                errors.append("dispatcher calls_match_expected must be true")
            errors_by_method = dispatcher.get("errors_by_method", {})
            if any(value for value in errors_by_method.values()):
                errors.append(
                    f"dispatcher recorded backend errors: {errors_by_method}"
                )
            if configured_method == "sparge":
                validate_sparge_dispatcher(dispatcher, errors)
            elif configured_method == "radial":
                validate_radial_dispatcher(dispatcher, errors)
        elif dispatcher is not None:
            errors.append("video_self_attention_dispatcher must be a JSON object")

        gpu_monitor = metrics.get("gpu_process_monitor")
        if gpu_monitor is not None and not isinstance(gpu_monitor, dict):
            errors.append("gpu_process_monitor must be a JSON object")

    snapshots = [artifact_snapshot]
    if metrics_snapshot is not None:
        snapshots.append(metrics_snapshot)
    for snapshot in snapshots:
        try:
            _revalidate_snapshot(snapshot)
        except EvidenceSnapshotError as exc:
            errors.append(f"stable evidence snapshot failed: {exc}")
    if evidence_snapshots is not None:
        evidence_snapshots.extend(snapshots)

    return {
        "path": str(path),
        "sha256": artifact_sha256,
        "measurement_index": metrics.get("measurement_index") if metrics else None,
        "prompt_index": metrics.get("prompt_index") if metrics else None,
        "sample_index": metrics.get("sample_index") if metrics else None,
        "prompt": metrics.get("prompt") if metrics else None,
        "seed": metrics.get("seed") if metrics else None,
        "metrics_path": str(metrics_path),
        "artifact_binding": _file_binding(artifact_snapshot),
        "metrics_binding": (
            _file_binding(metrics_snapshot) if metrics_snapshot is not None else None
        ),
        "status": "failed" if errors else "ok",
        "errors": errors,
        "video": {
            "codec": video.get("codec_name"),
            "width": width,
            "height": height,
            "frames": frames,
            "duration_seconds": duration,
            "decoded_frames": int(gray.size // (64 * 64)),
            "decoded_raw_bytes": int(gray.nbytes),
            "decoded_raw_sha256": hashlib.sha256(gray).hexdigest(),
            "decoded_pixel_std": video_std,
        },
        "audio": {
            "codec": audios[0].get("codec_name") if audios else None,
            "decoded_samples_16khz_mono": int(samples.size),
            "rms": rms,
            "peak": peak,
            "dbfs": dbfs,
            "active_sample_ratio_abs_gt_1e-3": active_ratio,
        },
    }


def artifact_report_protocol_errors(
    reports, timings, run_dir, metrics_payloads=None
):
    """Bind every verified artifact/sidecar to exactly one timing record."""

    run_dir = Path(run_dir).resolve(strict=True)
    errors = []
    timing_by_path = {}
    timing_by_tuple = {}
    identity_fields = (
        "measurement_index",
        "prompt_index",
        "sample_index",
        "prompt",
        "seed",
    )

    for index, timing in enumerate(timings):
        context = f"timing[{index}]"
        if not isinstance(timing, dict):
            errors.append(f"{context} must be a JSON object")
            continue
        raw_path = timing.get("output_path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"{context} output_path must be a non-empty string")
            continue
        try:
            canonical_path = _canonical_leaf_path(raw_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{context} output_path cannot be canonicalized: {exc}")
            continue
        if not Path(raw_path).is_absolute() or raw_path != str(canonical_path):
            errors.append(f"{context} output_path is not canonical: {raw_path!r}")
        if canonical_path.parent != run_dir:
            errors.append(f"{context} output_path is outside run directory")
        path_key = str(canonical_path)
        if path_key in timing_by_path:
            errors.append(f"duplicate timing output_path: {path_key}")
        else:
            timing_by_path[path_key] = timing
        tuple_key = tuple(timing.get(field) for field in identity_fields[:3])
        if not all(is_json_int(value) for value in tuple_key):
            errors.append(f"{context} measurement/prompt/sample indexes must be JSON integers")
        elif tuple_key in timing_by_tuple:
            errors.append(f"duplicate timing measurement/prompt/sample tuple: {tuple_key}")
        else:
            timing_by_tuple[tuple_key] = timing

    report_by_path = {}
    report_by_tuple = {}
    for index, report in enumerate(reports):
        context = f"artifact[{index}]"
        if not isinstance(report, dict):
            errors.append(f"{context} must be a JSON object")
            continue
        raw_path = report.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"{context} path must be a non-empty string")
            continue
        try:
            canonical_path = _canonical_leaf_path(raw_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{context} path cannot be canonicalized: {exc}")
            continue
        path_key = str(canonical_path)
        if not Path(raw_path).is_absolute() or raw_path != path_key:
            errors.append(f"{context} path is not canonical: {raw_path!r}")
        if canonical_path.parent != run_dir:
            errors.append(f"{context} path is outside run directory")
        if path_key in report_by_path:
            errors.append(f"duplicate artifact report path: {path_key}")
        else:
            report_by_path[path_key] = report

        expected_metrics_path = canonical_path.with_suffix(".metrics.json")
        raw_metrics_path = report.get("metrics_path")
        if (
            not isinstance(raw_metrics_path, str)
            or raw_metrics_path != str(expected_metrics_path)
        ):
            errors.append(
                f"{context} metrics_path does not match canonical sidecar path"
            )

        artifact_binding = report.get("artifact_binding")
        errors.extend(
            _binding_shape_errors(
                artifact_binding, {"path", "bytes", "sha256"},
                f"{context} artifact_binding",
            )
        )
        metrics_binding = report.get("metrics_binding")
        errors.extend(
            _binding_shape_errors(
                metrics_binding, {"path", "bytes", "sha256"},
                f"{context} metrics_binding",
            )
        )
        if isinstance(artifact_binding, dict):
            if artifact_binding.get("path") != path_key:
                errors.append(f"{context} artifact_binding path mismatch")
            if not strict_json_equal(
                artifact_binding.get("sha256"), report.get("sha256")
            ):
                errors.append(f"{context} artifact SHA256 fields disagree")
        if isinstance(metrics_binding, dict) and (
            metrics_binding.get("path") != str(expected_metrics_path)
        ):
            errors.append(f"{context} metrics_binding path mismatch")

        tuple_values = tuple(report.get(field) for field in identity_fields[:3])
        if not all(is_json_int(value) for value in tuple_values):
            errors.append(
                f"{context} measurement/prompt/sample indexes must be JSON integers"
            )
        elif tuple_values in report_by_tuple:
            errors.append(
                f"duplicate artifact measurement/prompt/sample tuple: {tuple_values}"
            )
        else:
            report_by_tuple[tuple_values] = report

        timing = timing_by_path.get(path_key)
        if timing is None:
            errors.append(f"{context} has no timing record at the same canonical path")
            continue
        for field in identity_fields:
            if not strict_json_equal(report.get(field), timing.get(field)):
                errors.append(f"{context} {field} does not match timing record")
        if isinstance(artifact_binding, dict) and not strict_json_equal(
            artifact_binding.get("sha256"), timing.get("output_sha256")
        ):
            errors.append(
                f"{context} artifact hash does not match same-path timing record"
            )
        metrics_payload = (
            metrics_payloads.get(path_key)
            if isinstance(metrics_payloads, dict)
            else None
        )
        if not isinstance(metrics_payload, dict):
            errors.append(
                f"{context} parsed metrics sidecar payload is unavailable"
            )
        elif not strict_json_equal(metrics_payload, timing):
            errors.append(
                f"{context} metrics sidecar differs from same-path timing record"
            )

    if set(report_by_path) != set(timing_by_path):
        errors.append("artifact and timing canonical path sets differ")
    if set(report_by_tuple) != set(timing_by_tuple):
        errors.append("artifact and timing measurement/prompt/sample tuple sets differ")
    return errors


def verify_run_protocol(
    run_dir,
    reports,
    evidence_snapshots=None,
    metrics_payloads=None,
):
    run_dir = Path(run_dir).resolve(strict=True)
    errors = []
    radial_claims = None
    evidence_cache = {}
    parsed_json_cache = {}
    protocol_guards = []

    def capture_evidence(path, context):
        try:
            canonical_path = _canonical_leaf_path(path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"invalid run evidence path {context}: {exc}")
            return None
        if canonical_path.parent != run_dir:
            errors.append(f"run evidence path escaped run directory: {context}")
            return None
        key = str(canonical_path)
        if key in evidence_cache:
            return evidence_cache[key]
        try:
            snapshot = _stable_file_snapshot(canonical_path)
        except EvidenceSnapshotError as exc:
            evidence_cache[key] = None
            errors.append(f"invalid run evidence file {context}: {exc}")
            return None
        evidence_cache[key] = snapshot
        protocol_guards.append(snapshot)
        if evidence_snapshots is not None:
            evidence_snapshots.append(snapshot)
        return snapshot

    def parse_evidence_json(path, context):
        canonical_path = _canonical_leaf_path(path)
        key = str(canonical_path)
        if key in parsed_json_cache:
            return parsed_json_cache[key]
        snapshot = capture_evidence(canonical_path, context)
        if snapshot is None:
            parsed_json_cache[key] = None
            return None
        try:
            payload = _json_from_snapshot(snapshot, context)
        except EvidenceSnapshotError as exc:
            errors.append(str(exc))
            payload = None
        parsed_json_cache[key] = payload
        return payload

    environment_path = run_dir / "environment.json"
    environment = parse_evidence_json(environment_path, "environment.json") or {}
    validate_run_protocol(environment, errors)
    attention_method = environment.get("attention_method")
    required_files = [
        "environment.json",
        "run_config.yaml",
        "pre_run_gpu.json",
        "preflight.json",
        "environment.freeze.txt",
        "checkpoint_manifest.json",
    ]
    if attention_method == "sparge":
        required_files.extend(
            (
                "spargeattn-install.json",
                "spargeattn-build.log",
                "spargeattn-install-pre_run_gpu.json",
            )
        )
    elif attention_method == "radial":
        required_files.extend(
            (
                "radialattn-install.json",
                "radial-flashinfer-manifest.json",
                "radial-attention-source.py",
                "radial-attention-derived.py",
                "radial-attention-optional-imports.patch",
            )
        )
    required_snapshots = {
        filename: capture_evidence(run_dir / filename, filename)
        for filename in required_files
    }

    pre_run_gpu_path = run_dir / "pre_run_gpu.json"
    pre_run_gpu = parse_evidence_json(pre_run_gpu_path, "pre_run_gpu.json")
    expected_gpu_identity = validate_pre_run_gpu(
        pre_run_gpu, environment, errors
    )

    if type(environment.get("benchmark_eligible")) is not bool:
        errors.append("environment benchmark_eligible must be a JSON boolean")
    candidate = environment.get("benchmark_eligible") is True
    for field in ("debug_forward", "git_dirty", "use_cfg_cache", "use_block_cache"):
        if type(environment.get(field)) is not bool:
            errors.append(f"environment {field} must be a JSON boolean")
    expected_measurements = _nonnegative_json_int(
        environment.get("expected_measurement_records")
    )
    expected_warmups = _nonnegative_json_int(
        environment.get("expected_warmup_records")
    )
    measurement_runs = _nonnegative_json_int(environment.get("measurement_runs"))

    timings_path = run_dir / "timings.jsonl"
    timings_snapshot = None
    timings_binding = None
    try:
        timings_snapshot, timings = _snapshot_jsonl(timings_path)
        timings_binding = _jsonl_binding(timings_snapshot, timings)
        if evidence_snapshots is not None:
            evidence_snapshots.append(timings_snapshot)
    except EvidenceSnapshotError as exc:
        timings = []
        errors.append(f"invalid timings.jsonl evidence: {exc}")
    warmup_timings_path = run_dir / "warmup_timings.jsonl"
    warmup_snapshot = None
    if expected_warmups == 0:
        warmups = []
        warmup_binding = None
        warmup_absence_guard = _AbsentPathGuard(warmup_timings_path)
        protocol_guards.append(warmup_absence_guard)
        if evidence_snapshots is not None:
            evidence_snapshots.append(warmup_absence_guard)
        if os.path.lexists(warmup_timings_path):
            errors.append(
                "zero-warmup protocol must not create warmup_timings.jsonl"
            )
    elif os.path.lexists(warmup_timings_path):
        try:
            warmup_snapshot, warmups = _snapshot_jsonl(warmup_timings_path)
            warmup_binding = _warmup_timings_binding(warmup_snapshot, warmups)
            if evidence_snapshots is not None:
                evidence_snapshots.append(warmup_snapshot)
        except EvidenceSnapshotError as exc:
            errors.append(f"invalid warmup_timings.jsonl evidence: {exc}")
            warmups = []
            warmup_binding = None
    else:
        warmups = []
        warmup_binding = None
    if len(reports) != expected_measurements:
        errors.append(f"MP4 count {len(reports)} != expected {expected_measurements}")
    if len(timings) != expected_measurements:
        errors.append(f"timings count {len(timings)} != expected {expected_measurements}")
    measurement_errors, ordered_prompts = measurement_record_protocol_errors(
        timings, environment
    )
    errors.extend(measurement_errors)
    for item in timings:
        if item.get("status") != "ok" or item.get("record_type") != "measurement":
            errors.append("timings.jsonl contains a non-ok/non-measurement record")
            break
    errors.extend(
        warmup_record_protocol_errors(warmups, environment, ordered_prompts)
    )

    all_run_records = [*warmups, *timings]
    for record_type, records in (("warmup", warmups), ("measurement", timings)):
        for index, item in enumerate(records):
            for field in (
                "benchmark_candidate",
                "benchmark_valid",
                "use_cfg_cache",
                "use_block_cache",
            ):
                if type(item.get(field)) is not bool:
                    errors.append(f"{record_type}[{index}] {field} must be a JSON boolean")
            if item.get("benchmark_valid") is not False:
                errors.append(
                    f"{record_type}[{index}] benchmark_valid must remain false"
                )
            if not strict_json_equal(item.get("benchmark_candidate"), candidate):
                errors.append(
                    f"{record_type}[{index}] benchmark_candidate disagrees with environment.json"
                )
            for field in ("use_cfg_cache", "use_block_cache"):
                if not strict_json_equal(item.get(field), environment.get(field)):
                    errors.append(
                        f"{record_type}[{index}] {field} disagrees with environment.json"
                    )
    if environment.get("use_block_cache") is True or any(
        item.get("use_block_cache") is True for item in all_run_records
    ):
        schedule_fields = (
            "use_block_cache",
            "sample_steps",
            "slg_layer",
            "use_cfg_cache",
            "cfg_cache_start_step",
            "cfg_cache_end_step",
            "cfg_cache_refresh_interval",
            "block_cache_start_block",
            "block_cache_end_block",
            "block_cache_policy",
            "block_cache_cosine_threshold",
            "block_cache_max_consecutive_reuses",
        )
        for record_type, records in (
            ("warmup", warmups),
            ("measurement", timings),
        ):
            for index, item in enumerate(records):
                for field in schedule_fields:
                    if not strict_json_equal(item.get(field), environment.get(field)):
                        errors.append(
                            f"{record_type}[{index}] {field}="
                            f"{item.get(field)!r} != environment "
                            f"{environment.get(field)!r}"
                        )
                if (
                    item.get("use_block_cache") is True
                    and item.get("block_cache_policy") == "fixed"
                ):
                    errors.extend(
                        f"{record_type}[{index}] fixed block-cache schedule: "
                        f"{error}"
                        for error in fixed_block_cache_metric_errors(item)
                    )

    for record_type, records in (("warmup", warmups), ("measurement", timings)):
        for record_index, item in enumerate(records):
            validate_gpu_monitor(
                item.get("gpu_process_monitor"),
                expected_gpu_identity,
                (
                    pre_run_gpu.get("nvidia_smi_binary")
                    if isinstance(pre_run_gpu, dict)
                    else None
                ),
                (
                    pre_run_gpu.get("boot_id")
                    if isinstance(pre_run_gpu, dict)
                    else None
                ),
                environment.get("gpu_process_monitor_interval_seconds"),
                item.get("total_generation_seconds"),
                candidate,
                f"{record_type}[{record_index}]",
                errors,
            )

    errors.extend(
        artifact_report_protocol_errors(
            reports, timings, run_dir, metrics_payloads=metrics_payloads
        )
    )

    preflight = {}
    preflight_path = run_dir / "preflight.json"
    parsed_preflight = parse_evidence_json(preflight_path, "preflight.json")
    if isinstance(parsed_preflight, dict):
        preflight = parsed_preflight
        if preflight.get("errors"):
            errors.append(f"preflight contains errors: {preflight['errors']}")

    if attention_method == "sparge":
        receipt_path = run_dir / "spargeattn-install.json"
        copied_receipt = parse_evidence_json(
            receipt_path, "copied SpargeAttn receipt"
        )
        expected_gpu_uuid = (
            expected_gpu_identity[1]
            if expected_gpu_identity is not None
            else None
        )
        for error in sparge_receipt_evidence_errors(
            copied_receipt, expected_gpu_uuid=expected_gpu_uuid
        ):
            errors.append(f"copied SpargeAttn receipt: {error}")

        if isinstance(copied_receipt, dict):
            build_metadata = copied_receipt.get("build_log")
            build_log_path = run_dir / "spargeattn-build.log"
            build_log_snapshot = required_snapshots.get("spargeattn-build.log")
            if isinstance(build_metadata, dict) and build_log_snapshot is not None:
                if (
                    not strict_json_equal(
                        build_log_snapshot.size, build_metadata.get("bytes")
                    )
                    or not strict_json_equal(
                        build_log_snapshot.sha256, build_metadata.get("sha256")
                    )
                ):
                    errors.append(
                        "copied SpargeAttn build log differs from install receipt"
                    )
            install_gpu_metadata = copied_receipt.get("install_pre_run_gpu")
            install_gpu_path = run_dir / "spargeattn-install-pre_run_gpu.json"
            install_gpu_snapshot = required_snapshots.get(
                "spargeattn-install-pre_run_gpu.json"
            )
            if (
                isinstance(install_gpu_metadata, dict)
                and install_gpu_snapshot is not None
            ):
                if (
                    not strict_json_equal(
                        install_gpu_snapshot.size,
                        install_gpu_metadata.get("bytes"),
                    )
                    or not strict_json_equal(
                        install_gpu_snapshot.sha256,
                        install_gpu_metadata.get("sha256"),
                    )
                ):
                    errors.append(
                        "copied SpargeAttn install GPU evidence differs from receipt"
                    )
                install_gpu_report = parse_evidence_json(
                    install_gpu_path,
                    "SpargeAttn install GPU evidence",
                )
                if isinstance(install_gpu_report, dict):
                    if (
                        not is_json_int(
                            install_gpu_report.get("schema_version")
                        )
                        or install_gpu_report.get("schema_version")
                        != GPU_EVIDENCE_SCHEMA_VERSION
                        or install_gpu_report.get("check_type") != "pre_run_idle"
                        or install_gpu_report.get("valid_for_run") is not True
                        or install_gpu_report.get("idle") is not True
                        or not is_json_int(
                            install_gpu_report.get("process_count")
                        )
                        or install_gpu_report.get("process_count") != 0
                        or install_gpu_report.get("processes") != []
                        or install_gpu_report.get("errors") != []
                        or install_gpu_report.get("device_uuid")
                        != expected_gpu_uuid
                    ):
                        errors.append(
                            "SpargeAttn install GPU evidence is not an idle "
                            "record for the benchmark GPU UUID"
                        )

        if not environment.get("spas_sage_attn"):
            errors.append("environment is missing spas_sage_attn package version")
        preflight_sparge = preflight.get("spargeattn")
        if not isinstance(preflight_sparge, dict):
            errors.append("Sparge run preflight is missing spargeattn evidence")
        else:
            if preflight_sparge.get("pinned_commit") != SPARGE_PROVENANCE["pinned_commit"]:
                errors.append("preflight SpargeAttn commit differs from formal pin")
            if preflight_sparge.get("api") != SPARGE_PROVENANCE["api"]:
                errors.append("preflight SpargeAttn API differs from formal pin")
            if preflight_sparge.get("installed_files_verified") is not True:
                errors.append("preflight did not verify installed SpargeAttn files")
            if preflight_sparge.get("install_receipt_contents") != copied_receipt:
                errors.append("preflight SpargeAttn receipt differs from copied receipt")
        receipt_microtest = (
            copied_receipt.get("microtest")
            if isinstance(copied_receipt, dict)
            else None
        )
        for error in sparge_microtest_evidence_errors(
            receipt_microtest, expected_gpu_uuid=expected_gpu_uuid
        ):
            errors.append(f"Sparge install microtest: {error}")
        preflight_microtest = preflight.get("spargeattn_microtest")
        for error in sparge_microtest_evidence_errors(
            preflight_microtest, expected_gpu_uuid=expected_gpu_uuid
        ):
            errors.append(f"Sparge preflight microtest: {error}")

        expected_settings = {
            "topk": environment.get("sparge_topk"),
            "pvthreshd": environment.get("sparge_pvthreshd"),
            "smooth_k": environment.get("sparge_smooth_k"),
        }
        for record_type, records in (("measurement", timings), ("warmup", warmups)):
            for index, item in enumerate(records):
                dispatcher = item.get("video_self_attention_dispatcher")
                if not isinstance(dispatcher, dict):
                    errors.append(
                        f"{record_type}[{index}] is missing video dispatcher evidence"
                    )
                    continue
                validate_sparge_dispatcher(
                    dispatcher,
                    errors,
                    expected_receipt=copied_receipt,
                    expected_settings=expected_settings,
                    expected_gpu_uuid=expected_gpu_uuid,
                    context=f"{record_type}[{index}]",
                )

    elif attention_method == "radial":
        receipt_path = run_dir / "radialattn-install.json"
        copied_receipt = parse_evidence_json(
            receipt_path, "copied Radial receipt"
        )
        for error in radial_receipt_evidence_errors(copied_receipt):
            errors.append(f"copied Radial receipt: {error}")
        expected_runtime_dependencies = (
            expected_radial_runtime_dependency_evidence(copied_receipt)
        )
        expected_optional_import_path = (
            "/cache/liluchen/FastA2V/envs/ovi/lib/python3.11/lib64"
        )

        copied_artifacts = {
            "source_module": run_dir / "radial-attention-source.py",
            "derived_module": run_dir / "radial-attention-derived.py",
            "optional_imports_patch": (
                run_dir / "radial-attention-optional-imports.patch"
            ),
        }
        if isinstance(copied_receipt, dict):
            for field, path in copied_artifacts.items():
                metadata = copied_receipt.get(field)
                artifact_snapshot = required_snapshots.get(path.name)
                if not isinstance(metadata, dict):
                    errors.append(f"copied Radial receipt lacks {field}")
                elif artifact_snapshot is not None and (
                    not strict_json_equal(
                        metadata.get("bytes"), artifact_snapshot.size
                    )
                    or not strict_json_equal(
                        metadata.get("sha256"), artifact_snapshot.sha256
                    )
                ):
                    errors.append(
                        f"copied Radial {field} differs from install receipt"
                    )

        flashinfer_manifest_path = (
            run_dir / "radial-flashinfer-manifest.json"
        )
        copied_flashinfer_manifest = parse_evidence_json(
            flashinfer_manifest_path, "copied FlashInfer manifest"
        )
        flashinfer_manifest_snapshot = required_snapshots.get(
            "radial-flashinfer-manifest.json"
        )
        if isinstance(copied_receipt, dict):
            manifest_fingerprint = copied_receipt.get("flashinfer_manifest")
            if not isinstance(manifest_fingerprint, dict):
                errors.append("copied Radial receipt lacks flashinfer_manifest")
            elif flashinfer_manifest_snapshot is not None and (
                not strict_json_equal(
                    manifest_fingerprint.get("bytes"),
                    flashinfer_manifest_snapshot.size,
                )
                or not strict_json_equal(
                    manifest_fingerprint.get("sha256"),
                    flashinfer_manifest_snapshot.sha256,
                )
            ):
                errors.append(
                    "copied FlashInfer manifest differs from install receipt"
                )
        for error in flashinfer_manifest_evidence_errors(
            copied_flashinfer_manifest, copied_receipt
        ):
            errors.append(f"copied FlashInfer manifest: {error}")

        if environment.get("flashinfer_python") != FLASHINFER_VERSION:
            errors.append(
                "environment FlashInfer version differs from fixed candidate"
            )
        preflight_radial = preflight.get("radialattn")
        if not isinstance(preflight_radial, dict):
            errors.append("Radial run preflight is missing radialattn evidence")
        else:
            validate_radial_preflight_static_evidence(
                preflight_radial,
                errors,
            )
            if not strict_json_equal(
                preflight_radial.get("install_receipt_contents"),
                copied_receipt,
            ):
                errors.append("preflight Radial receipt differs from copied receipt")
            validate_radial_runtime_dependency_evidence(
                preflight_radial.get(
                    "runtime_dependencies_before_optional_imports"
                ),
                expected_runtime_dependencies,
                "preflight Radial before optional imports",
                errors,
            )
            validate_radial_optional_import_loader_evidence(
                preflight_radial.get("optional_import_loader_evidence"),
                expected_runtime_dependencies,
                expected_optional_import_path,
                "preflight Radial optional imports",
                errors,
            )
        expected_gpu_uuid = (
            expected_gpu_identity[1]
            if expected_gpu_identity is not None
            else None
        )
        preflight_microtest = preflight.get("radialattn_microtest")
        radial_microtest_errors = radial_microtest_evidence_errors(
            preflight_microtest,
            expected_gpu_uuid=expected_gpu_uuid,
            expected_pre_run_gpu=pre_run_gpu,
            expected_pre_run_gpu_sha256=(
                required_snapshots["pre_run_gpu.json"].sha256
                if required_snapshots.get("pre_run_gpu.json") is not None
                else None
            ),
            expected_pre_run_gpu_path=str(pre_run_gpu_path),
            expected_python_executable=preflight.get("python_executable"),
        )
        for error in radial_microtest_errors:
            errors.append(f"Radial preflight microtest: {error}")
        radial_claims = validated_radial_preflight_claims(
            preflight_microtest,
            radial_microtest_errors,
        )
        if isinstance(preflight_microtest, dict):
            for phase in ("before_cuda", "after_cuda"):
                validate_radial_runtime_dependency_evidence(
                    preflight_microtest.get(
                        f"runtime_dependencies_{phase}"
                    ),
                    expected_runtime_dependencies,
                    f"Radial preflight microtest {phase}",
                    errors,
                )

        inference_bootstrap = environment.get("radial_loader_bootstrap")
        if not isinstance(inference_bootstrap, dict):
            errors.append(
                "environment is missing Radial inference loader bootstrap evidence"
            )
        else:
            if set(inference_bootstrap) != {
                "status",
                "receipt_path",
                "before_optional_imports",
                "after_optional_imports",
            }:
                errors.append(
                    "environment Radial loader bootstrap fields are invalid"
                )
            if inference_bootstrap.get("status") != "ok":
                errors.append(
                    "environment Radial loader bootstrap status is not ok"
                )
            if inference_bootstrap.get("receipt_path") != (
                "/cache/liluchen/FastA2V/radialattn-install.json"
            ):
                errors.append(
                    "environment Radial loader bootstrap used the wrong receipt"
                )
            validate_radial_runtime_dependency_evidence(
                inference_bootstrap.get("before_optional_imports"),
                expected_runtime_dependencies,
                "environment Radial before optional imports",
                errors,
            )
            validate_radial_optional_import_loader_evidence(
                inference_bootstrap.get("after_optional_imports"),
                expected_runtime_dependencies,
                expected_optional_import_path,
                "environment Radial optional imports",
                errors,
            )

        expected_settings = {
            "profile": environment.get("radial_profile"),
            "decay_factor": environment.get("radial_decay_factor"),
            "model_type": environment.get("radial_model_type"),
            "block_size": environment.get("radial_block_size"),
        }
        receipt_for_dispatcher = dict(copied_receipt or {})
        receipt_for_dispatcher["_original_path"] = (
            "/cache/liluchen/FastA2V/radialattn-install.json"
        )
        receipt_for_dispatcher["_copied_path"] = str(receipt_path)
        for record_type, records in (
            ("measurement", timings),
            ("warmup", warmups),
        ):
            for index, item in enumerate(records):
                dispatcher = item.get("video_self_attention_dispatcher")
                if not isinstance(dispatcher, dict):
                    errors.append(
                        f"{record_type}[{index}] is missing video dispatcher evidence"
                    )
                    continue
                validate_radial_dispatcher(
                    dispatcher,
                    errors,
                    expected_receipt=receipt_for_dispatcher,
                    expected_settings=expected_settings,
                    context=f"{record_type}[{index}]",
                )

    evidence_hashes = environment.get("evidence_file_sha256", {})
    if not isinstance(evidence_hashes, dict):
        errors.append("environment evidence_file_sha256 must be a JSON object")
        evidence_hashes = {}
    required_hashed_evidence = {
        "pre_run_gpu.json",
        "preflight.json",
        "environment.freeze.txt",
        "checkpoint_manifest.json",
    }
    if attention_method == "sparge":
        required_hashed_evidence.update(
            {
                "spargeattn-install.json",
                "spargeattn-build.log",
                "spargeattn-install-pre_run_gpu.json",
            }
        )
    elif attention_method == "radial":
        required_hashed_evidence.update(
            {
                "radialattn-install.json",
                "radial-flashinfer-manifest.json",
                "radial-attention-source.py",
                "radial-attention-derived.py",
                "radial-attention-optional-imports.patch",
            }
        )
    missing_hashes = sorted(required_hashed_evidence - set(evidence_hashes))
    if missing_hashes:
        errors.append(f"environment is missing evidence hashes: {missing_hashes}")
    for filename, expected_hash in evidence_hashes.items():
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            errors.append(f"invalid evidence hash filename: {filename!r}")
            continue
        snapshot = capture_evidence(run_dir / filename, filename)
        actual_hash = snapshot.sha256 if snapshot is not None else None
        if not expected_hash or expected_hash != actual_hash:
            errors.append(
                f"evidence hash mismatch for {filename}: expected={expected_hash} actual={actual_hash}"
            )
    pre_run_snapshot = required_snapshots.get("pre_run_gpu.json")
    actual_pre_run_hash = (
        pre_run_snapshot.sha256 if pre_run_snapshot is not None else None
    )
    if environment.get("pre_run_gpu_sha256") != actual_pre_run_hash:
        errors.append("pre_run_gpu.json SHA256 does not match environment.json")
    run_config_path = run_dir / "run_config.yaml"
    run_config_snapshot = required_snapshots.get("run_config.yaml")
    if run_config_snapshot is not None and (
        environment.get("run_config_sha256") != run_config_snapshot.sha256
    ):
        errors.append("run_config.yaml SHA256 does not match environment.json")

    for guard in protocol_guards:
        try:
            _revalidate_publication_guard(guard)
        except EvidenceSnapshotError as exc:
            errors.append(f"run evidence stable snapshot failed: {exc}")

    if timings_snapshot is not None:
        try:
            _revalidate_snapshot(timings_snapshot)
        except EvidenceSnapshotError as exc:
            errors.append(f"timings.jsonl stable snapshot failed: {exc}")
    errors.extend(
        _binding_shape_errors(
            timings_binding,
            {"path", "bytes", "sha256", "record_count"},
            "timings_binding",
        )
    )
    if isinstance(timings_binding, dict) and not strict_json_equal(
        timings_binding.get("record_count"), expected_measurements
    ):
        errors.append("timings_binding record_count differs from protocol")
    if warmup_snapshot is not None:
        try:
            _revalidate_snapshot(warmup_snapshot)
        except EvidenceSnapshotError as exc:
            errors.append(f"warmup_timings.jsonl stable snapshot failed: {exc}")
    if expected_warmups == 0:
        if warmup_binding is not None:
            errors.append("zero-warmup protocol must persist a null warmup binding")
    else:
        errors.extend(
            _binding_shape_errors(
                warmup_binding,
                {"path", "bytes", "sha256", "record_count"},
                "warmup_timings_binding",
            )
        )
        if isinstance(warmup_binding, dict) and not strict_json_equal(
            warmup_binding.get("record_count"), expected_warmups
        ):
            errors.append("warmup_timings_binding record_count differs from protocol")

    benchmark_valid = bool(
        candidate
        and not errors
        and environment.get("debug_forward") is False
        and environment.get("git_dirty") is False
        and expected_warmups >= 1
        and measurement_runs >= 3
        and all(isinstance(report, dict) and not report.get("errors") for report in reports)
    )
    protocol = {
        "status": "failed" if errors else "ok",
        "errors": errors,
        "expected_warmup_records": expected_warmups,
        "observed_warmup_records": len(warmups),
        "expected_measurement_records": expected_measurements,
        "observed_measurement_records": len(timings),
        "timings_binding": timings_binding,
        "warmup_timings_binding": warmup_binding,
        "benchmark_candidate": candidate,
        "benchmark_valid": benchmark_valid,
    }
    if attention_method == "radial":
        protocol["radial_evidence"] = radial_claims
    return protocol


def build_verification_summary(reports, protocol):
    """Build the persisted report, including validated method-specific claims."""

    failed = any(item["errors"] for item in reports) or (
        protocol is not None and protocol["errors"]
    )
    summary = {
        "status": "failed" if failed else "ok",
        "artifact_count": len(reports),
        "artifacts": reports,
        "protocol": protocol,
        "benchmark_valid": bool(
            isinstance(protocol, dict) and protocol.get("benchmark_valid") is True
        ),
    }
    if isinstance(protocol, dict) and "radial_evidence" in protocol:
        summary["radial_evidence"] = protocol["radial_evidence"]
    return summary


def _write_json_temp(output_path, payload):
    """Write and fsync an O_EXCL candidate beside its final destination."""

    output_path = Path(output_path)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = None
    temporary_path = None
    for _attempt in range(128):
        temporary_path = output_path.parent / (
            f".{output_path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
        )
        try:
            descriptor = os.open(temporary_path, flags, 0o600)
            break
        except FileExistsError:
            continue
    if descriptor is None or temporary_path is None:
        raise FileExistsError(
            f"cannot allocate exclusive verification temp beside {output_path}"
        )
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)
    return temporary_path


def _replace_json_temp(temporary_path, output_path):
    output_path = Path(output_path)
    os.replace(temporary_path, output_path)
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_descriptor = os.open(output_path.parent, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _atomic_publish_json(output_path, payload):
    temporary_path = _write_json_temp(output_path, payload)
    try:
        _replace_json_temp(temporary_path, output_path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _failed_publication_summary(reason, reports=None, protocol=None):
    rendered_reason = str(reason)
    if isinstance(protocol, dict):
        protocol = dict(protocol)
        protocol["status"] = "failed"
        protocol["benchmark_valid"] = False
        protocol["errors"] = [
            *list(protocol.get("errors") or []),
            f"verification publication failed: {rendered_reason}",
        ]
    return {
        "status": "failed",
        "reason": rendered_reason,
        "artifact_count": len(reports or []),
        "artifacts": reports or [],
        "protocol": protocol,
        "benchmark_valid": False,
    }


def _publish_verified_summary(output_path, summary, evidence_snapshots):
    """Revalidate before and immediately after publishing a successful summary."""

    def revalidation_errors():
        observed = []
        for guard in evidence_snapshots:
            try:
                _revalidate_publication_guard(guard)
            except EvidenceSnapshotError as exc:
                observed.append(str(exc))
        return observed

    def failed_summary(source, phase, observed):
        failed = json.loads(json.dumps(source, allow_nan=False))
        failed["status"] = "failed"
        failed["benchmark_valid"] = False
        messages = [f"{phase}: {error}" for error in observed]
        failed.setdefault("publication_errors", []).extend(messages)
        protocol = failed.get("protocol")
        if isinstance(protocol, dict):
            protocol["status"] = "failed"
            protocol["benchmark_valid"] = False
            protocol.setdefault("errors", []).extend(
                f"publication evidence revalidation: {message}"
                for message in messages
            )
        return failed

    candidate_path = _write_json_temp(output_path, summary)
    drift_errors = revalidation_errors()
    if drift_errors:
        try:
            candidate_path.unlink()
        except FileNotFoundError:
            pass
        summary = failed_summary(summary, "pre-publish", drift_errors)
        candidate_path = _write_json_temp(output_path, summary)
    try:
        _replace_json_temp(candidate_path, output_path)
    finally:
        try:
            candidate_path.unlink()
        except FileNotFoundError:
            pass
    if summary.get("status") == "ok":
        drift_errors = revalidation_errors()
        if drift_errors:
            summary = failed_summary(summary, "post-publish", drift_errors)
            _atomic_publish_json(output_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="MP4 file or run directory")
    parser.add_argument(
        "--media-only",
        action="store_true",
        help="validate streams/content without FastA2V metrics or run protocol",
    )
    parser.add_argument(
        "--expected-video-frames",
        type=int,
        default=121,
        help="exact decoded frame count required (default: 121)",
    )
    args = parser.parse_args()

    if args.expected_video_frames < 1:
        parser.error("--expected-video-frames must be positive")
    input_is_file = args.path.suffix.lower() == ".mp4"
    run_dir = (
        args.path.parent.resolve(strict=True)
        if input_is_file
        else args.path.resolve(strict=True)
    )
    output_path = (
        _canonical_leaf_path(args.path.with_suffix(".verification.json"))
        if input_is_file
        else run_dir / "verification.json"
    )
    _atomic_publish_json(
        output_path,
        _failed_publication_summary("verification_in_progress"),
    )

    reports = []
    protocol = None
    evidence_snapshots = []
    metrics_payloads = {}
    try:
        for executable in ("ffmpeg", "ffprobe"):
            if shutil.which(executable) is None:
                raise RuntimeError(f"required executable not found: {executable}")

        if input_is_file:
            paths = [args.path]
        else:
            paths = sorted(
                path
                for path in run_dir.iterdir()
                if path.name.lower().endswith(".mp4")
            )
        if not paths:
            raise RuntimeError(f"no MP4 artifacts found under {args.path}")
        for path in paths:
            try:
                reports.append(
                    verify(
                        path,
                        require_metrics=not args.media_only,
                        expected_video_frames=args.expected_video_frames,
                        run_dir=run_dir,
                        evidence_snapshots=evidence_snapshots,
                        metrics_payloads=metrics_payloads,
                    )
                )
            except (EvidenceSnapshotError, OSError, ValueError, subprocess.SubprocessError) as exc:
                try:
                    failed_path = str(_canonical_leaf_path(path))
                except OSError:
                    failed_path = str(path)
                reports.append(
                    {
                        "path": failed_path,
                        "sha256": None,
                        "measurement_index": None,
                        "prompt_index": None,
                        "sample_index": None,
                        "prompt": None,
                        "seed": None,
                        "metrics_path": str(Path(failed_path).with_suffix(".metrics.json")),
                        "artifact_binding": None,
                        "metrics_binding": None,
                        "status": "failed",
                        "errors": [str(exc)],
                        "video": {},
                        "audio": {},
                    }
                )
        protocol = (
            verify_run_protocol(
                run_dir,
                reports,
                evidence_snapshots=evidence_snapshots,
                metrics_payloads=metrics_payloads,
            )
            if not input_is_file and not args.media_only
            else None
        )
        summary = build_verification_summary(reports, protocol)
        summary = _publish_verified_summary(
            output_path, summary, evidence_snapshots
        )
    except BaseException as exc:
        summary = _failed_publication_summary(exc, reports, protocol)
        _atomic_publish_json(output_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 1 if summary["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
