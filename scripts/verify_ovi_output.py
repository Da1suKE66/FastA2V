#!/usr/bin/env python3
"""Fail unless every generated MP4 has valid Ovi video and non-silent audio."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
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

def run(command):
    return subprocess.check_output(command, stderr=subprocess.STDOUT)


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


def decode_video_gray(path):
    raw = run([
        "ffmpeg",
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
def verify(path, require_metrics=True, expected_video_frames=121):
    info = probe(path)
    artifact_sha256 = sha256(path)
    videos = [stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"]
    audios = [stream for stream in info.get("streams", []) if stream.get("codec_type") == "audio"]
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

    samples = decode_audio(path) if audios else np.empty(0, dtype=np.float32)
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

    gray = decode_video_gray(path) if videos else np.empty(0, dtype=np.uint8)
    video_std = float(gray.std()) if gray.size else 0.0
    if gray.size == 0 or video_std <= 2.0:
        errors.append(f"decoded video is blank or nearly constant: std={video_std}")

    metrics_path = path.with_suffix(".metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
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
            "benchmark_candidate",
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
        if Path(metrics.get("output_path", "")).resolve() != path.resolve():
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

        block_cache_enabled = bool(metrics.get("use_block_cache"))
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

    return {
        "path": str(path.resolve()),
        "sha256": artifact_sha256,
        "status": "failed" if errors else "ok",
        "errors": errors,
        "video": {
            "codec": video.get("codec_name"),
            "width": width,
            "height": height,
            "frames": frames,
            "duration_seconds": duration,
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


def verify_run_protocol(run_dir, reports):
    errors = []
    radial_claims = None
    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text()) if environment_path.is_file() else {}
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
    for filename in required_files:
        if not (run_dir / filename).is_file():
            errors.append(f"missing run evidence file: {filename}")

    pre_run_gpu_path = run_dir / "pre_run_gpu.json"
    try:
        pre_run_gpu = json.loads(pre_run_gpu_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        pre_run_gpu = None
        errors.append(f"invalid pre_run_gpu.json: {exc}")
    expected_gpu_identity = validate_pre_run_gpu(
        pre_run_gpu, environment, errors
    )

    candidate = bool(environment.get("benchmark_eligible"))
    expected_measurements = int(environment.get("expected_measurement_records", -1))
    expected_warmups = int(environment.get("expected_warmup_records", -1))
    measurement_runs = int(environment.get("measurement_runs", -1))
    per_repeat = int(environment.get("prompt_count", 0)) * int(
        environment.get("each_example_n_times", 0)
    )

    timings = read_jsonl(run_dir / "timings.jsonl")
    warmups = read_jsonl(run_dir / "warmup_timings.jsonl")
    if len(reports) != expected_measurements:
        errors.append(f"MP4 count {len(reports)} != expected {expected_measurements}")
    if len(timings) != expected_measurements:
        errors.append(f"timings count {len(timings)} != expected {expected_measurements}")
    if len(warmups) != expected_warmups:
        errors.append(f"warmup count {len(warmups)} != expected {expected_warmups}")

    expected_indices = {
        index for index in range(max(measurement_runs, 0))
        for _ in range(max(per_repeat, 0))
    }
    actual_indices = {item.get("measurement_index") for item in timings}
    if actual_indices != expected_indices:
        errors.append(
            f"measurement indices {sorted(str(x) for x in actual_indices)} "
            f"!= expected {sorted(expected_indices)}"
        )
    for index in range(max(measurement_runs, 0)):
        count = sum(item.get("measurement_index") == index for item in timings)
        if count != per_repeat:
            errors.append(f"measurement index {index} has {count} records, expected {per_repeat}")
    for item in timings:
        if item.get("status") != "ok" or item.get("record_type") != "measurement":
            errors.append("timings.jsonl contains a non-ok/non-measurement record")
            break
    for item in warmups:
        if item.get("status") != "ok" or item.get("record_type") != "warmup":
            errors.append("warmup_timings.jsonl contains an invalid warm-up record")
            break

    expected_prompts_sha256 = environment.get("prompts_sha256")
    for measurement_index in range(max(measurement_runs, 0)):
        measurement_prompts = [
            item.get("prompt")
            for item in timings
            if item.get("measurement_index") == measurement_index
        ]
        if not all(isinstance(prompt, str) for prompt in measurement_prompts):
            errors.append(
                f"measurement index {measurement_index} contains an invalid "
                "prompt value"
            )
        elif prompt_sequence_sha256(measurement_prompts) != expected_prompts_sha256:
            errors.append(
                f"measurement index {measurement_index} prompt sequence does "
                "not match the fixed environment prompt hash"
            )
    for warmup_index, item in enumerate(warmups):
        warmup_prompt = item.get("prompt")
        if not isinstance(warmup_prompt, str):
            errors.append(
                f"warmup index {warmup_index} has an invalid prompt value"
            )
        elif prompt_sequence_sha256([warmup_prompt]) != expected_prompts_sha256:
            errors.append(
                f"warmup index {warmup_index} prompt does not match the fixed "
                "environment prompt hash"
            )

    all_run_records = [*warmups, *timings]
    if environment.get("use_block_cache") or any(
        item.get("use_block_cache") for item in all_run_records
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
                    if item.get(field) != environment.get(field):
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

    artifact_hashes = {report["sha256"] for report in reports}
    timing_hashes = {item.get("output_sha256") for item in timings}
    if artifact_hashes != timing_hashes:
        errors.append("timings.jsonl output hashes do not match the verified artifacts")

    preflight = {}
    preflight_path = run_dir / "preflight.json"
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text())
        if preflight.get("errors"):
            errors.append(f"preflight contains errors: {preflight['errors']}")

    if attention_method == "sparge":
        receipt_path = run_dir / "spargeattn-install.json"
        try:
            copied_receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            copied_receipt = None
            errors.append(f"invalid copied SpargeAttn receipt: {exc}")
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
            if isinstance(build_metadata, dict) and build_log_path.is_file():
                if (
                    build_log_path.stat().st_size != build_metadata.get("bytes")
                    or sha256(build_log_path) != build_metadata.get("sha256")
                ):
                    errors.append(
                        "copied SpargeAttn build log differs from install receipt"
                    )
            install_gpu_metadata = copied_receipt.get("install_pre_run_gpu")
            install_gpu_path = run_dir / "spargeattn-install-pre_run_gpu.json"
            if isinstance(install_gpu_metadata, dict) and install_gpu_path.is_file():
                if (
                    install_gpu_path.stat().st_size
                    != install_gpu_metadata.get("bytes")
                    or sha256(install_gpu_path)
                    != install_gpu_metadata.get("sha256")
                ):
                    errors.append(
                        "copied SpargeAttn install GPU evidence differs from receipt"
                    )
                try:
                    install_gpu_report = json.loads(install_gpu_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    install_gpu_report = None
                    errors.append(
                        f"invalid SpargeAttn install GPU evidence: {exc}"
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
        try:
            copied_receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            copied_receipt = None
            errors.append(f"invalid copied Radial receipt: {exc}")
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
                if not isinstance(metadata, dict):
                    errors.append(f"copied Radial receipt lacks {field}")
                elif path.is_file() and (
                    not strict_json_equal(
                        metadata.get("bytes"), path.stat().st_size
                    )
                    or not strict_json_equal(
                        metadata.get("sha256"), sha256(path)
                    )
                ):
                    errors.append(
                        f"copied Radial {field} differs from install receipt"
                    )

        flashinfer_manifest_path = (
            run_dir / "radial-flashinfer-manifest.json"
        )
        try:
            copied_flashinfer_manifest = json.loads(
                flashinfer_manifest_path.read_text()
            )
        except (OSError, json.JSONDecodeError) as exc:
            copied_flashinfer_manifest = None
            errors.append(f"invalid copied FlashInfer manifest: {exc}")
        if isinstance(copied_receipt, dict):
            manifest_fingerprint = copied_receipt.get("flashinfer_manifest")
            if not isinstance(manifest_fingerprint, dict):
                errors.append("copied Radial receipt lacks flashinfer_manifest")
            elif flashinfer_manifest_path.is_file() and (
                not strict_json_equal(
                    manifest_fingerprint.get("bytes"),
                    flashinfer_manifest_path.stat().st_size,
                )
                or not strict_json_equal(
                    manifest_fingerprint.get("sha256"),
                    sha256(flashinfer_manifest_path),
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
                sha256(pre_run_gpu_path)
                if pre_run_gpu_path.is_file()
                else None
            ),
            expected_pre_run_gpu_path=str(pre_run_gpu_path.resolve()),
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
        path = run_dir / filename
        actual_hash = sha256(path) if path.is_file() else None
        if not expected_hash or expected_hash != actual_hash:
            errors.append(
                f"evidence hash mismatch for {filename}: expected={expected_hash} actual={actual_hash}"
            )
    actual_pre_run_hash = (
        sha256(pre_run_gpu_path) if pre_run_gpu_path.is_file() else None
    )
    if environment.get("pre_run_gpu_sha256") != actual_pre_run_hash:
        errors.append("pre_run_gpu.json SHA256 does not match environment.json")
    run_config_path = run_dir / "run_config.yaml"
    if run_config_path.is_file() and environment.get("run_config_sha256") != sha256(run_config_path):
        errors.append("run_config.yaml SHA256 does not match environment.json")

    if any(item.get("benchmark_candidate") != candidate for item in timings):
        errors.append("per-measurement benchmark_candidate disagrees with environment.json")
    benchmark_valid = bool(
        candidate
        and not errors
        and not environment.get("debug_forward")
        and not environment.get("git_dirty")
        and expected_warmups >= 1
        and measurement_runs >= 3
        and all(not report["errors"] for report in reports)
    )
    protocol = {
        "status": "failed" if errors else "ok",
        "errors": errors,
        "expected_warmup_records": expected_warmups,
        "observed_warmup_records": len(warmups),
        "expected_measurement_records": expected_measurements,
        "observed_measurement_records": len(timings),
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
        "benchmark_valid": bool(protocol and protocol["benchmark_valid"]),
    }
    if isinstance(protocol, dict) and "radial_evidence" in protocol:
        summary["radial_evidence"] = protocol["radial_evidence"]
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

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable not found: {executable}")

    paths = [args.path] if args.path.is_file() else sorted(args.path.glob("*.mp4"))
    if not paths:
        raise SystemExit(f"no MP4 artifacts found under {args.path}")
    reports = [
        verify(
            path,
            require_metrics=not args.media_only,
            expected_video_frames=args.expected_video_frames,
        )
        for path in paths
    ]
    run_dir = args.path if args.path.is_dir() else args.path.parent
    protocol = (
        verify_run_protocol(run_dir, reports)
        if args.path.is_dir() and not args.media_only
        else None
    )
    summary = build_verification_summary(reports, protocol)
    output_path = args.path.with_suffix(".verification.json") if args.path.is_file() else args.path / "verification.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 1 if summary["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
