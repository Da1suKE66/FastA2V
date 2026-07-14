"""Fail-closed NVIDIA GPU identity and compute-process evidence.

This module deliberately uses only the Python standard library.  The run
scripts can therefore prove that physical GPU 0 is idle before importing
PyTorch or performing any CUDA preflight/model loading.
"""

from datetime import datetime, timezone
import hashlib
import math
import os
import re
import secrets
import stat
import subprocess
import threading
import time


GPU_EVIDENCE_SCHEMA_VERSION = 1
TRUSTED_NVIDIA_SMI_PATH = "/usr/bin/nvidia-smi"
TRUSTED_NVIDIA_SMI_BYTES = 686384
TRUSTED_NVIDIA_SMI_SHA256 = (
    "70b7292808702f3ef3ff93af56367f45b3232c9ef131f48fc9264a940c00e57a"
)


def _is_json_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def trusted_nvidia_smi_metadata_errors(metadata):
    """Validate the fixed root-owned system nvidia-smi binary evidence."""

    if not isinstance(metadata, dict):
        return ["trusted nvidia-smi metadata must be a JSON object"]
    errors = []
    expected = {
        "requested_path": TRUSTED_NVIDIA_SMI_PATH,
        "resolved_path": TRUSTED_NVIDIA_SMI_PATH,
        "owner_uid": 0,
        "owner_gid": 0,
        "mode": 0o755,
        "bytes": TRUSTED_NVIDIA_SMI_BYTES,
        "sha256": TRUSTED_NVIDIA_SMI_SHA256,
    }
    for field, expected_value in expected.items():
        value = metadata.get(field)
        if (
            isinstance(expected_value, int)
            and not _is_json_int(value)
        ) or value != expected_value:
            errors.append(
                f"trusted nvidia-smi {field}={value!r} != "
                f"{expected_value!r}"
            )
    for field in ("device", "inode"):
        value = metadata.get(field)
        if not _is_json_int(value) or value <= 0:
            errors.append(f"trusted nvidia-smi {field} is invalid")
    return errors


def trusted_nvidia_smi_metadata():
    """Fingerprint the audited root-owned executable before invoking it."""

    resolved_path = os.path.realpath(TRUSTED_NVIDIA_SMI_PATH)
    try:
        metadata = os.stat(resolved_path)
        digest = hashlib.sha256()
        with open(resolved_path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(
            f"cannot inspect trusted nvidia-smi executable: {exc!r}"
        ) from exc
    result = {
        "requested_path": TRUSTED_NVIDIA_SMI_PATH,
        "resolved_path": resolved_path,
        "owner_uid": int(metadata.st_uid),
        "owner_gid": int(metadata.st_gid),
        "mode": stat.S_IMODE(metadata.st_mode),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "bytes": int(metadata.st_size),
        "sha256": digest.hexdigest(),
    }
    errors = trusted_nvidia_smi_metadata_errors(result)
    if not stat.S_ISREG(metadata.st_mode):
        errors.append("trusted nvidia-smi path is not a regular file")
    if metadata.st_mode & 0o022:
        errors.append("trusted nvidia-smi is group/world writable")
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def _current_boot_id():
    try:
        with open(
            "/proc/sys/kernel/random/boot_id",
            "r",
            encoding="utf-8",
        ) as handle:
            value = handle.read().strip()
    except OSError:
        return None
    return value or None


def _unavailable_snapshot(device_index, error, nvidia_smi_binary=None):
    return {
        "available": False,
        "error": str(error),
        "device_index": int(device_index),
        "device_uuid": None,
        "device_name": None,
        "processes": [],
        "process_count": None,
        "sampled_at_unix_seconds": time.time(),
        "sampled_at_monotonic_seconds": time.monotonic(),
        "boot_id": _current_boot_id(),
        "nvidia_smi_binary": nvidia_smi_binary,
    }


def _command_output(command):
    return subprocess.check_output(
        command,
        text=True,
        stderr=subprocess.STDOUT,
        timeout=10,
    )


def query_gpu_compute_processes(
    device_index=0,
    command_fn=None,
    *,
    include_process_name=False,
    binary_metadata_fn=None,
):
    """Return identity plus compute processes for one physical GPU index."""
    device_index = int(device_index)
    command_fn = command_fn or _command_output
    binary_metadata_fn = binary_metadata_fn or trusted_nvidia_smi_metadata
    try:
        nvidia_smi_binary = binary_metadata_fn()
    except Exception as error:
        return _unavailable_snapshot(device_index, repr(error))
    binary_errors = trusted_nvidia_smi_metadata_errors(nvidia_smi_binary)
    if binary_errors:
        return _unavailable_snapshot(
            device_index,
            "; ".join(binary_errors),
            nvidia_smi_binary,
        )
    executable = nvidia_smi_binary["resolved_path"]
    identity_command = [
        executable,
        "--id",
        str(device_index),
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader,nounits",
    ]
    process_fields = (
        "pid,process_name,used_memory"
        if include_process_name
        else "pid,used_memory"
    )
    process_command = [
        executable,
        "--id",
        str(device_index),
        f"--query-compute-apps={process_fields}",
        "--format=csv,noheader,nounits",
    ]
    sampled_at = time.time()
    sampled_at_monotonic = time.monotonic()
    boot_id = _current_boot_id()
    try:
        identity_output = command_fn(identity_command)
        process_output = command_fn(process_command)
        finished_at = time.time()
        finished_at_monotonic = time.monotonic()
    except (OSError, subprocess.SubprocessError) as error:
        return _unavailable_snapshot(
            device_index,
            repr(error),
            nvidia_smi_binary,
        )
    except Exception as error:
        return _unavailable_snapshot(
            device_index,
            repr(error),
            nvidia_smi_binary,
        )

    identity_rows = [
        line.strip() for line in identity_output.splitlines() if line.strip()
    ]
    if len(identity_rows) != 1:
        return _unavailable_snapshot(
            device_index,
            f"expected one nvidia-smi GPU identity row, found {len(identity_rows)}",
            nvidia_smi_binary,
        )
    identity_fields = [field.strip() for field in identity_rows[0].split(",", 2)]
    if len(identity_fields) != 3:
        return _unavailable_snapshot(
            device_index,
            f"unexpected nvidia-smi identity row: {identity_rows[0]!r}",
            nvidia_smi_binary,
        )
    try:
        observed_index = int(identity_fields[0])
    except ValueError:
        return _unavailable_snapshot(
            device_index,
            f"unparseable nvidia-smi GPU index: {identity_fields[0]!r}",
            nvidia_smi_binary,
        )
    device_uuid = identity_fields[1]
    device_name = identity_fields[2]
    if observed_index != device_index or not device_uuid or not device_name:
        return _unavailable_snapshot(
            device_index,
            "nvidia-smi returned an incomplete or mismatched GPU identity",
            nvidia_smi_binary,
        )

    processes = []
    for line in process_output.splitlines():
        if not line.strip():
            continue
        expected_fields = 3 if include_process_name else 2
        fields = [
            field.strip()
            for field in line.split(",", expected_fields - 1)
        ]
        if len(fields) != expected_fields:
            return _unavailable_snapshot(
                device_index,
                f"unexpected nvidia-smi compute-process row: {line!r}",
                nvidia_smi_binary,
            )
        try:
            process = {
                "host_pid": int(fields[0]),
                "used_memory_mib": int(fields[-1]),
            }
            if include_process_name:
                process["process_name"] = fields[1]
            processes.append(process)
        except ValueError:
            return _unavailable_snapshot(
                device_index,
                f"unparseable nvidia-smi compute-process row: {line!r}",
                nvidia_smi_binary,
            )
    return {
        "available": True,
        "error": None,
        "device_index": observed_index,
        "device_uuid": device_uuid,
        "device_name": device_name,
        "processes": processes,
        "process_count": len(processes),
        "sampled_at_unix_seconds": sampled_at,
        "sampled_at_monotonic_seconds": sampled_at_monotonic,
        "query_started_at_unix_seconds": sampled_at,
        "query_finished_at_unix_seconds": finished_at,
        "query_started_at_monotonic_seconds": sampled_at_monotonic,
        "query_finished_at_monotonic_seconds": finished_at_monotonic,
        "boot_id": boot_id,
        "nvidia_smi_binary": dict(nvidia_smi_binary),
    }


def _cuda_visible_devices_selects_physical_zero(value, device_uuid):
    """Accept only unambiguous logical-CUDA-0 mappings to physical GPU 0."""
    if value is None or not value.strip():
        return True
    selected = value.strip()
    return selected == "0" or selected == device_uuid


def build_pre_run_gpu_report(snapshot, cuda_visible_devices=None):
    """Build the persisted fail-closed pre-run evidence document."""
    snapshot = dict(snapshot)
    if cuda_visible_devices is None:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    errors = []
    if snapshot.get("available") is not True:
        errors.append(f"GPU query failed: {snapshot.get('error')}")
    if not _is_json_int(snapshot.get("device_index")) or snapshot.get(
        "device_index"
    ) != 0:
        errors.append(
            f"only physical GPU index 0 is supported, got {snapshot.get('device_index')!r}"
        )
    device_uuid = snapshot.get("device_uuid")
    device_name = snapshot.get("device_name")
    if not isinstance(device_uuid, str) or not device_uuid:
        errors.append("GPU UUID is missing")
    if not isinstance(device_name, str) or not device_name:
        errors.append("GPU name is missing")
    errors.extend(
        trusted_nvidia_smi_metadata_errors(
            snapshot.get("nvidia_smi_binary")
        )
    )
    process_count = snapshot.get("process_count")
    processes = snapshot.get("processes")
    if not _is_json_int(process_count) or process_count != 0 or processes != []:
        errors.append(
            "physical GPU 0 is not idle: "
            f"process_count={process_count!r} processes={processes!r}"
        )
    if not _cuda_visible_devices_selects_physical_zero(
        cuda_visible_devices, device_uuid
    ):
        errors.append(
            "CUDA_VISIBLE_DEVICES must be unset, '0', or the physical GPU 0 UUID; "
            f"got {cuda_visible_devices!r}"
        )
    valid = not errors
    return {
        "schema_version": GPU_EVIDENCE_SCHEMA_VERSION,
        "check_type": "pre_run_idle",
        "physical_device_index": 0,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": cuda_visible_devices,
        "available": snapshot.get("available") is True,
        "error": snapshot.get("error"),
        "device_index": snapshot.get("device_index"),
        "device_uuid": device_uuid,
        "device_name": device_name,
        "processes": processes if isinstance(processes, list) else [],
        "process_count": process_count,
        "sampled_at_unix_seconds": snapshot.get("sampled_at_unix_seconds"),
        "sampled_at_monotonic_seconds": snapshot.get(
            "sampled_at_monotonic_seconds"
        ),
        "boot_id": snapshot.get("boot_id"),
        "run_nonce": secrets.token_hex(16),
        "nvidia_smi_binary": snapshot.get("nvidia_smi_binary"),
        "idle": (
            _is_json_int(process_count)
            and process_count == 0
            and processes == []
        ),
        "valid_for_run": valid,
        "errors": errors,
    }


def validate_pre_run_gpu_report(report, cuda_visible_devices=None):
    """Return all reasons a persisted pre-run report is not safe to use."""
    errors = []
    if not isinstance(report, dict):
        return ["pre-run GPU report must be a JSON object"]
    if not _is_json_int(report.get("schema_version")) or report.get(
        "schema_version"
    ) != GPU_EVIDENCE_SCHEMA_VERSION:
        errors.append("unsupported pre-run GPU evidence schema")
    if report.get("check_type") != "pre_run_idle":
        errors.append("pre-run GPU evidence has the wrong check type")
    if (
        not _is_json_int(report.get("physical_device_index"))
        or report.get("physical_device_index") != 0
        or not _is_json_int(report.get("device_index"))
        or report.get("device_index") != 0
    ):
        errors.append("pre-run GPU evidence must target physical GPU index 0")
    if report.get("available") is not True:
        errors.append("pre-run GPU identity/process query was unavailable")
    if not isinstance(report.get("device_uuid"), str) or not report.get(
        "device_uuid"
    ):
        errors.append("pre-run GPU UUID is missing")
    if not isinstance(report.get("device_name"), str) or not report.get(
        "device_name"
    ):
        errors.append("pre-run GPU name is missing")
    if (
        not _is_json_int(report.get("process_count"))
        or report.get("process_count") != 0
        or report.get("processes") != []
    ):
        errors.append("pre-run GPU evidence is not idle")
    if report.get("idle") is not True or report.get("valid_for_run") is not True:
        errors.append("pre-run GPU evidence was not approved for the run")
    if report.get("errors") != [] or report.get("error") is not None:
        errors.append("pre-run GPU evidence contains collection/validation errors")
    errors.extend(
        trusted_nvidia_smi_metadata_errors(
            report.get("nvidia_smi_binary")
        )
    )
    for label, value in (
        ("wall-clock", report.get("sampled_at_unix_seconds")),
        ("monotonic", report.get("sampled_at_monotonic_seconds")),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            errors.append(f"pre-run GPU {label} timestamp is invalid")
    boot_id = report.get("boot_id")
    if (
        not isinstance(boot_id, str)
        or re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            boot_id,
        )
        is None
    ):
        errors.append("pre-run GPU boot ID is invalid")
    run_nonce = report.get("run_nonce")
    if (
        not isinstance(run_nonce, str)
        or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None
    ):
        errors.append("pre-run GPU nonce is invalid")
    checked_at_utc = report.get("checked_at_utc")
    try:
        checked_at = datetime.fromisoformat(checked_at_utc)
    except (TypeError, ValueError):
        checked_at = None
    if checked_at is None or checked_at.tzinfo is None:
        errors.append("pre-run GPU checked_at_utc timestamp is invalid")
    expected_visible = (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None
        else cuda_visible_devices
    )
    if report.get("cuda_visible_devices") != expected_visible:
        errors.append("CUDA_VISIBLE_DEVICES changed after the pre-run GPU check")
    if not _cuda_visible_devices_selects_physical_zero(
        expected_visible, report.get("device_uuid")
    ):
        errors.append("CUDA_VISIBLE_DEVICES does not select physical GPU 0")
    return errors


class GpuProcessMonitor:
    """Poll one physical GPU while exactly one generation process is active."""

    def __init__(self, device_index=0, interval_seconds=5.0, sample_fn=None):
        interval_seconds = float(interval_seconds)
        if interval_seconds <= 0:
            raise ValueError("GPU process monitor interval must be positive")
        if not _is_json_int(device_index):
            raise ValueError("GPU process monitor device index must be an integer")
        self.device_index = int(device_index)
        if self.device_index != 0:
            raise ValueError("GPU process evidence currently supports physical GPU 0 only")
        self.interval_seconds = interval_seconds
        self.sample_fn = sample_fn or query_gpu_compute_processes
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    def _sample_once(self):
        try:
            sample = self.sample_fn(self.device_index)
        except Exception as error:
            sample = _unavailable_snapshot(self.device_index, repr(error))
        self._samples.append(dict(sample))

    def _poll(self):
        while not self._stop.wait(self.interval_seconds):
            self._sample_once()

    def __enter__(self):
        self._sample_once()
        self._thread = threading.Thread(
            target=self._poll,
            name="fasta2v-gpu-process-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds, 1.0) + 1.0)
        self._sample_once()

    def summary(self):
        samples = [dict(sample) for sample in self._samples]
        available = [sample for sample in samples if sample.get("available") is True]
        unavailable = [sample for sample in samples if sample.get("available") is not True]
        first_nvidia_smi_binary = (
            samples[0].get("nvidia_smi_binary") if samples else None
        )
        nvidia_smi_binary_validation_errors = []
        for sample_index, sample in enumerate(samples):
            if sample.get("available") is not True:
                continue
            nvidia_smi_binary_validation_errors.extend(
                f"samples[{sample_index}]: {error}"
                for error in trusted_nvidia_smi_metadata_errors(
                    sample.get("nvidia_smi_binary")
                )
            )
        nvidia_smi_binary_fixed_valid = (
            bool(available) and not nvidia_smi_binary_validation_errors
        )
        nvidia_smi_binary_consistent = (
            bool(available)
            and first_nvidia_smi_binary is not None
            and all(
                sample.get("nvidia_smi_binary")
                == first_nvidia_smi_binary
                for sample in available
            )
        )
        process_counts = []
        over_subscribed = []
        empty = []
        sample_validation_errors = []
        distinct_pid_values = set()
        for sample_index, sample in enumerate(samples):
            if sample.get("available") is not True:
                continue
            if (
                not _is_json_int(sample.get("device_index"))
                or sample.get("device_index") != self.device_index
            ):
                sample_validation_errors.append(
                    f"samples[{sample_index}]: device_index must be integer "
                    f"{self.device_index}"
                )
            process_count = sample.get("process_count")
            if not _is_json_int(process_count) or process_count < 0:
                sample_validation_errors.append(
                    f"samples[{sample_index}]: process_count is invalid"
                )
                continue
            process_counts.append(process_count)
            if process_count > 1:
                over_subscribed.append(sample)
            if process_count == 0:
                empty.append(sample)
            processes = sample.get("processes")
            if not isinstance(processes, list) or len(processes) != process_count:
                sample_validation_errors.append(
                    f"samples[{sample_index}]: processes do not match "
                    "process_count"
                )
                continue
            for process_index, process in enumerate(processes):
                if not isinstance(process, dict):
                    sample_validation_errors.append(
                        f"samples[{sample_index}].processes[{process_index}]: "
                        "process must be an object"
                    )
                    continue
                host_pid = process.get("host_pid")
                used_memory_mib = process.get("used_memory_mib")
                if not _is_json_int(host_pid) or host_pid <= 0:
                    sample_validation_errors.append(
                        f"samples[{sample_index}].processes[{process_index}]: "
                        "host_pid is invalid"
                    )
                else:
                    distinct_pid_values.add(host_pid)
                if not _is_json_int(used_memory_mib) or used_memory_mib <= 0:
                    sample_validation_errors.append(
                        f"samples[{sample_index}].processes[{process_index}]: "
                        "used_memory_mib is invalid"
                    )
        identities = {
            (
                sample.get("device_index"),
                sample.get("device_uuid"),
                sample.get("device_name"),
            )
            for sample in available
        }
        identity_consistent = (
            bool(available)
            and len(identities) == 1
            and _is_json_int(next(iter(identities))[0])
            and next(iter(identities))[0] == self.device_index
            and bool(next(iter(identities))[1])
            and bool(next(iter(identities))[2])
        )
        identity = next(iter(identities)) if identity_consistent else (None, None, None)
        distinct_pids = sorted(distinct_pid_values)
        exact_singleton = (
            bool(samples)
            and len(available) == len(samples)
            and len(process_counts) == len(samples)
            and all(count == 1 for count in process_counts)
            and not sample_validation_errors
        )
        single_distinct_pid = exact_singleton and len(distinct_pids) == 1
        return {
            "device_index": identity[0] if identity_consistent else self.device_index,
            "device_uuid": identity[1],
            "device_name": identity[2],
            "identity_consistent": identity_consistent,
            "nvidia_smi_binary": (
                dict(first_nvidia_smi_binary)
                if isinstance(first_nvidia_smi_binary, dict)
                else first_nvidia_smi_binary
            ),
            "nvidia_smi_binary_fixed_valid": nvidia_smi_binary_fixed_valid,
            "nvidia_smi_binary_consistent": nvidia_smi_binary_consistent,
            "nvidia_smi_binary_validation_errors": (
                nvidia_smi_binary_validation_errors
            ),
            "sample_validation_errors": sample_validation_errors,
            "interval_seconds": self.interval_seconds,
            "sample_count": len(samples),
            "available_sample_count": len(available),
            "unavailable_sample_count": len(unavailable),
            "min_process_count": min(process_counts, default=None),
            "max_process_count": max(process_counts, default=None),
            "distinct_host_pids": distinct_pids,
            "single_distinct_host_pid": single_distinct_pid,
            "exact_singleton_process_per_sample": exact_singleton,
            "contention_detected": bool(over_subscribed),
            "no_process_detected": bool(empty),
            "valid_for_benchmark": (
                identity_consistent
                and nvidia_smi_binary_fixed_valid
                and nvidia_smi_binary_consistent
                and not sample_validation_errors
                and exact_singleton
                and single_distinct_pid
                and not unavailable
            ),
            "first_sample": samples[0] if samples else None,
            "last_sample": samples[-1] if samples else None,
            "contention_samples": over_subscribed[:5],
            "collection_errors": [sample.get("error") for sample in unavailable[:5]],
            "samples": samples,
        }
