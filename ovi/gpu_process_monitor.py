"""Fail-closed NVIDIA GPU identity and compute-process evidence.

This module deliberately uses only the Python standard library.  The run
scripts can therefore prove that physical GPU 0 is idle before importing
PyTorch or performing any CUDA preflight/model loading.
"""

from datetime import datetime, timezone
import os
import subprocess
import threading
import time


GPU_EVIDENCE_SCHEMA_VERSION = 1


def _unavailable_snapshot(device_index, error):
    return {
        "available": False,
        "error": str(error),
        "device_index": int(device_index),
        "device_uuid": None,
        "device_name": None,
        "processes": [],
        "process_count": None,
        "sampled_at_unix_seconds": time.time(),
    }


def _command_output(command):
    return subprocess.check_output(
        command,
        text=True,
        stderr=subprocess.STDOUT,
        timeout=10,
    )


def query_gpu_compute_processes(device_index=0, command_fn=None):
    """Return identity plus compute processes for one physical GPU index."""
    device_index = int(device_index)
    command_fn = command_fn or _command_output
    identity_command = [
        "nvidia-smi",
        "--id",
        str(device_index),
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        "nvidia-smi",
        "--id",
        str(device_index),
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    sampled_at = time.time()
    try:
        identity_output = command_fn(identity_command)
        process_output = command_fn(process_command)
    except (OSError, subprocess.SubprocessError) as error:
        return _unavailable_snapshot(device_index, repr(error))
    except Exception as error:
        return _unavailable_snapshot(device_index, repr(error))

    identity_rows = [
        line.strip() for line in identity_output.splitlines() if line.strip()
    ]
    if len(identity_rows) != 1:
        return _unavailable_snapshot(
            device_index,
            f"expected one nvidia-smi GPU identity row, found {len(identity_rows)}",
        )
    identity_fields = [field.strip() for field in identity_rows[0].split(",", 2)]
    if len(identity_fields) != 3:
        return _unavailable_snapshot(
            device_index,
            f"unexpected nvidia-smi identity row: {identity_rows[0]!r}",
        )
    try:
        observed_index = int(identity_fields[0])
    except ValueError:
        return _unavailable_snapshot(
            device_index,
            f"unparseable nvidia-smi GPU index: {identity_fields[0]!r}",
        )
    device_uuid = identity_fields[1]
    device_name = identity_fields[2]
    if observed_index != device_index or not device_uuid or not device_name:
        return _unavailable_snapshot(
            device_index,
            "nvidia-smi returned an incomplete or mismatched GPU identity",
        )

    processes = []
    for line in process_output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            return _unavailable_snapshot(
                device_index,
                f"unexpected nvidia-smi compute-process row: {line!r}",
            )
        try:
            processes.append(
                {
                    "host_pid": int(fields[0]),
                    "used_memory_mib": int(fields[1]),
                }
            )
        except ValueError:
            return _unavailable_snapshot(
                device_index,
                f"unparseable nvidia-smi compute-process row: {line!r}",
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
    if snapshot.get("device_index") != 0:
        errors.append(
            f"only physical GPU index 0 is supported, got {snapshot.get('device_index')!r}"
        )
    device_uuid = snapshot.get("device_uuid")
    device_name = snapshot.get("device_name")
    if not isinstance(device_uuid, str) or not device_uuid:
        errors.append("GPU UUID is missing")
    if not isinstance(device_name, str) or not device_name:
        errors.append("GPU name is missing")
    process_count = snapshot.get("process_count")
    processes = snapshot.get("processes")
    if process_count != 0 or processes != []:
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
        "idle": process_count == 0 and processes == [],
        "valid_for_run": valid,
        "errors": errors,
    }


def validate_pre_run_gpu_report(report, cuda_visible_devices=None):
    """Return all reasons a persisted pre-run report is not safe to use."""
    errors = []
    if not isinstance(report, dict):
        return ["pre-run GPU report must be a JSON object"]
    if report.get("schema_version") != GPU_EVIDENCE_SCHEMA_VERSION:
        errors.append("unsupported pre-run GPU evidence schema")
    if report.get("check_type") != "pre_run_idle":
        errors.append("pre-run GPU evidence has the wrong check type")
    if report.get("physical_device_index") != 0 or report.get("device_index") != 0:
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
    if report.get("process_count") != 0 or report.get("processes") != []:
        errors.append("pre-run GPU evidence is not idle")
    if report.get("idle") is not True or report.get("valid_for_run") is not True:
        errors.append("pre-run GPU evidence was not approved for the run")
    if report.get("errors") != [] or report.get("error") is not None:
        errors.append("pre-run GPU evidence contains collection/validation errors")
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
        process_counts = [
            int(sample["process_count"])
            for sample in available
            if isinstance(sample.get("process_count"), int)
        ]
        over_subscribed = [
            sample for sample in available if sample.get("process_count", 0) > 1
        ]
        empty = [
            sample for sample in available if sample.get("process_count") == 0
        ]
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
            and next(iter(identities))[0] == self.device_index
            and bool(next(iter(identities))[1])
            and bool(next(iter(identities))[2])
        )
        identity = next(iter(identities)) if identity_consistent else (None, None, None)
        distinct_pids = sorted(
            {
                int(process["host_pid"])
                for sample in available
                for process in sample.get("processes", [])
                if isinstance(process, dict)
                and isinstance(process.get("host_pid"), int)
            }
        )
        exact_singleton = (
            bool(samples)
            and len(available) == len(samples)
            and len(process_counts) == len(samples)
            and all(count == 1 for count in process_counts)
        )
        single_distinct_pid = exact_singleton and len(distinct_pids) == 1
        return {
            "device_index": identity[0] if identity_consistent else self.device_index,
            "device_uuid": identity[1],
            "device_name": identity[2],
            "identity_consistent": identity_consistent,
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
