from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ovi.gpu_process_monitor import (
    GpuProcessMonitor,
    TRUSTED_NVIDIA_SMI_BYTES,
    TRUSTED_NVIDIA_SMI_PATH,
    TRUSTED_NVIDIA_SMI_SHA256,
    build_pre_run_gpu_report,
    query_gpu_compute_processes,
    validate_pre_run_gpu_report,
)
from scripts.check_pre_run_gpu import main as check_pre_run_gpu


GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"
GPU_NAME = "NVIDIA A100-SXM4-80GB"


def trusted_binary():
    return {
        "requested_path": TRUSTED_NVIDIA_SMI_PATH,
        "resolved_path": TRUSTED_NVIDIA_SMI_PATH,
        "owner_uid": 0,
        "owner_gid": 0,
        "mode": 0o755,
        "device": 2050,
        "inode": 2490545,
        "bytes": TRUSTED_NVIDIA_SMI_BYTES,
        "sha256": TRUSTED_NVIDIA_SMI_SHA256,
    }


def sample(count, *, available=True, uuid=GPU_UUID, index=0, pid_base=100):
    return {
        "available": available,
        "error": None if available else "synthetic query failure",
        "device_index": index,
        "device_uuid": uuid if available else None,
        "device_name": GPU_NAME if available else None,
        "processes": [
            {"host_pid": pid_base + offset, "used_memory_mib": 1000}
            for offset in range(count)
        ],
        "process_count": count if available else None,
        "sampled_at_unix_seconds": 0.0,
        "sampled_at_monotonic_seconds": 1.0,
        "boot_id": "11111111-2222-3333-4444-555555555555",
        "nvidia_smi_binary": trusted_binary(),
    }


class GpuQueryTests(unittest.TestCase):
    def test_query_binds_identity_and_processes(self):
        outputs = iter((
            f"0, {GPU_UUID}, {GPU_NAME}\n",
            "4123, 2048\n",
        ))
        snapshot = query_gpu_compute_processes(
            0,
            command_fn=lambda _command: next(outputs),
            binary_metadata_fn=trusted_binary,
        )
        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["device_index"], 0)
        self.assertEqual(snapshot["device_uuid"], GPU_UUID)
        self.assertEqual(snapshot["device_name"], GPU_NAME)
        self.assertEqual(
            snapshot["processes"],
            [{"host_pid": 4123, "used_memory_mib": 2048}],
        )

    def test_malformed_identity_is_fail_closed(self):
        snapshot = query_gpu_compute_processes(
            0,
            command_fn=lambda _command: "not-an-identity-row\n",
            binary_metadata_fn=trusted_binary,
        )
        self.assertFalse(snapshot["available"])
        self.assertIsNone(snapshot["process_count"])

    def test_optional_process_name_is_preserved_for_mps_audits(self):
        commands = []
        outputs = iter((
            f"0, {GPU_UUID}, {GPU_NAME}\n",
            "4123, /fixed/python, 2048\n",
        ))

        def command_fn(command):
            commands.append(command)
            return next(outputs)

        snapshot = query_gpu_compute_processes(
            0,
            command_fn=command_fn,
            include_process_name=True,
            binary_metadata_fn=trusted_binary,
        )
        self.assertEqual(
            snapshot["processes"],
            [
                {
                    "host_pid": 4123,
                    "process_name": "/fixed/python",
                    "used_memory_mib": 2048,
                }
            ],
        )
        self.assertIn(
            "--query-compute-apps=pid,process_name,used_memory",
            commands[1],
        )


class PreRunGpuEvidenceTests(unittest.TestCase):
    def test_idle_physical_zero_is_valid(self):
        report = build_pre_run_gpu_report(
            sample(0), cuda_visible_devices="0"
        )
        self.assertTrue(report["valid_for_run"])
        self.assertEqual(report["processes"], [])
        self.assertEqual(
            validate_pre_run_gpu_report(report, cuda_visible_devices="0"), []
        )

    def test_existing_process_is_fail_closed(self):
        report = build_pre_run_gpu_report(
            sample(1), cuda_visible_devices="0"
        )
        self.assertFalse(report["valid_for_run"])
        self.assertTrue(any("not idle" in error for error in report["errors"]))

    def test_ambiguous_cuda_mapping_is_rejected(self):
        report = build_pre_run_gpu_report(
            sample(0), cuda_visible_devices="1,0"
        )
        self.assertFalse(report["valid_for_run"])
        self.assertTrue(
            any("CUDA_VISIBLE_DEVICES" in error for error in report["errors"])
        )

    def test_nonfinite_time_or_malformed_boot_id_is_rejected(self):
        report = build_pre_run_gpu_report(
            sample(0), cuda_visible_devices="0"
        )
        report["sampled_at_monotonic_seconds"] = float("nan")
        report["boot_id"] = "-" * 36
        errors = validate_pre_run_gpu_report(
            report,
            cuda_visible_devices="0",
        )
        self.assertTrue(any("monotonic" in error for error in errors))
        self.assertTrue(any("boot ID" in error for error in errors))

    def test_untrusted_nvidia_smi_binary_is_rejected(self):
        report = build_pre_run_gpu_report(
            sample(0), cuda_visible_devices="0"
        )
        report["nvidia_smi_binary"]["sha256"] = "0" * 64
        errors = validate_pre_run_gpu_report(
            report,
            cuda_visible_devices="0",
        )
        self.assertTrue(any("nvidia-smi" in error for error in errors))

    def test_untrusted_snapshot_is_never_marked_valid_for_run(self):
        snapshot = sample(0)
        snapshot["nvidia_smi_binary"]["sha256"] = "0" * 64
        report = build_pre_run_gpu_report(
            snapshot,
            cuda_visible_devices="0",
        )
        self.assertFalse(report["valid_for_run"])
        self.assertTrue(any("nvidia-smi" in error for error in report["errors"]))

    def test_cli_persists_failure_before_returning_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pre_run_gpu.json"
            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    status = check_pre_run_gpu(
                        output,
                        sample_fn=lambda _index: sample(2),
                    )
            persisted = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(status, 1)
        self.assertFalse(persisted["valid_for_run"])
        self.assertEqual(persisted["process_count"], 2)


class GpuProcessMonitorTests(unittest.TestCase):
    def test_every_sample_must_have_exactly_one_process(self):
        snapshots = iter((sample(1, pid_base=321), sample(1, pid_base=321)))
        monitor = GpuProcessMonitor(sample_fn=lambda _device: next(snapshots))
        monitor._sample_once()
        monitor._sample_once()
        summary = monitor.summary()
        self.assertTrue(summary["valid_for_benchmark"])
        self.assertTrue(summary["identity_consistent"])
        self.assertEqual(summary["nvidia_smi_binary"], trusted_binary())
        self.assertTrue(summary["nvidia_smi_binary_fixed_valid"])
        self.assertTrue(summary["nvidia_smi_binary_consistent"])
        self.assertEqual(summary["nvidia_smi_binary_validation_errors"], [])
        self.assertEqual(summary["sample_validation_errors"], [])
        self.assertTrue(summary["exact_singleton_process_per_sample"])
        self.assertEqual(summary["min_process_count"], 1)
        self.assertEqual(summary["max_process_count"], 1)
        self.assertEqual(summary["distinct_host_pids"], [321])
        self.assertTrue(summary["single_distinct_host_pid"])
        self.assertEqual(len(summary["samples"]), 2)

    def test_zero_process_sample_is_invalid(self):
        snapshots = iter((sample(1), sample(0)))
        monitor = GpuProcessMonitor(sample_fn=lambda _device: next(snapshots))
        monitor._sample_once()
        monitor._sample_once()
        summary = monitor.summary()
        self.assertFalse(summary["valid_for_benchmark"])
        self.assertTrue(summary["no_process_detected"])
        self.assertEqual(summary["min_process_count"], 0)

    def test_second_process_marks_contention(self):
        snapshots = iter((sample(1), sample(2), sample(1)))
        monitor = GpuProcessMonitor(sample_fn=lambda _device: next(snapshots))
        for _ in range(3):
            monitor._sample_once()
        summary = monitor.summary()
        self.assertFalse(summary["valid_for_benchmark"])
        self.assertTrue(summary["contention_detected"])
        self.assertEqual(summary["max_process_count"], 2)
        self.assertEqual(len(summary["contention_samples"]), 1)

    def test_identity_drift_is_fail_closed(self):
        snapshots = iter((sample(1), sample(1, uuid="GPU-different")))
        monitor = GpuProcessMonitor(sample_fn=lambda _device: next(snapshots))
        monitor._sample_once()
        monitor._sample_once()
        summary = monitor.summary()
        self.assertFalse(summary["valid_for_benchmark"])
        self.assertFalse(summary["identity_consistent"])
        self.assertIsNone(summary["device_uuid"])

    def test_collection_failure_is_fail_closed(self):
        snapshots = iter((sample(1), sample(0, available=False)))
        monitor = GpuProcessMonitor(sample_fn=lambda _device: next(snapshots))
        monitor._sample_once()
        monitor._sample_once()
        summary = monitor.summary()
        self.assertFalse(summary["valid_for_benchmark"])
        self.assertEqual(summary["unavailable_sample_count"], 1)

    def test_nvidia_smi_metadata_drift_is_fail_closed(self):
        first = sample(1, pid_base=321)
        second = sample(1, pid_base=321)
        second["nvidia_smi_binary"]["inode"] += 1
        snapshots = iter((first, second))
        monitor = GpuProcessMonitor(sample_fn=lambda _device: next(snapshots))
        monitor._sample_once()
        monitor._sample_once()
        summary = monitor.summary()
        self.assertFalse(summary["valid_for_benchmark"])
        self.assertTrue(summary["nvidia_smi_binary_fixed_valid"])
        self.assertFalse(summary["nvidia_smi_binary_consistent"])

    def test_untrusted_sample_binary_is_fail_closed(self):
        first = sample(1, pid_base=321)
        second = sample(1, pid_base=321)
        second["nvidia_smi_binary"]["sha256"] = "0" * 64
        snapshots = iter((first, second))
        monitor = GpuProcessMonitor(sample_fn=lambda _device: next(snapshots))
        monitor._sample_once()
        monitor._sample_once()
        summary = monitor.summary()
        self.assertFalse(summary["valid_for_benchmark"])
        self.assertFalse(summary["nvidia_smi_binary_fixed_valid"])
        self.assertTrue(summary["nvidia_smi_binary_validation_errors"])

    def test_bool_forged_sample_integers_are_fail_closed(self):
        mutations = (
            ("device_index", lambda item: item.__setitem__("device_index", False)),
            ("process_count", lambda item: item.__setitem__("process_count", True)),
            (
                "host_pid",
                lambda item: item["processes"][0].__setitem__("host_pid", True),
            ),
            (
                "used_memory_mib",
                lambda item: item["processes"][0].__setitem__(
                    "used_memory_mib", True
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                first = sample(1, pid_base=321)
                second = sample(1, pid_base=321)
                mutate(second)
                snapshots = iter((first, second))
                monitor = GpuProcessMonitor(
                    sample_fn=lambda _device: next(snapshots)
                )
                monitor._sample_once()
                monitor._sample_once()
                summary = monitor.summary()
                self.assertFalse(summary["valid_for_benchmark"])
                self.assertTrue(summary["sample_validation_errors"])

    def test_nonzero_physical_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "physical GPU 0"):
            GpuProcessMonitor(device_index=1)

    def test_bool_physical_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            GpuProcessMonitor(device_index=False)

    def test_invalid_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "interval"):
            GpuProcessMonitor(interval_seconds=0)


if __name__ == "__main__":
    unittest.main()
