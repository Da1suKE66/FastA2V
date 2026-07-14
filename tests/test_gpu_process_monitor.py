from contextlib import redirect_stderr, redirect_stdout
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ovi.gpu_process_monitor import (
    GPU_EVIDENCE_SCHEMA_VERSION,
    GPU_PROCESS_MONITOR_SCHEMA_VERSION,
    GpuProcessMonitor,
    TRUSTED_NVIDIA_SMI_BYTES,
    TRUSTED_NVIDIA_SMI_PATH,
    TRUSTED_NVIDIA_SMI_SHA256,
    build_pre_run_gpu_report,
    gpu_compute_snapshot_errors,
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
    if available:
        process_output = "".join(
            f"{pid_base + offset}, 1000\n" for offset in range(count)
        )
        outputs = iter((
            f"{index}, {uuid}, {GPU_NAME}\n",
            process_output,
        ))
        command_fn = lambda _command: next(outputs)
    else:
        def command_fn(_command):
            raise OSError("synthetic query failure")

    snapshot = query_gpu_compute_processes(
        index,
        command_fn=command_fn,
        binary_metadata_fn=trusted_binary,
    )
    snapshot["boot_id"] = "11111111-2222-3333-4444-555555555555"
    return snapshot


def shift_snapshot_times(snapshot, delta):
    shifted = copy.deepcopy(snapshot)
    receipt = shifted["query_receipt"]
    for clock in ("unix", "monotonic"):
        for prefix in ("sampled_at", "query_started_at", "query_finished_at"):
            field = f"{prefix}_{clock}_seconds"
            if field in shifted:
                shifted[field] += delta
        for prefix in ("query_started_at", "query_finished_at"):
            field = f"{prefix}_{clock}_seconds"
            receipt[field] += delta
        for command in receipt["commands"]:
            for prefix in ("started_at", "finished_at"):
                field = f"{prefix}_{clock}_seconds"
                command[field] += delta
    return shifted


class GpuQueryTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            "ovi.gpu_process_monitor._current_boot_id",
            return_value="11111111-2222-3333-4444-555555555555",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

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
        self.assertEqual(gpu_compute_snapshot_errors(snapshot), [])
        receipt = snapshot["query_receipt"]
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(receipt["locale"], {"LANG": "C", "LC_ALL": "C"})
        self.assertEqual(receipt["resolved_executable"], TRUSTED_NVIDIA_SMI_PATH)
        self.assertEqual(len(receipt["commands"]), 2)
        self.assertEqual(
            receipt["commands"][0]["raw_stdout"],
            f"0, {GPU_UUID}, {GPU_NAME}\n",
        )
        self.assertEqual(receipt["commands"][0]["raw_stderr"], "")
        self.assertEqual(receipt["commands"][0]["exit_code"], 0)
        self.assertGreater(receipt["commands"][0]["raw_stdout_bytes"], 0)

    def test_malformed_identity_is_fail_closed(self):
        snapshot = query_gpu_compute_processes(
            0,
            command_fn=lambda _command: "not-an-identity-row\n",
            binary_metadata_fn=trusted_binary,
        )
        self.assertFalse(snapshot["available"])
        self.assertIsNone(snapshot["process_count"])
        self.assertEqual(snapshot["query_receipt"]["status"], "parse_failed")
        self.assertEqual(gpu_compute_snapshot_errors(snapshot), [])

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
        self.assertEqual(gpu_compute_snapshot_errors(snapshot), [])

    def test_raw_stderr_and_exit_status_are_preserved_exactly(self):
        outputs = iter((
            (f"0, {GPU_UUID}, {GPU_NAME}\n", "identity warning\n"),
        ))

        def command_fn(command):
            stdout, stderr = next(outputs)
            return __import__("subprocess").CompletedProcess(
                command,
                0,
                stdout=stdout.encode("utf-8"),
                stderr=stderr.encode("utf-8"),
            )

        snapshot = query_gpu_compute_processes(
            0,
            command_fn=command_fn,
            binary_metadata_fn=trusted_binary,
        )
        commands = snapshot["query_receipt"]["commands"]
        self.assertEqual(commands[0]["raw_stderr"], "identity warning\n")
        self.assertFalse(commands[1]["attempted"])
        self.assertFalse(snapshot["available"])
        self.assertEqual(
            snapshot["query_receipt"]["status"],
            "identity_command_failed",
        )
        self.assertEqual(gpu_compute_snapshot_errors(snapshot), [])

    def test_failed_command_retains_auditable_unavailable_receipt(self):
        snapshot = query_gpu_compute_processes(
            0,
            command_fn=lambda _command: (_ for _ in ()).throw(
                OSError("synthetic failure")
            ),
            binary_metadata_fn=trusted_binary,
        )
        receipt = snapshot["query_receipt"]
        self.assertFalse(snapshot["available"])
        self.assertEqual(receipt["status"], "identity_command_failed")
        self.assertTrue(receipt["commands"][0]["attempted"])
        self.assertIsNotNone(receipt["commands"][0]["execution_error"])
        self.assertFalse(receipt["commands"][1]["attempted"])
        self.assertEqual(receipt["commands"][1]["raw_stdout_bytes"], 0)
        self.assertEqual(gpu_compute_snapshot_errors(snapshot), [])

    def test_binary_failure_marks_both_commands_not_run(self):
        snapshot = query_gpu_compute_processes(
            0,
            binary_metadata_fn=lambda: (_ for _ in ()).throw(
                OSError("missing trusted binary")
            ),
        )
        receipt = snapshot["query_receipt"]
        self.assertEqual(receipt["status"], "binary_metadata_unavailable")
        self.assertEqual(
            [item["attempted"] for item in receipt["commands"]],
            [False, False],
        )
        self.assertTrue(
            all(item["exit_code"] is None for item in receipt["commands"])
        )
        self.assertTrue(gpu_compute_snapshot_errors(snapshot))

    def test_canonical_validator_rejects_receipt_and_derived_tampering(self):
        base = sample(1, pid_base=4123)
        mutations = (
            (
                "argv",
                lambda item: item["query_receipt"]["commands"][0][
                    "command"
                ].__setitem__(0, "/tmp/nvidia-smi"),
            ),
            (
                "locale",
                lambda item: item["query_receipt"].__setitem__(
                    "locale", {"LANG": "en_US.UTF-8", "LC_ALL": "C"}
                ),
            ),
            (
                "raw-stdout",
                lambda item: item["query_receipt"]["commands"][0].__setitem__(
                    "raw_stdout", "forged\n"
                ),
            ),
            (
                "byte-count-bool",
                lambda item: item["query_receipt"]["commands"][0].__setitem__(
                    "raw_stdout_bytes", True
                ),
            ),
            (
                "exit-code-bool",
                lambda item: item["query_receipt"]["commands"][0].__setitem__(
                    "exit_code", False
                ),
            ),
            (
                "derived-process-count-bool",
                lambda item: item.__setitem__("process_count", True),
            ),
            (
                "derived-pid-bool",
                lambda item: item["processes"][0].__setitem__(
                    "host_pid", True
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                snapshot = copy.deepcopy(base)
                mutate(snapshot)
                self.assertTrue(gpu_compute_snapshot_errors(snapshot))

    def test_canonical_validator_reparses_self_consistent_raw_bytes(self):
        snapshot = sample(1, pid_base=4123)
        command = snapshot["query_receipt"]["commands"][1]
        forged = b"9999, 2048\n"
        command["raw_stdout"] = forged.decode("utf-8")
        command["raw_stdout_base64"] = base64.b64encode(forged).decode("ascii")
        command["raw_stdout_bytes"] = len(forged)
        command["raw_stdout_sha256"] = hashlib.sha256(forged).hexdigest()
        errors = gpu_compute_snapshot_errors(snapshot)
        self.assertTrue(any("process fields" in error for error in errors))

    def test_canonical_validator_rejects_forged_success_with_stderr(self):
        snapshot = sample(1, pid_base=4123)
        command = snapshot["query_receipt"]["commands"][0]
        forged = b"warning\n"
        command["raw_stderr"] = forged.decode("utf-8")
        command["raw_stderr_base64"] = base64.b64encode(forged).decode("ascii")
        command["raw_stderr_bytes"] = len(forged)
        command["raw_stderr_sha256"] = hashlib.sha256(forged).hexdigest()
        errors = gpu_compute_snapshot_errors(snapshot)
        self.assertTrue(any("stderr/exit" in error for error in errors))


class PreRunGpuEvidenceTests(unittest.TestCase):
    def test_idle_physical_zero_is_valid(self):
        report = build_pre_run_gpu_report(
            sample(0), cuda_visible_devices="0"
        )
        self.assertTrue(report["valid_for_run"])
        self.assertEqual(
            report["schema_version"], GPU_EVIDENCE_SCHEMA_VERSION
        )
        self.assertEqual(report["processes"], [])
        self.assertEqual(
            validate_pre_run_gpu_report(report, cuda_visible_devices="0"), []
        )

    def test_legacy_pre_run_schema_is_rejected(self):
        report = build_pre_run_gpu_report(
            sample(0), cuda_visible_devices="0"
        )
        report["schema_version"] = 1
        errors = validate_pre_run_gpu_report(
            report,
            cuda_visible_devices="0",
        )
        self.assertTrue(any("schema" in error for error in errors))

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

    def test_pre_run_validator_reparses_persisted_raw_receipt(self):
        report = build_pre_run_gpu_report(
            sample(0),
            cuda_visible_devices="0",
        )
        command = report["query_receipt"]["commands"][1]
        forged = b"9999, 2048\n"
        command["raw_stdout"] = forged.decode("utf-8")
        command["raw_stdout_base64"] = base64.b64encode(forged).decode("ascii")
        command["raw_stdout_bytes"] = len(forged)
        command["raw_stdout_sha256"] = hashlib.sha256(forged).hexdigest()
        errors = validate_pre_run_gpu_report(
            report,
            cuda_visible_devices="0",
        )
        self.assertTrue(any("process fields" in error for error in errors))

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
        self.assertEqual(
            summary["schema_version"], GPU_PROCESS_MONITOR_SCHEMA_VERSION
        )
        self.assertTrue(summary["valid_for_benchmark"])
        self.assertTrue(summary["identity_consistent"])
        self.assertEqual(summary["nvidia_smi_binary"], trusted_binary())
        self.assertTrue(summary["nvidia_smi_binary_fixed_valid"])
        self.assertTrue(summary["nvidia_smi_binary_consistent"])
        self.assertEqual(summary["nvidia_smi_binary_validation_errors"], [])
        self.assertEqual(summary["sample_validation_errors"], [])
        self.assertEqual(summary["snapshot_validation_errors"], [])
        self.assertEqual(summary["sample_sequence_validation_errors"], [])
        self.assertGreater(summary["observation_span_seconds"], 0.0)
        self.assertEqual(
            summary["boot_id"],
            "11111111-2222-3333-4444-555555555555",
        )
        self.assertTrue(summary["boot_id_consistent"])
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

    def test_boot_id_drift_is_fail_closed(self):
        first = sample(1, pid_base=321)
        second = sample(1, pid_base=321)
        second["boot_id"] = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        snapshots = iter((first, second))
        monitor = GpuProcessMonitor(sample_fn=lambda _device: next(snapshots))
        monitor._sample_once()
        monitor._sample_once()
        summary = monitor.summary()
        self.assertFalse(summary["valid_for_benchmark"])
        self.assertFalse(summary["boot_id_consistent"])

    def test_duplicate_snapshot_cannot_satisfy_entry_exit_sampling(self):
        snapshot = sample(1, pid_base=321)
        snapshots = iter((snapshot, snapshot))
        monitor = GpuProcessMonitor(sample_fn=lambda _device: next(snapshots))
        monitor._sample_once()
        monitor._sample_once()
        summary = monitor.summary()
        self.assertFalse(summary["valid_for_benchmark"])
        self.assertTrue(summary["sample_sequence_validation_errors"])

    def test_large_cadence_gap_is_fail_closed(self):
        first = sample(1, pid_base=321)
        second = shift_snapshot_times(sample(1, pid_base=321), 100.0)
        snapshots = iter((first, second))
        monitor = GpuProcessMonitor(
            interval_seconds=5.0,
            sample_fn=lambda _device: next(snapshots),
        )
        monitor._sample_once()
        monitor._sample_once()
        summary = monitor.summary()
        self.assertFalse(summary["valid_for_benchmark"])
        self.assertGreater(summary["maximum_sample_gap_seconds"], 90.0)
        self.assertTrue(
            any(
                "gap" in error
                for error in summary["sample_sequence_validation_errors"]
            )
        )

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
        with self.assertRaisesRegex(ValueError, "interval"):
            GpuProcessMonitor(interval_seconds=True)


if __name__ == "__main__":
    unittest.main()
