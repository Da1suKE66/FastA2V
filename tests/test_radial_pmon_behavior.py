import importlib.util
import io
from datetime import datetime, timezone
from pathlib import Path
import signal
import sys
import types
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "radial_flashinfer_microtest.py"
SPEC = importlib.util.spec_from_file_location(
    "radial_flashinfer_microtest_behavior_under_test", SCRIPT_PATH
)
MICROTEST = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MICROTEST


class StubRadialAttentionDependencyError(RuntimeError):
    pass


MODULES_STUB = types.ModuleType("ovi.modules")
BACKEND_STUB = types.ModuleType("ovi.modules.radial_attention_backend")
BACKEND_STUB.RadialAttentionDependencyError = StubRadialAttentionDependencyError
BACKEND_STUB.RadialVideoSelfAttentionBackend = object
for name in (
    "load_flashinfer_api",
    "load_official_radial_mask_module",
    "verify_radial_install_receipt",
    "verify_radial_runtime_loaded_dependencies",
    "verify_radial_runtime_loader_environment",
):
    setattr(BACKEND_STUB, name, lambda *args, **kwargs: None)
with mock.patch.dict(
    sys.modules,
    {
        "ovi.modules": MODULES_STUB,
        "ovi.modules.radial_attention_backend": BACKEND_STUB,
    },
):
    SPEC.loader.exec_module(MICROTEST)


class FakePmonProcess:
    def __init__(self, *, pid=9001, return_code=None):
        self.pid = pid
        self.return_code = return_code
        self.stdout = None
        self.stderr = None

    def poll(self):
        return self.return_code

    def terminate(self):
        self.return_code = -signal.SIGTERM

    def wait(self, timeout=None):
        return self.return_code

    def kill(self):
        self.return_code = -signal.SIGKILL


def pmon_line(source_timestamp, *, host_pid=None, process_type=None):
    source = datetime.fromtimestamp(source_timestamp, tz=timezone.utc)
    pid = "-" if host_pid is None else str(host_pid)
    process = "-" if process_type is None else process_type
    command = "-" if host_pid is None else "python"
    return (
        f"{source:%Y%m%d %H:%M:%S} 0 {pid} {process} "
        f"- - - - {command}\n"
    )


class ContinuousPmonBehaviorTests(unittest.TestCase):
    host_pid = 4242
    header = "#Date Time gpu pid type sm mem enc dec command\n"

    def make_monitor(self):
        metadata = {
            "requested_path": "/usr/bin/nvidia-smi",
            "resolved_path": "/usr/bin/nvidia-smi",
        }
        with mock.patch.object(
            MICROTEST,
            "trusted_nvidia_smi_metadata",
            return_value=metadata,
        ):
            monitor = MICROTEST._ContinuousPmon(0)
        monitor.process = FakePmonProcess()
        monitor._spawn_started = (99.0, 99.0)
        monitor._process_started = (99.1, 99.1)
        return monitor

    def record(self, monitor, raw_line, received_at):
        monitor._record_stream_line(
            "stdout",
            raw_line,
            (float(received_at), float(received_at)),
        )

    def prepare_window(self, monitor):
        self.record(monitor, self.header, 100.0)
        self.record(monitor, pmon_line(101.0), 101.1)
        self.record(monitor, pmon_line(102.0), 102.1)
        monitor.wait_until_ready()
        with mock.patch.object(
            monitor, "_now", return_value=(102.15, 102.15)
        ):
            monitor.bind_expected_host_pid(self.host_pid)
        with mock.patch.object(
            monitor, "_now", return_value=(102.2, 102.2)
        ):
            monitor.begin_backend_window()

    def stop(self, monitor, at):
        with mock.patch.object(
            monitor,
            "_now",
            side_effect=[(float(at), float(at)), (float(at + 0.1), float(at + 0.1))],
        ):
            monitor.stop()

    def test_direct_c_records_bound_window_and_final_sync_tail(self):
        monitor = self.make_monitor()
        self.prepare_window(monitor)
        self.record(
            monitor,
            pmon_line(103.0, host_pid=self.host_pid, process_type="C"),
            103.1,
        )
        self.assertTrue(monitor.window_compute_seen())
        self.record(monitor, pmon_line(104.0), 104.1)
        monitor.wait_for_final_sync_coverage(103.5, 103.5)
        self.stop(monitor, 104.2)

        evidence = monitor.evidence()
        self.assertEqual(evidence["status"], "ok", evidence["errors"])
        self.assertEqual(evidence["observation_mode"], "direct_c_observed")
        self.assertTrue(evidence["direct_compute_type_observed"])
        self.assertEqual(evidence["window_compute_line_number"], 4)
        self.assertEqual(evidence["final_sync_covered_line_number"], 5)

    def test_complete_twenty_second_all_idle_window_is_degraded_not_direct(self):
        monitor = self.make_monitor()
        self.prepare_window(monitor)
        for timestamp in range(103, 124):
            self.record(
                monitor,
                pmon_line(float(timestamp)),
                float(timestamp) + 0.1,
            )
        with mock.patch.object(
            monitor, "_now", return_value=(123.2, 123.2)
        ):
            monitor.mark_compute_observation_deadline_reached()
        self.record(monitor, pmon_line(124.0), 124.1)
        monitor.wait_for_final_sync_coverage(123.5, 123.5)
        self.stop(monitor, 124.2)

        evidence = monitor.evidence()
        self.assertEqual(evidence["status"], "degraded", evidence["errors"])
        self.assertEqual(
            evidence["observation_mode"],
            "pmon_reported_all_idle_during_audited_window",
        )
        self.assertFalse(evidence["direct_compute_type_observed"])
        self.assertIsNone(evidence["window_compute_line_number"])

    def test_large_pmon_gap_fails_producer_evidence(self):
        monitor = self.make_monitor()
        self.prepare_window(monitor)
        self.record(monitor, pmon_line(103.0), 103.1)
        self.record(monitor, pmon_line(107.0), 107.1)
        self.stop(monitor, 107.2)

        evidence = monitor.evidence()
        self.assertEqual(evidence["status"], "failed")
        self.assertTrue(
            any("fixed 1-second" in error for error in evidence["errors"]),
            evidence["errors"],
        )

    def test_stdout_eof_while_process_runs_is_an_interruption(self):
        monitor = self.make_monitor()
        monitor._read_stream("stdout", io.StringIO(self.header))

        with self.assertRaisesRegex(
            MICROTEST.RadialAttentionDependencyError,
            "stdout ended while the process was still running",
        ):
            monitor.require_running("interrupted stream test")
        self.stop(monitor, 101.0)
        evidence = monitor.evidence()
        self.assertEqual(evidence["status"], "failed")

    def test_c_observed_only_after_all_idle_deadline_is_rejected(self):
        monitor = self.make_monitor()
        self.prepare_window(monitor)
        for timestamp in range(103, 124):
            self.record(
                monitor,
                pmon_line(float(timestamp)),
                float(timestamp) + 0.1,
            )
        with mock.patch.object(
            monitor, "_now", return_value=(123.2, 123.2)
        ):
            monitor.mark_compute_observation_deadline_reached()
        self.record(
            monitor,
            pmon_line(124.0, host_pid=self.host_pid, process_type="C"),
            124.1,
        )
        self.stop(monitor, 124.2)

        evidence = monitor.evidence()
        self.assertEqual(evidence["status"], "failed")
        self.assertIsNone(evidence["window_compute_line_number"])
        self.assertTrue(
            any("after the bounded all-idle" in error for error in evidence["errors"]),
            evidence["errors"],
        )

    def test_missing_post_sync_source_dt_tail_times_out_and_fails_evidence(self):
        monitor = self.make_monitor()
        self.prepare_window(monitor)
        self.record(
            monitor,
            pmon_line(103.0, host_pid=self.host_pid, process_type="C"),
            103.1,
        )
        with mock.patch.object(
            MICROTEST, "PMON_FINAL_COVERAGE_TIMEOUT_SECONDS", 0.0
        ):
            with self.assertRaisesRegex(
                MICROTEST.RadialAttentionDependencyError,
                "timed out during final CUDA sync coverage",
            ):
                monitor.wait_for_final_sync_coverage(103.5, 103.5)
        self.stop(monitor, 103.6)

        evidence = monitor.evidence()
        self.assertEqual(evidence["status"], "failed")
        self.assertTrue(evidence["timed_out"])

    def test_stderr_and_abnormal_exit_are_both_fail_closed(self):
        monitor = self.make_monitor()
        self.prepare_window(monitor)
        monitor._record_stream_line(
            "stderr", "driver warning\n", (102.3, 102.3)
        )
        monitor.process.return_code = 7
        with self.assertRaisesRegex(
            MICROTEST.RadialAttentionDependencyError,
            "exited early.*7",
        ):
            monitor.require_running("abnormal exit test")
        self.stop(monitor, 102.4)

        evidence = monitor.evidence()
        self.assertEqual(evidence["status"], "failed")
        self.assertEqual(evidence["raw_stderr"], "driver warning\n")
        self.assertEqual(evidence["exit_code"], 7)
        self.assertEqual(evidence["termination_method"], "unexpected_early_exit")


if __name__ == "__main__":
    unittest.main()
