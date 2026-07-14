#!/usr/bin/env python3
"""Launch the exact pinned Radial prefix/tail protocol on real BF16 sm80."""

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ovi.gpu_process_monitor import (
    GpuProcessMonitor,
    query_gpu_compute_processes,
    trusted_nvidia_smi_metadata,
    validate_pre_run_gpu_report,
)
from ovi.modules.radial_attention_backend import (
    RadialAttentionDependencyError,
    RadialVideoSelfAttentionBackend,
    load_flashinfer_api,
    load_official_radial_mask_module,
    verify_radial_install_receipt,
    verify_radial_runtime_loaded_dependencies,
    verify_radial_runtime_loader_environment,
)
from ovi.radial_evidence import (
    RADIAL_GRID,
    RADIAL_HEAD_DIM,
    RADIAL_HEADS,
    RADIAL_MASK_API,
    RADIAL_SEQUENCE,
    parse_nvidia_smi_pmon_output,
)


class _IdentityOviAttention:
    use_sp = False
    window_size = (-1, -1)

    def __init__(self, q, k, v):
        self.q = q
        self.k = k
        self.v = v

    def qkv_fn(self, _unused_hidden):
        return self.q, self.k, self.v

    def o(self, value):
        return value


GPU_PROCESS_BINDING_SCHEMA_VERSION = 1
GPU_IDLE_GUARD_MAX_AGE_SECONDS = 600.0
PMON_READY_TIMEOUT_SECONDS = 15.0
PMON_WINDOW_TIMEOUT_SECONDS = 20.0
PMON_TERMINATE_TIMEOUT_SECONDS = 5.0
PMON_KILL_TIMEOUT_SECONDS = 5.0


class RadialPmonEvidenceError(RadialAttentionDependencyError):
    """Expose stopped pmon evidence to the outer preflight failure report."""

    def __init__(self, message, pmon_evidence):
        super().__init__(message)
        self.pmon_evidence = pmon_evidence
        self.radialattn_pmon_failure_evidence = pmon_evidence


def _current_pid_namespace_evidence():
    """Parse NSpid without turning read/format errors into a fallback."""

    local_pid = os.getpid()
    try:
        lines = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {
            "status": "read_error",
            "error": repr(exc),
            "chain": [],
        }
    for line in lines:
        if line.startswith("NSpid:"):
            try:
                chain = [int(value) for value in line.split()[1:]]
            except ValueError as exc:
                return {
                    "status": "parse_error",
                    "error": repr(exc),
                    "chain": [],
                }
            if not chain or chain[-1] != local_pid:
                return {
                    "status": "chain_mismatch",
                    "error": (
                        f"NSpid chain {chain!r} does not end in PID {local_pid}"
                    ),
                    "chain": chain,
                }
            return {"status": "ok", "error": None, "chain": chain}
    return {
        "status": "missing",
        "error": "NSpid field is missing from /proc/self/status",
        "chain": [],
    }


def _proc_pid_visibility(host_pid):
    """Distinguish an invisible host PID from permission or other errors."""

    try:
        os.stat(f"/proc/{int(host_pid)}")
    except FileNotFoundError:
        return {"status": "not_visible", "error": None}
    except PermissionError as exc:
        return {"status": "permission_error", "error": repr(exc)}
    except OSError as exc:
        return {"status": "os_error", "error": repr(exc)}
    return {"status": "visible", "error": None}


class _ContinuousPmon:
    """Audit one backend window from an idle baseline through final sync."""

    def __init__(self, device_index, expected_host_pid=None):
        self.device_index = int(device_index)
        self.expected_host_pid = (
            int(expected_host_pid) if expected_host_pid is not None else None
        )
        self.nvidia_smi_binary = trusted_nvidia_smi_metadata()
        self.resolved_executable = self.nvidia_smi_binary["resolved_path"]
        self.command = [
            "/usr/bin/nvidia-smi",
            "pmon",
            "-i",
            str(self.device_index),
            "-s",
            "um",
            "-d",
            "1",
            "-o",
            "DT",
        ]
        self.locale = {"LC_ALL": "C", "LANG": "C", "TZ": "UTC"}
        self.process = None
        self._condition = threading.Condition()
        self._stdout_line_records = []
        self._stderr_line_records = []
        self._reader_errors = []
        self._lifecycle_errors = []
        self._threads = []
        self._header_columns = None
        self._parsed_rows = []
        self._last_source_timestamp = None
        self._parser_errors = []
        self._timed_out = False
        self._termination_method = None
        self._exit_code = None
        self._spawn_started = (None, None)
        self._process_started = (None, None)
        self._header_ready = (None, None)
        self._idle_baseline_ready = (None, None)
        self._idle_baseline_line_number = None
        self._host_pid_bound = (None, None)
        self._backend_window_started = (None, None)
        self._backend_window_start_line_number = None
        self._window_compute_ready = (None, None)
        self._window_compute_line_number = None
        self._stop_requested = (None, None)
        self._process_exited = (None, None)

    @staticmethod
    def _now():
        return time.time(), time.monotonic()

    def _append_lifecycle_error(self, message):
        message = str(message)
        if message not in self._lifecycle_errors:
            self._lifecycle_errors.append(message)

    @staticmethod
    def _row_is_mps(row):
        command = str(row.get("command") or "").strip().lower()
        return "mps" in command

    @staticmethod
    def _row_has_source_dt(row):
        source_date = row.get("source_date")
        source_time = row.get("source_time")
        source_timestamp = row.get("source_timestamp_unix_seconds")
        return (
            isinstance(source_date, str)
            and bool(source_date.strip())
            and isinstance(source_time, str)
            and bool(source_time.strip())
            and isinstance(source_timestamp, (int, float))
            and not isinstance(source_timestamp, bool)
            and math.isfinite(float(source_timestamp))
            and float(source_timestamp) >= 0.0
        )

    def _row_receipt(self, row, fallback):
        line_number = row.get("line_number")
        if isinstance(line_number, int) and 0 < line_number <= len(
            self._stdout_line_records
        ):
            record = self._stdout_line_records[line_number - 1]
            return (
                record["received_at_unix_seconds"],
                record["received_at_monotonic_seconds"],
            )
        return fallback

    def _validate_bound_compute_row(self, row, row_receipt):
        host_pid = row.get("host_pid")
        if (
            not isinstance(host_pid, int)
            or isinstance(host_pid, bool)
            or host_pid <= 0
        ):
            self._append_lifecycle_error(
                f"pmon returned an invalid host PID: {row!r}"
            )
            return False
        if self.expected_host_pid is None:
            return False
        if host_pid != self.expected_host_pid:
            self._append_lifecycle_error(
                "pmon detected a different GPU process after the idle guard: "
                f"{row!r}"
            )
            return False
        if row.get("gpu_index") != self.device_index:
            self._append_lifecycle_error(
                f"pmon row targeted the wrong GPU: {row!r}"
            )
            return False
        if row.get("process_type") != "C":
            self._append_lifecycle_error(
                f"pmon row is not direct compute type C: {row!r}"
            )
            return False
        command = row.get("command")
        if not isinstance(command, str) or not command.strip():
            self._append_lifecycle_error(
                f"pmon compute row lacks a command: {row!r}"
            )
            return False
        if not self._row_has_source_dt(row):
            self._append_lifecycle_error(
                f"pmon compute row lacks source DT evidence: {row!r}"
            )
            return False
        line_number = row.get("line_number")
        if (
            isinstance(self._backend_window_start_line_number, int)
            and isinstance(line_number, int)
            and line_number > self._backend_window_start_line_number
            and float(row_receipt[1])
            >= float(self._backend_window_started[1])
            and float(row["source_timestamp_unix_seconds"])
            > float(self._backend_window_started[0])
            and self._window_compute_ready[0] is None
        ):
            self._window_compute_ready = row_receipt
            self._window_compute_line_number = line_number
        return True

    def _validate_new_rows(self, rows, receipt):
        if len(rows) < len(self._parsed_rows):
            self._append_lifecycle_error(
                "pmon parser removed rows after additional stdout"
            )
            return
        if rows[: len(self._parsed_rows)] != self._parsed_rows:
            self._append_lifecycle_error(
                "pmon parser changed previously parsed rows"
            )
            return
        for row in rows[len(self._parsed_rows) :]:
            if not isinstance(row, dict):
                self._append_lifecycle_error("pmon parser returned a non-object row")
                continue
            row_receipt = self._row_receipt(row, receipt)
            source_timestamp = row.get("source_timestamp_unix_seconds")
            received_wall = row_receipt[0]
            if (
                not self._row_has_source_dt(row)
                or not isinstance(received_wall, (int, float))
                or isinstance(received_wall, bool)
                or not math.isfinite(float(received_wall))
                or float(source_timestamp) > float(received_wall) + 1.0
                or float(received_wall) - float(source_timestamp) > 5.0
                or (
                    self._last_source_timestamp is not None
                    and float(source_timestamp) < self._last_source_timestamp
                )
            ):
                self._append_lifecycle_error(
                    f"pmon row source timestamp is invalid: {row!r}"
                )
                continue
            self._last_source_timestamp = float(source_timestamp)
            if self._row_is_mps(row):
                self._append_lifecycle_error(
                    f"pmon detected an MPS process: {row!r}"
                )
            host_pid = row.get("host_pid")
            if host_pid is None:
                if (
                    row.get("gpu_index") != self.device_index
                    or row.get("process_type") is not None
                    or row.get("command") != "-"
                    or not self._row_has_source_dt(row)
                ):
                    self._append_lifecycle_error(
                        f"pmon emitted an ambiguous idle row: {row!r}"
                    )
                    continue
                if self._idle_baseline_ready[0] is None:
                    self._idle_baseline_ready = row_receipt
                    self._idle_baseline_line_number = row.get("line_number")
                continue
            if self._idle_baseline_ready[0] is None:
                self._append_lifecycle_error(
                    "pmon observed a process before the required idle baseline: "
                    f"{row!r}"
                )
                continue
            if self.expected_host_pid is not None:
                self._validate_bound_compute_row(row, row_receipt)
        self._parsed_rows = [dict(row) for row in rows]

    def _refresh_parse(self, receipt):
        raw_stdout = "".join(
            record["raw_line"] for record in self._stdout_line_records
        )
        try:
            parsed = parse_nvidia_smi_pmon_output(raw_stdout)
        except Exception as exc:
            self._append_lifecycle_error(
                f"pmon shared parser raised unexpectedly: {exc!r}"
            )
            return
        if not isinstance(parsed, dict):
            self._append_lifecycle_error(
                "pmon shared parser did not return a JSON object"
            )
            return
        header_columns = parsed.get("header_columns")
        rows = parsed.get("rows")
        parser_errors = parsed.get("errors")
        if header_columns is not None and not isinstance(header_columns, list):
            self._append_lifecycle_error("pmon header_columns is malformed")
            return
        if not isinstance(rows, list) or not isinstance(parser_errors, list):
            self._append_lifecycle_error("pmon shared parser fields are malformed")
            return
        if header_columns:
            if self._header_columns is None:
                self._header_columns = list(header_columns)
                self._header_ready = receipt
            elif self._header_columns != header_columns:
                self._append_lifecycle_error(
                    "pmon header changed while the monitor was running"
                )
        self._parser_errors = list(parser_errors)
        if self._header_columns is not None and parser_errors:
            self._append_lifecycle_error(
                "pmon shared parser rejected stdout: " + "; ".join(parser_errors)
            )
        self._validate_new_rows(rows, receipt)

    def _read_stream(self, stream_name, stream):
        target = (
            self._stdout_line_records
            if stream_name == "stdout"
            else self._stderr_line_records
        )
        try:
            while True:
                raw_line = stream.readline()
                if raw_line == "":
                    break
                receipt = self._now()
                with self._condition:
                    record = {
                        "line_index": len(target),
                        "raw_line": raw_line,
                        "received_at_unix_seconds": receipt[0],
                        "received_at_monotonic_seconds": receipt[1],
                    }
                    target.append(record)
                    if stream_name == "stdout":
                        self._refresh_parse(receipt)
                    elif "mps" in raw_line.lower():
                        self._append_lifecycle_error(
                            "pmon stderr mentioned MPS"
                        )
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._reader_errors.append(
                    f"pmon {stream_name} reader failed: {exc!r}"
                )
                self._condition.notify_all()
        finally:
            with self._condition:
                self._condition.notify_all()

    def start(self):
        environment = os.environ.copy()
        environment.update(self.locale)
        self._spawn_started = self._now()
        try:
            self.process = subprocess.Popen(
                self.command,
                executable=self.resolved_executable,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
                env=environment,
            )
        except OSError as exc:
            raise RadialAttentionDependencyError(
                f"Cannot start continuous NVIDIA pmon: {exc!r}"
            ) from exc
        self._process_started = self._now()
        for stream_name, stream in (
            ("stdout", self.process.stdout),
            ("stderr", self.process.stderr),
        ):
            thread = threading.Thread(
                target=self._read_stream,
                args=(stream_name, stream),
                name=f"fasta2v-pmon-{stream_name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _raise_if_failed(self, phase):
        if self._reader_errors or self._lifecycle_errors:
            errors = [*self._reader_errors, *self._lifecycle_errors]
            raise RadialAttentionDependencyError(
                f"continuous pmon failed during {phase}: " + "; ".join(errors)
            )
        if self.process is None:
            raise RadialAttentionDependencyError(
                f"continuous pmon is absent during {phase}"
            )
        return_code = self.process.poll()
        if return_code is not None:
            raise RadialAttentionDependencyError(
                f"continuous pmon exited early during {phase}: {return_code}"
            )

    def _wait_for(self, predicate, phase, timeout_seconds):
        deadline = time.monotonic() + float(timeout_seconds)
        with self._condition:
            while True:
                self._raise_if_failed(phase)
                if predicate():
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._timed_out = True
                    raise RadialAttentionDependencyError(
                        f"continuous pmon timed out during {phase}"
                    )
                self._condition.wait(timeout=min(remaining, 0.1))

    def wait_until_ready(self):
        self._wait_for(
            lambda: self._header_ready[0] is not None,
            "header readiness",
            PMON_READY_TIMEOUT_SECONDS,
        )
        self._wait_for(
            lambda: self._idle_baseline_ready[0] is not None,
            "idle baseline readiness",
            PMON_READY_TIMEOUT_SECONDS,
        )

    def bind_expected_host_pid(self, host_pid):
        host_pid = int(host_pid)
        if host_pid <= 0:
            raise RadialAttentionDependencyError(
                f"Cannot bind invalid pmon host PID {host_pid!r}"
            )
        with self._condition:
            self._raise_if_failed("host PID binding")
            if self._idle_baseline_ready[0] is None:
                raise RadialAttentionDependencyError(
                    "Cannot bind pmon host PID before the idle baseline"
                )
            if self.expected_host_pid not in (None, host_pid):
                raise RadialAttentionDependencyError(
                    "Cannot replace an already-bound pmon host PID"
                )
            self.expected_host_pid = host_pid
            self._host_pid_bound = self._now()
            baseline_line = self._idle_baseline_line_number
            for row in self._parsed_rows:
                if (
                    row.get("host_pid") is not None
                    and isinstance(row.get("line_number"), int)
                    and isinstance(baseline_line, int)
                    and row["line_number"] > baseline_line
                ):
                    self._validate_bound_compute_row(
                        row,
                        self._row_receipt(row, self._host_pid_bound),
                    )
            self._raise_if_failed("host PID binding")

    def begin_backend_window(self):
        with self._condition:
            self._raise_if_failed("backend window start")
            if self.expected_host_pid is None or self._host_pid_bound[0] is None:
                raise RadialAttentionDependencyError(
                    "Cannot start backend window before binding the host PID"
                )
            if self._backend_window_started[0] is not None:
                raise RadialAttentionDependencyError(
                    "Radial backend window was started more than once"
                )
            self._backend_window_start_line_number = len(
                self._stdout_line_records
            )
            self._backend_window_started = self._now()
            return self._backend_window_started

    def window_compute_seen(self):
        with self._condition:
            self._raise_if_failed("backend window sampling")
            return self._window_compute_ready[0] is not None

    def require_running(self, phase):
        with self._condition:
            self._raise_if_failed(phase)

    def stop(self):
        if self._stop_requested[0] is not None:
            return
        self._stop_requested = self._now()
        if self.process is None:
            self._termination_method = "not_started"
            self._append_lifecycle_error(
                "continuous pmon process was never started"
            )
            return
        return_code = self.process.poll()
        if return_code is not None:
            self._termination_method = "unexpected_early_exit"
            self._append_lifecycle_error(
                f"continuous pmon exited before stop was requested: {return_code}"
            )
        else:
            self._termination_method = "terminate"
            try:
                self.process.terminate()
                self.process.wait(timeout=PMON_TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self._termination_method = "kill_after_terminate_timeout"
                self._timed_out = True
                self._append_lifecycle_error(
                    "continuous pmon required SIGKILL after terminate timeout"
                )
                try:
                    self.process.kill()
                    self.process.wait(timeout=PMON_KILL_TIMEOUT_SECONDS)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self._append_lifecycle_error(
                        f"continuous pmon could not be killed: {exc!r}"
                    )
            except OSError as exc:
                self._append_lifecycle_error(
                    f"continuous pmon terminate failed: {exc!r}"
                )
        self._exit_code = self.process.poll()
        if self._exit_code is not None:
            self._process_exited = self._now()
        else:
            self._append_lifecycle_error(
                "continuous pmon has no bounded exit status"
            )
        for thread in self._threads:
            thread.join(timeout=PMON_KILL_TIMEOUT_SECONDS)
            if thread.is_alive():
                self._append_lifecycle_error(
                    f"continuous pmon reader did not exit: {thread.name}"
                )
        if (
            self._termination_method == "terminate"
            and self._exit_code not in (0, -signal.SIGTERM)
        ):
            self._append_lifecycle_error(
                f"continuous pmon returned unexpected exit code {self._exit_code}"
            )

    def evidence(self):
        with self._condition:
            stdout_records = [dict(record) for record in self._stdout_line_records]
            stderr_records = [dict(record) for record in self._stderr_line_records]
            raw_stdout = "".join(record["raw_line"] for record in stdout_records)
            raw_stderr = "".join(record["raw_line"] for record in stderr_records)
        try:
            parsed = parse_nvidia_smi_pmon_output(raw_stdout)
        except Exception as exc:
            parsed = {
                "header_columns": None,
                "rows": [],
                "errors": [f"shared parser raised: {exc!r}"],
            }
        rows_by_line = {
            row.get("line_number"): row
            for row in parsed.get("rows", [])
            if isinstance(row, dict)
            and isinstance(row.get("line_number"), int)
        }
        for record in stdout_records:
            record["parsed_row"] = rows_by_line.get(record["line_index"] + 1)
        parser_errors = parsed.get("errors")
        errors = [*self._reader_errors, *self._lifecycle_errors]
        if raw_stderr:
            errors.append("continuous pmon emitted stderr during the audit")
        if not isinstance(parser_errors, list):
            errors.append("final pmon parser errors field is malformed")
            parser_errors = ["malformed errors field"]
        elif parser_errors:
            errors.extend(f"pmon parser: {error}" for error in parser_errors)
        if parsed.get("header_columns") is None:
            errors.append("final pmon stdout lacks a dynamic header")
        if self._idle_baseline_ready[0] is None:
            errors.append("pmon never produced the required idle baseline row")
        if self.expected_host_pid is None or self._host_pid_bound[0] is None:
            errors.append("pmon never bound the context-live host PID")
        if (
            self._backend_window_started[0] is None
            or not isinstance(self._backend_window_start_line_number, int)
        ):
            errors.append("pmon never marked the exact backend window")
        final_rows = parsed.get("rows")
        if not isinstance(final_rows, list):
            errors.append("final pmon rows field is malformed")
            final_rows = []
        baseline_line = self._idle_baseline_line_number
        matching_rows_in_window = []
        previous_source_timestamp = None
        for row in final_rows:
            if not isinstance(row, dict):
                errors.append("final pmon parser returned a non-object row")
                continue
            line_number = row.get("line_number")
            record = (
                stdout_records[line_number - 1]
                if isinstance(line_number, int)
                and 0 < line_number <= len(stdout_records)
                else None
            )
            source_timestamp = row.get("source_timestamp_unix_seconds")
            received_wall = (
                record.get("received_at_unix_seconds")
                if isinstance(record, dict)
                else None
            )
            if (
                not isinstance(source_timestamp, (int, float))
                or isinstance(source_timestamp, bool)
                or not math.isfinite(float(source_timestamp))
                or not isinstance(received_wall, (int, float))
                or isinstance(received_wall, bool)
                or not math.isfinite(float(received_wall))
                or float(source_timestamp) > float(received_wall) + 1.0
                or float(received_wall) - float(source_timestamp) > 5.0
                or (
                    previous_source_timestamp is not None
                    and float(source_timestamp) < previous_source_timestamp
                )
            ):
                errors.append(
                    f"final pmon source timestamp is invalid: {row!r}"
                )
            else:
                previous_source_timestamp = float(source_timestamp)
            host_pid = row.get("host_pid")
            if host_pid is None:
                if (
                    row.get("gpu_index") != self.device_index
                    or row.get("process_type") is not None
                    or row.get("command") != "-"
                    or not self._row_has_source_dt(row)
                ):
                    errors.append(f"final pmon idle row is ambiguous: {row!r}")
                continue
            if self._row_is_mps(row):
                errors.append(f"final pmon row identifies MPS: {row!r}")
            if (
                row.get("gpu_index") != self.device_index
                or host_pid != self.expected_host_pid
                or row.get("process_type") != "C"
                or not isinstance(row.get("command"), str)
                or not row["command"].strip()
                or not self._row_has_source_dt(row)
            ):
                errors.append(
                    f"final pmon row is not the expected direct C client: {row!r}"
                )
                continue
            raw_line = (
                stdout_records[line_number - 1]["raw_line"]
                if isinstance(line_number, int)
                and 0 < line_number <= len(stdout_records)
                else ""
            )
            if (
                isinstance(self._backend_window_start_line_number, int)
                and isinstance(line_number, int)
                and line_number > self._backend_window_start_line_number
                and row["source_date"] in raw_line
                and row["source_time"] in raw_line
            ):
                matching_rows_in_window.append(row)
        if not matching_rows_in_window:
            errors.append(
                "pmon observed no source-DT direct C row in the backend window"
            )
        selected_window_row = next(
            (
                row
                for row in matching_rows_in_window
                if row.get("line_number") == self._window_compute_line_number
            ),
            None,
        )
        selected_window_record = (
            stdout_records[self._window_compute_line_number - 1]
            if isinstance(self._window_compute_line_number, int)
            and 0 < self._window_compute_line_number <= len(stdout_records)
            else None
        )
        if (
            self._window_compute_ready[0] is None
            or not isinstance(selected_window_row, dict)
            or not isinstance(selected_window_record, dict)
            or self._window_compute_ready[0]
            != selected_window_record.get("received_at_unix_seconds")
            or self._window_compute_ready[1]
            != selected_window_record.get("received_at_monotonic_seconds")
        ):
            errors.append(
                "pmon window-C readiness does not bind a matching raw line"
            )
        return {
            "status": "ok" if not errors else "failed",
            "command": list(self.command),
            "nvidia_smi_binary": dict(self.nvidia_smi_binary),
            "resolved_executable": self.resolved_executable,
            "process_pid": self.process.pid if self.process is not None else None,
            "expected_host_pid": self.expected_host_pid,
            "locale": dict(self.locale),
            "raw_stdout": raw_stdout,
            "raw_stderr": raw_stderr,
            "raw_stdout_bytes": len(raw_stdout.encode("utf-8")),
            "raw_stderr_bytes": len(raw_stderr.encode("utf-8")),
            "raw_stdout_sha256": hashlib.sha256(
                raw_stdout.encode("utf-8")
            ).hexdigest(),
            "raw_stderr_sha256": hashlib.sha256(
                raw_stderr.encode("utf-8")
            ).hexdigest(),
            "exit_code": self._exit_code,
            "termination_method": self._termination_method,
            "timed_out": self._timed_out,
            "spawn_started_at_unix_seconds": self._spawn_started[0],
            "spawn_started_at_monotonic_seconds": self._spawn_started[1],
            "process_started_at_unix_seconds": self._process_started[0],
            "process_started_at_monotonic_seconds": self._process_started[1],
            "header_ready_at_unix_seconds": self._header_ready[0],
            "header_ready_at_monotonic_seconds": self._header_ready[1],
            "idle_baseline_ready_at_unix_seconds": (
                self._idle_baseline_ready[0]
            ),
            "idle_baseline_ready_at_monotonic_seconds": (
                self._idle_baseline_ready[1]
            ),
            "idle_baseline_line_number": self._idle_baseline_line_number,
            "host_pid_bound_at_unix_seconds": self._host_pid_bound[0],
            "host_pid_bound_at_monotonic_seconds": self._host_pid_bound[1],
            "backend_window_started_at_unix_seconds": (
                self._backend_window_started[0]
            ),
            "backend_window_started_at_monotonic_seconds": (
                self._backend_window_started[1]
            ),
            "backend_window_start_line_number": (
                self._backend_window_start_line_number
            ),
            "window_compute_ready_at_unix_seconds": (
                self._window_compute_ready[0]
            ),
            "window_compute_ready_at_monotonic_seconds": (
                self._window_compute_ready[1]
            ),
            "window_compute_line_number": self._window_compute_line_number,
            "stop_requested_at_unix_seconds": self._stop_requested[0],
            "stop_requested_at_monotonic_seconds": self._stop_requested[1],
            "process_exited_at_unix_seconds": self._process_exited[0],
            "process_exited_at_monotonic_seconds": self._process_exited[1],
            "header_columns": parsed.get("header_columns"),
            "rows": parsed.get("rows"),
            "parser_errors": parser_errors,
            "stdout_line_records": stdout_records,
            "stderr_line_records": stderr_records,
            "errors": errors,
        }


def _read_and_bind_pre_run_gpu(path, expected):
    path = Path(path).resolve()
    try:
        payload = path.read_bytes()
        parsed = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise RadialAttentionDependencyError(
            f"Cannot read runner-created pre-run GPU evidence {path}: {exc}"
        ) from exc
    if parsed != expected:
        raise RadialAttentionDependencyError(
            "Passed pre-run GPU evidence differs from its persisted file"
        )
    return path, hashlib.sha256(payload).hexdigest()


def run_microtest(device_index=0, pre_run_gpu=None, pre_run_gpu_path=None):
    microtest_started_at = time.time()
    microtest_started_at_monotonic = time.monotonic()
    if int(device_index) != 0:
        raise RadialAttentionDependencyError(
            "Radial microtest must launch on logical CUDA device 0"
        )
    if pre_run_gpu_path is None:
        raise RadialAttentionDependencyError(
            "Radial microtest requires the persisted pre_run_gpu.json path"
        )
    pre_run_gpu_path, pre_run_gpu_sha256 = _read_and_bind_pre_run_gpu(
        pre_run_gpu_path,
        pre_run_gpu,
    )
    mps_variables = sorted(
        name for name in os.environ if name.startswith("CUDA_MPS_")
    )
    if mps_variables:
        raise RadialAttentionDependencyError(
            f"Radial microtest forbids CUDA MPS variables: {mps_variables}"
        )
    pre_run_errors = validate_pre_run_gpu_report(pre_run_gpu)
    pre_run_sampled_at = (
        pre_run_gpu.get("sampled_at_unix_seconds")
        if isinstance(pre_run_gpu, dict)
        else None
    )
    if (
        not isinstance(pre_run_sampled_at, (int, float))
        or isinstance(pre_run_sampled_at, bool)
        or not math.isfinite(float(pre_run_sampled_at))
    ):
        pre_run_errors.append("pre-run GPU sample timestamp is invalid")
    elif (
        microtest_started_at < float(pre_run_sampled_at)
        or microtest_started_at - float(pre_run_sampled_at)
        > GPU_IDLE_GUARD_MAX_AGE_SECONDS
    ):
        pre_run_errors.append(
            "pre-run GPU idle guard is future-dated or stale"
        )
    if pre_run_errors:
        raise RadialAttentionDependencyError(
            "Radial microtest requires valid runner-created idle GPU evidence: "
            + "; ".join(pre_run_errors)
        )
    receipt_path, receipt = verify_radial_install_receipt()
    verify_radial_runtime_loader_environment(receipt)
    import torch

    flashinfer = load_flashinfer_api(
        receipt["installed_flashinfer_package_root"]
    )
    runtime_dependencies_before_cuda = (
        verify_radial_runtime_loaded_dependencies(receipt)
    )
    source_module = load_official_radial_mask_module(
        receipt["derived_module"]["path"]
    )

    def named_gpu_query(index):
        return query_gpu_compute_processes(
            index,
            include_process_name=True,
        )

    monitor = GpuProcessMonitor(
        device_index=0,
        interval_seconds=0.1,
        sample_fn=named_gpu_query,
    )
    with monitor:
        immediate_idle_sample = monitor.summary()["samples"][0]
        if (
            immediate_idle_sample.get("available") is not True
            or immediate_idle_sample.get("error") is not None
            or immediate_idle_sample.get("device_index") != 0
            or immediate_idle_sample.get("device_uuid")
            != pre_run_gpu.get("device_uuid")
            or immediate_idle_sample.get("device_name")
            != pre_run_gpu.get("device_name")
            or immediate_idle_sample.get("process_count") != 0
            or immediate_idle_sample.get("processes") != []
            or immediate_idle_sample.get("boot_id") != pre_run_gpu.get("boot_id")
            or immediate_idle_sample.get("nvidia_smi_binary")
            != pre_run_gpu.get("nvidia_smi_binary")
        ):
            raise RadialAttentionDependencyError(
                "Physical GPU 0 was not idle immediately before CUDA touch: "
                f"{immediate_idle_sample}"
            )

        pmon = _ContinuousPmon(0)
        if pmon.nvidia_smi_binary != pre_run_gpu.get("nvidia_smi_binary"):
            raise RadialAttentionDependencyError(
                "Trusted nvidia-smi binary changed after the idle guard"
            )
        pmon_error = None
        try:
            pmon.start()
            pmon.wait_until_ready()
            cuda_touch_started_at_unix = time.time()
            cuda_touch_started_at_monotonic = time.monotonic()
            if immediate_idle_sample.get(
                "query_finished_at_monotonic_seconds",
                math.inf,
            ) > cuda_touch_started_at_monotonic:
                raise RadialAttentionDependencyError(
                    "Immediate idle query did not finish before CUDA touch"
                )
            if not torch.cuda.is_available():
                raise RadialAttentionDependencyError(
                    "CUDA is unavailable for the required Radial FlashInfer microtest"
                )
            device = torch.device("cuda", int(device_index))
            properties = torch.cuda.get_device_properties(device)
            compute_capability = (int(properties.major), int(properties.minor))
            if compute_capability != (8, 0):
                raise RadialAttentionDependencyError(
                    "The fixed Radial protocol targets A100 sm80; got "
                    f"compute capability {compute_capability}"
                )
            runtime_device_name = torch.cuda.get_device_name(device)
            raw_device_uuid = str(getattr(properties, "uuid", ""))
            current_device_uuid = (
                raw_device_uuid
                if raw_device_uuid.startswith("GPU-")
                else f"GPU-{raw_device_uuid}"
            )
            visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
            if (
                runtime_device_name != pre_run_gpu.get("device_name")
                or current_device_uuid != pre_run_gpu.get("device_uuid")
                or visible_device != current_device_uuid
            ):
                raise RadialAttentionDependencyError(
                    "CUDA logical device 0 identity differs from the idle guard: "
                    f"name={runtime_device_name!r}, uuid={current_device_uuid!r}, "
                    f"CUDA_VISIBLE_DEVICES={visible_device!r}"
                )

            generator = torch.Generator(device=device).manual_seed(0)
            shape = (1, RADIAL_SEQUENCE, RADIAL_HEADS, RADIAL_HEAD_DIM)
            q, k, v = (
                torch.randn(
                    shape,
                    generator=generator,
                    device=device,
                    dtype=torch.bfloat16,
                )
                for _ in range(3)
            )
            qkv_storage_bytes = sum(
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in (q, k, v)
            )
            setup_cuda_synchronize_started_at_unix_seconds = time.time()
            setup_cuda_synchronize_started_at_monotonic_seconds = time.monotonic()
            torch.cuda.synchronize(device)
            setup_cuda_synchronized_at_unix_seconds = time.time()
            setup_cuda_synchronized_at_monotonic_seconds = time.monotonic()
            context_live_sample = named_gpu_query(0)
            context_processes = context_live_sample.get("processes")
            if (
                context_live_sample.get("available") is not True
                or context_live_sample.get("error") is not None
                or context_live_sample.get("device_index") != 0
                or context_live_sample.get("device_uuid")
                != pre_run_gpu.get("device_uuid")
                or context_live_sample.get("device_name")
                != pre_run_gpu.get("device_name")
                or context_live_sample.get("boot_id")
                != pre_run_gpu.get("boot_id")
                or context_live_sample.get("nvidia_smi_binary")
                != pre_run_gpu.get("nvidia_smi_binary")
                or context_live_sample.get("process_count") != 1
                or not isinstance(context_processes, list)
                or len(context_processes) != 1
                or not isinstance(context_processes[0], dict)
                or not isinstance(context_processes[0].get("host_pid"), int)
                or isinstance(context_processes[0].get("host_pid"), bool)
                or context_processes[0].get("host_pid", 0) <= 0
            ):
                raise RadialAttentionDependencyError(
                    "Cannot bind pmon without one context-live host PID: "
                    f"{context_live_sample}"
                )
            context_host_pid = int(context_processes[0]["host_pid"])
            pmon.bind_expected_host_pid(context_host_pid)
            backend = RadialVideoSelfAttentionBackend(
                torch_module=torch,
                flashinfer_module=flashinfer,
                mask_generator=getattr(source_module, RADIAL_MASK_API),
                get_indptr_from_mask=source_module.get_indptr_from_mask,
                get_indices_from_mask=source_module.get_indices_from_mask,
                rope_apply_fn=lambda value, _grid, _freqs: value,
                profile="conservative",
                install_receipt={
                    "path": str(receipt_path),
                    "commit": receipt["commit"],
                    "derived_module_sha256": receipt["derived_module"]["sha256"],
                    "flashinfer_version": receipt["flashinfer_version"],
                },
            )
            sequence_lengths = torch.tensor([RADIAL_SEQUENCE], dtype=torch.int64)
            grids = torch.tensor([RADIAL_GRID], dtype=torch.int64)
            (
                exact_backend_started_at_unix_seconds,
                exact_backend_started_at_monotonic_seconds,
            ) = pmon.begin_backend_window()
            backend_window_deadline = (
                exact_backend_started_at_monotonic_seconds
                + PMON_WINDOW_TIMEOUT_SECONDS
            )
            exact_backend_call_count = 0
            while True:
                output = backend(
                    _IdentityOviAttention(q, k, v),
                    None,
                    sequence_lengths,
                    grids,
                    None,
                )
                exact_backend_call_count += 1
                torch.cuda.synchronize(device)
                pmon.require_running("exact backend return")
                if pmon.window_compute_seen():
                    break
                if time.monotonic() >= backend_window_deadline:
                    raise RadialAttentionDependencyError(
                        "continuous pmon observed no direct C sample during "
                        f"{exact_backend_call_count} exact backend calls within "
                        f"{PMON_WINDOW_TIMEOUT_SECONDS:.1f}s"
                    )
            exact_backend_returned_at_unix_seconds = time.time()
            exact_backend_returned_at_monotonic_seconds = time.monotonic()
            if tuple(output.shape) != (
                1,
                RADIAL_SEQUENCE,
                RADIAL_HEADS * RADIAL_HEAD_DIM,
            ):
                raise RadialAttentionDependencyError(
                    "Radial microtest returned incompatible shape "
                    f"{tuple(output.shape)}"
                )
            if output.dtype != torch.bfloat16 or output.device != device:
                raise RadialAttentionDependencyError(
                    "Radial microtest changed BF16 dtype or CUDA device"
                )
            finite = bool(torch.isfinite(output).all().item())
            if not finite:
                raise RadialAttentionDependencyError(
                    "Radial microtest output contains NaN or Inf"
                )
            output_abs_mean = float(output.float().abs().mean().item())
            output_abs_max = float(output.float().abs().max().item())
            allocator_bytes = int(torch.cuda.memory_allocated(device))
            reserved_bytes = int(torch.cuda.memory_reserved(device))
            if (
                allocator_bytes < qkv_storage_bytes
                or reserved_bytes < allocator_bytes
            ):
                raise RadialAttentionDependencyError(
                    "Current Python process does not retain the expected live "
                    "CUDA storage: "
                    f"qkv={qkv_storage_bytes}, allocated={allocator_bytes}, "
                    f"reserved={reserved_bytes}"
                )
            cuda_synchronize_started_at_unix_seconds = time.time()
            cuda_synchronize_started_at_monotonic_seconds = time.monotonic()
            torch.cuda.synchronize(device)
            cuda_synchronized_at_unix = time.time()
            cuda_synchronized_at_monotonic = time.monotonic()
        except Exception as exc:
            pmon_error = exc
        finally:
            pmon.stop()
            process_type_sample = pmon.evidence()
        if pmon_error is not None:
            raise RadialPmonEvidenceError(
                f"Radial pmon-audited backend window failed: {pmon_error!r}",
                process_type_sample,
            ) from pmon_error
        if process_type_sample.get("status") != "ok":
            raise RadialPmonEvidenceError(
                "continuous pmon evidence is invalid: "
                + "; ".join(process_type_sample.get("errors", [])),
                process_type_sample,
            )
        window_line_number = process_type_sample.get(
            "window_compute_line_number"
        )
        window_rows = process_type_sample.get("rows")
        window_line_records = process_type_sample.get("stdout_line_records")
        window_row = next(
            (
                row
                for row in window_rows
                if isinstance(row, dict)
                and row.get("line_number") == window_line_number
            ),
            None,
        ) if isinstance(window_rows, list) else None
        window_record = (
            window_line_records[window_line_number - 1]
            if isinstance(window_line_records, list)
            and isinstance(window_line_number, int)
            and 0 < window_line_number <= len(window_line_records)
            else None
        )
        window_source_time = (
            window_row.get("source_timestamp_unix_seconds")
            if isinstance(window_row, dict)
            else None
        )
        window_received_monotonic = (
            window_record.get("received_at_monotonic_seconds")
            if isinstance(window_record, dict)
            else None
        )
        pmon_stop_monotonic = process_type_sample.get(
            "stop_requested_at_monotonic_seconds"
        )
        if not (
            isinstance(window_source_time, (int, float))
            and not isinstance(window_source_time, bool)
            and math.isfinite(float(window_source_time))
            and exact_backend_started_at_unix_seconds
            < float(window_source_time)
            <= exact_backend_returned_at_unix_seconds
            and isinstance(window_received_monotonic, (int, float))
            and not isinstance(window_received_monotonic, bool)
            and isinstance(pmon_stop_monotonic, (int, float))
            and not isinstance(pmon_stop_monotonic, bool)
            and exact_backend_started_at_monotonic_seconds
            <= float(window_received_monotonic)
            <= exact_backend_returned_at_monotonic_seconds
            <= float(pmon_stop_monotonic)
            and isinstance(window_row, dict)
            and window_row.get("gpu_index") == 0
            and window_row.get("host_pid") == context_host_pid
            and window_row.get("process_type") == "C"
        ):
            raise RadialPmonEvidenceError(
                "continuous pmon C sample does not intersect the exact "
                "backend/final-sync window",
                process_type_sample,
            )
        post_cuda_sample_1 = named_gpu_query(0)
        time.sleep(0.1)
        post_cuda_sample_2 = named_gpu_query(0)

    interval_samples = monitor.summary()["samples"]
    runtime_dependencies_after_cuda = (
        verify_radial_runtime_loaded_dependencies(receipt)
    )
    nspid_evidence = _current_pid_namespace_evidence()
    post_processes = post_cuda_sample_2.get("processes")
    host_pid = (
        post_processes[0].get("host_pid")
        if isinstance(post_processes, list) and len(post_processes) == 1
        else None
    )
    proc_visibility = _proc_pid_visibility(host_pid) if host_pid else {
        "status": "invalid_pid",
        "error": None,
    }
    pid_namespace_chain = nspid_evidence.get("chain", [])
    host_pid_namespace_visible = (
        len(pid_namespace_chain) >= 2
        and host_pid == pid_namespace_chain[0]
    )
    pid_binding_method = (
        "direct_nspid"
        if host_pid_namespace_visible
        else "snapshot_bound_singleton_after_idle_guard"
    )
    binding_errors = []
    gpu_identity = post_cuda_sample_2
    if gpu_identity.get("process_count") != 1:
        binding_errors.append("final post-CUDA GPU identity is not a singleton")
    expected_identity = (
        0,
        pre_run_gpu.get("device_uuid"),
        pre_run_gpu.get("device_name"),
        pre_run_gpu.get("boot_id"),
    )
    named_samples = [
        immediate_idle_sample,
        *interval_samples,
        context_live_sample,
        post_cuda_sample_1,
        post_cuda_sample_2,
    ]
    for index, sample in enumerate(named_samples):
        identity = (
            sample.get("device_index"),
            sample.get("device_uuid"),
            sample.get("device_name"),
            sample.get("boot_id"),
        )
        if (
            sample.get("available") is not True
            or sample.get("error") is not None
            or identity != expected_identity
            or sample.get("nvidia_smi_binary")
            != pre_run_gpu.get("nvidia_smi_binary")
            or not isinstance(sample.get("processes"), list)
            or sample.get("process_count") != len(sample.get("processes", []))
            or sample.get("process_count") not in (0, 1)
        ):
            binding_errors.append(
                f"GPU sample {index} is unavailable, drifted, or has invalid "
                "process cardinality"
            )
        for process in sample.get("processes", []):
            if (
                not isinstance(process, dict)
                or not isinstance(process.get("host_pid"), int)
                or isinstance(process.get("host_pid"), bool)
                or process.get("host_pid", 0) <= 0
                or not isinstance(process.get("used_memory_mib"), int)
                or isinstance(process.get("used_memory_mib"), bool)
                or process.get("used_memory_mib", 0) <= 0
                or not isinstance(process.get("process_name"), str)
                or not process.get("process_name", "").strip()
            ):
                binding_errors.append(
                    f"GPU sample {index} has invalid process details"
                )
    for label, sample in (
        ("context_live", context_live_sample),
        ("post_cuda_1", post_cuda_sample_1),
        ("post_cuda_2", post_cuda_sample_2),
    ):
        if sample.get("process_count") != 1:
            binding_errors.append(f"{label} sample is not a singleton")

    ordered_samples = sorted(
        named_samples,
        key=lambda sample: float(
            sample.get("sampled_at_monotonic_seconds", math.inf)
        ),
    )
    context_live_monotonic = context_live_sample.get(
        "query_finished_at_monotonic_seconds"
    )
    live_phase_samples = [
        sample
        for sample in ordered_samples
        if isinstance(
            sample.get("query_started_at_monotonic_seconds"),
            (int, float),
        )
        and isinstance(context_live_monotonic, (int, float))
        and sample["query_started_at_monotonic_seconds"]
        >= context_live_monotonic
    ]
    if not live_phase_samples or any(
        sample.get("process_count") != 1 for sample in live_phase_samples
    ):
        binding_errors.append(
            "sampled live-phase GPU snapshots were not all singleton"
        )

    positive_processes = [
        sample["processes"][0]
        for sample in ordered_samples
        if sample.get("process_count") == 1
        and isinstance(sample.get("processes"), list)
        and len(sample["processes"]) == 1
    ]
    positive_identities = {
        (
            process.get("host_pid"),
            str(process.get("process_name", "")).strip().lower(),
        )
        for process in positive_processes
    }
    if len(positive_identities) != 1:
        binding_errors.append(
            "GPU singleton PID/process name changed during the sampled interval"
        )
    normalized_process_name = (
        next(iter(positive_identities))[1] if positive_identities else ""
    )
    if (
        normalized_process_name != "[not found]"
        and "python" not in normalized_process_name
    ) or "mps" in normalized_process_name:
        binding_errors.append(
            "GPU process name is neither direct Python nor the fixed "
            f"namespace-hidden sentinel: {normalized_process_name!r}"
        )
    if positive_processes:
        nvidia_bytes = int(positive_processes[-1]["used_memory_mib"]) * 1024 * 1024
        if nvidia_bytes + 1024 * 1024 < reserved_bytes:
            binding_errors.append(
                "nvidia-smi used memory is smaller than the live CUDA reserve"
            )

    pmon_rows = process_type_sample.get("rows")
    actual_pmon_rows = (
        [row for row in pmon_rows if row.get("host_pid") is not None]
        if isinstance(pmon_rows, list)
        and all(isinstance(row, dict) for row in pmon_rows)
        else []
    )
    idle_pmon_rows = (
        [row for row in pmon_rows if row.get("host_pid") is None]
        if isinstance(pmon_rows, list)
        and all(isinstance(row, dict) for row in pmon_rows)
        else []
    )
    idle_baseline_line_number = process_type_sample.get(
        "idle_baseline_line_number"
    )
    backend_window_start_line_number = process_type_sample.get(
        "backend_window_start_line_number"
    )
    window_compute_line_number = process_type_sample.get(
        "window_compute_line_number"
    )
    direct_c_rows_in_window = [
        row
        for row in actual_pmon_rows
        if isinstance(backend_window_start_line_number, int)
        and isinstance(row.get("line_number"), int)
        and row["line_number"] > backend_window_start_line_number
        and row.get("gpu_index") == 0
        and row.get("host_pid") == host_pid
        and row.get("process_type") == "C"
        and _ContinuousPmon._row_has_source_dt(row)
    ]
    selected_window_c_row = next(
        (
            row
            for row in direct_c_rows_in_window
            if row.get("line_number") == window_compute_line_number
        ),
        None,
    )
    pmon_mps_process_detected = any(
        "mps" in str(row.get("command") or "").lower()
        for row in pmon_rows
        if isinstance(row, dict)
    )
    if (
        process_type_sample.get("status") != "ok"
        or not isinstance(pmon_rows, list)
        or not isinstance(process_type_sample.get("header_columns"), list)
        or not process_type_sample.get("header_columns")
        or process_type_sample.get("parser_errors") != []
        or not isinstance(idle_baseline_line_number, int)
        or not idle_pmon_rows
        or not isinstance(backend_window_start_line_number, int)
        or backend_window_start_line_number < idle_baseline_line_number
        or not isinstance(selected_window_c_row, dict)
        or process_type_sample.get("expected_host_pid") != host_pid
        or process_type_sample.get(
            "backend_window_started_at_unix_seconds"
        ) != exact_backend_started_at_unix_seconds
        or process_type_sample.get(
            "backend_window_started_at_monotonic_seconds"
        ) != exact_backend_started_at_monotonic_seconds
        or float(selected_window_c_row.get(
            "source_timestamp_unix_seconds", -math.inf
        )) <= exact_backend_started_at_unix_seconds
        or float(selected_window_c_row.get(
            "source_timestamp_unix_seconds", math.inf
        )) > exact_backend_returned_at_unix_seconds
        or any(
            row.get("gpu_index") != 0
            or row.get("process_type") is not None
            or row.get("command") != "-"
            or not _ContinuousPmon._row_has_source_dt(row)
            for row in idle_pmon_rows
        )
        or any(
            row.get("gpu_index") != 0
            or row.get("host_pid") != host_pid
            or row.get("process_type") != "C"
            or not isinstance(row.get("command"), str)
            or not row["command"].strip()
            or not _ContinuousPmon._row_has_source_dt(row)
            for row in actual_pmon_rows
        )
        or pmon_mps_process_detected
    ):
        binding_errors.append(
            "continuous pmon lacks an idle baseline followed by the expected "
            "source-DT direct C client during the audited backend window"
        )

    if nspid_evidence.get("status") != "ok":
        binding_errors.append("NSpid evidence was unavailable or malformed")
    elif pid_binding_method == "direct_nspid":
        if len(pid_namespace_chain) < 2 or host_pid != pid_namespace_chain[0]:
            binding_errors.append("direct NSpid binding lacks outer host PID")
    elif (
        pid_namespace_chain != [os.getpid()]
        or proc_visibility.get("status") != "not_visible"
    ):
        binding_errors.append(
            "snapshot binding lacks isolated PID namespace/proc visibility evidence"
        )

    pre_wall = float(pre_run_gpu["sampled_at_unix_seconds"])
    pre_monotonic = float(pre_run_gpu["sampled_at_monotonic_seconds"])
    post_wall = post_cuda_sample_2.get("sampled_at_unix_seconds")
    post_monotonic = post_cuda_sample_2.get(
        "query_started_at_monotonic_seconds"
    )
    post_1_finished_monotonic = post_cuda_sample_1.get(
        "query_finished_at_monotonic_seconds"
    )
    if not (
        pre_wall <= microtest_started_at <= cuda_touch_started_at_unix
        <= cuda_synchronized_at_unix <= float(post_wall)
        and pre_monotonic <= microtest_started_at_monotonic
        <= cuda_touch_started_at_monotonic
        <= cuda_synchronized_at_monotonic <= float(post_monotonic)
        and float(post_monotonic) - float(post_1_finished_monotonic) >= 0.1
        and float(post_wall) - pre_wall <= GPU_IDLE_GUARD_MAX_AGE_SECONDS
    ):
        binding_errors.append("GPU idle/touch/sync/post timestamps are invalid")
    if binding_errors:
        raise RadialAttentionDependencyError(
            "Radial snapshot-based GPU process binding failed: "
            + "; ".join(binding_errors)
        )

    metrics = backend.metrics()
    if (
        metrics.get("calls") != exact_backend_call_count
        or metrics.get("plan_cache_entries") != 1
        or metrics.get("plan_cache_misses") != 1
        or metrics.get("plan_cache_hits") != exact_backend_call_count - 1
    ):
        raise RadialAttentionDependencyError(
            "Repeated exact Radial backend calls do not match plan-cache metrics: "
            f"calls={exact_backend_call_count}, metrics={metrics}"
        )
    return {
        "status": "ok",
        "device": runtime_device_name,
        "device_uuid": gpu_identity["device_uuid"],
        "cuda_visible_devices": visible_device,
        "physical_device_index": gpu_identity["device_index"],
        "logical_cuda_device_index": device.index,
        "host_pid": host_pid,
        "python_pid": os.getpid(),
        "pid_namespace_chain": pid_namespace_chain,
        "host_pid_namespace_visible": host_pid_namespace_visible,
        "pid_binding_method": pid_binding_method,
        "pre_run_gpu": dict(pre_run_gpu),
        "pre_run_gpu_sha256": pre_run_gpu_sha256,
        "post_cuda_sampled_at_unix_seconds": gpu_identity[
            "sampled_at_unix_seconds"
        ],
        "gpu_process_count": gpu_identity["process_count"],
        "gpu_processes": gpu_identity["processes"],
        "cuda_synchronized": True,
        "gpu_process_binding": {
            "schema_version": GPU_PROCESS_BINDING_SCHEMA_VERSION,
            "binding_method": pid_binding_method,
            "claim_scope": "snapshot_bound_not_continuous_exclusivity",
            "pre_run_gpu_path": str(pre_run_gpu_path),
            "pre_run_gpu_sha256": pre_run_gpu_sha256,
            "pre_run_gpu": dict(pre_run_gpu),
            "microtest_started_at_unix_seconds": microtest_started_at,
            "microtest_started_at_monotonic_seconds": (
                microtest_started_at_monotonic
            ),
            "cuda_touch_started_at_unix_seconds": (
                cuda_touch_started_at_unix
            ),
            "cuda_touch_started_at_monotonic_seconds": (
                cuda_touch_started_at_monotonic
            ),
            "setup_cuda_synchronize_started_at_unix_seconds": (
                setup_cuda_synchronize_started_at_unix_seconds
            ),
            "setup_cuda_synchronize_started_at_monotonic_seconds": (
                setup_cuda_synchronize_started_at_monotonic_seconds
            ),
            "setup_cuda_synchronized_at_unix_seconds": (
                setup_cuda_synchronized_at_unix_seconds
            ),
            "setup_cuda_synchronized_at_monotonic_seconds": (
                setup_cuda_synchronized_at_monotonic_seconds
            ),
            "exact_kernel_completed": True,
            "exact_backend_call_count": exact_backend_call_count,
            "cuda_synchronize_completed": True,
            "exact_backend_started_at_unix_seconds": (
                exact_backend_started_at_unix_seconds
            ),
            "exact_backend_started_at_monotonic_seconds": (
                exact_backend_started_at_monotonic_seconds
            ),
            "exact_backend_returned_at_unix_seconds": (
                exact_backend_returned_at_unix_seconds
            ),
            "exact_backend_returned_at_monotonic_seconds": (
                exact_backend_returned_at_monotonic_seconds
            ),
            "cuda_synchronize_started_at_unix_seconds": (
                cuda_synchronize_started_at_unix_seconds
            ),
            "cuda_synchronize_started_at_monotonic_seconds": (
                cuda_synchronize_started_at_monotonic_seconds
            ),
            "cuda_synchronized_at_unix_seconds": (
                cuda_synchronized_at_unix
            ),
            "cuda_synchronized_at_monotonic_seconds": (
                cuda_synchronized_at_monotonic
            ),
            "current_cuda_device_uuid": current_device_uuid,
            "current_cuda_device_name": runtime_device_name,
            "current_cuda_device_index": int(device.index),
            "cuda_visible_devices": visible_device,
            "qkv_storage_bytes": qkv_storage_bytes,
            "allocator_memory_bytes": allocator_bytes,
            "reserved_memory_bytes": reserved_bytes,
            "python_executable": sys.executable,
            "python_executable_resolved": str(Path(sys.executable).resolve()),
            "container_pid": os.getpid(),
            "nvidia_smi_host_pid": host_pid,
            "nspid": nspid_evidence,
            "host_pid_proc_visibility": proc_visibility,
            "mps": {
                "cuda_mps_environment_variables": mps_variables,
                "pmon": process_type_sample,
                "mps_process_detected": pmon_mps_process_detected,
            },
            "immediate_pre_cuda_sample": immediate_idle_sample,
            "context_live_sample": context_live_sample,
            "interval_seconds": 0.1,
            "interval_samples": interval_samples,
            "post_cuda_samples": [
                post_cuda_sample_1,
                post_cuda_sample_2,
            ],
        },
        "compute_capability": list(compute_capability),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "dtype": str(output.dtype),
        "shape": list(shape),
        "grid": list(RADIAL_GRID),
        "profile": metrics["profile"],
        "decay_factor": metrics["decay_factor"],
        "prefix_sequence": metrics["prefix_sequence"],
        "tail_sequence": metrics["tail_sequence"],
        "tail_strategy": metrics["tail_strategy"],
        "exact_backend_call_count": exact_backend_call_count,
        "calls": metrics["calls"],
        "plan_cache_entries": metrics["plan_cache_entries"],
        "plan_cache_misses": metrics["plan_cache_misses"],
        "plan_cache_hits": metrics["plan_cache_hits"],
        "mask_audit": metrics["last_mask_audit"],
        "finite": finite,
        "runtime_dependencies_before_cuda": runtime_dependencies_before_cuda,
        "runtime_dependencies_after_cuda": runtime_dependencies_after_cuda,
        "output_abs_mean": output_abs_mean,
        "output_abs_max": output_abs_max,
    }


if __name__ == "__main__":
    run_dir = os.environ.get("FASTA2V_RUN_DIR")
    pre_run_path = Path(run_dir or "") / "pre_run_gpu.json"
    if not run_dir or not pre_run_path.is_file():
        raise SystemExit(
            "FASTA2V_RUN_DIR/pre_run_gpu.json is required for Radial microtest"
        )
    print(
        json.dumps(
            run_microtest(
                pre_run_gpu=json.loads(pre_run_path.read_text(encoding="utf-8")),
                pre_run_gpu_path=pre_run_path,
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
