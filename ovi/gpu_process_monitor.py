"""Fail-closed NVIDIA GPU identity and compute-process evidence.

This module deliberately uses only the Python standard library.  The run
scripts can therefore prove that physical GPU 0 is idle before importing
PyTorch or performing any CUDA preflight/model loading.
"""

from datetime import datetime, timezone
import base64
import hashlib
import math
import os
import re
import secrets
import stat
import subprocess
import threading
import time


GPU_EVIDENCE_SCHEMA_VERSION = 2
GPU_PROCESS_MONITOR_SCHEMA_VERSION = 2
GPU_QUERY_RECEIPT_SCHEMA_VERSION = 1
GPU_QUERY_LOCALE = {"LANG": "C", "LC_ALL": "C"}
GPU_QUERY_CADENCE_TOLERANCE_SECONDS = 1.0
NVIDIA_SMI_STDERR_ERROR = "nvidia-smi wrote to stderr despite exit code 0"
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


def _command_receipt(label, command):
    empty_digest = hashlib.sha256(b"").hexdigest()
    return {
        "label": label,
        "attempted": False,
        "command": list(command),
        "locale": dict(GPU_QUERY_LOCALE),
        "resolved_executable": TRUSTED_NVIDIA_SMI_PATH,
        "started_at_unix_seconds": None,
        "finished_at_unix_seconds": None,
        "started_at_monotonic_seconds": None,
        "finished_at_monotonic_seconds": None,
        "exit_code": None,
        "execution_error": None,
        "raw_stdout": "",
        "raw_stderr": "",
        "raw_stdout_base64": "",
        "raw_stderr_base64": "",
        "raw_stdout_bytes": 0,
        "raw_stderr_bytes": 0,
        "raw_stdout_sha256": empty_digest,
        "raw_stderr_sha256": empty_digest,
    }


def _new_query_receipt(device_index, include_process_name=False):
    process_fields = (
        "pid,process_name,used_memory"
        if include_process_name
        else "pid,used_memory"
    )
    identity_command = [
        TRUSTED_NVIDIA_SMI_PATH,
        "--id",
        str(device_index),
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        TRUSTED_NVIDIA_SMI_PATH,
        "--id",
        str(device_index),
        f"--query-compute-apps={process_fields}",
        "--format=csv,noheader,nounits",
    ]
    return {
        "schema_version": GPU_QUERY_RECEIPT_SCHEMA_VERSION,
        "status": "not_started",
        "device_index": device_index,
        "include_process_name": bool(include_process_name),
        "locale": dict(GPU_QUERY_LOCALE),
        "resolved_executable": TRUSTED_NVIDIA_SMI_PATH,
        "query_started_at_unix_seconds": time.time(),
        "query_finished_at_unix_seconds": None,
        "query_started_at_monotonic_seconds": time.monotonic(),
        "query_finished_at_monotonic_seconds": None,
        "commands": [
            _command_receipt("gpu_identity", identity_command),
            _command_receipt("compute_processes", process_command),
        ],
    }


def _finish_query_receipt(receipt, status):
    receipt["status"] = status
    receipt["query_finished_at_unix_seconds"] = time.time()
    receipt["query_finished_at_monotonic_seconds"] = time.monotonic()


def _unavailable_snapshot(
    device_index,
    error,
    nvidia_smi_binary=None,
    query_receipt=None,
):
    if query_receipt is None:
        query_receipt = _new_query_receipt(device_index)
        _finish_query_receipt(query_receipt, "not_run")
    return {
        "available": False,
        "error": str(error),
        "device_index": int(device_index),
        "device_uuid": None,
        "device_name": None,
        "processes": [],
        "process_count": None,
        "sampled_at_unix_seconds": query_receipt[
            "query_started_at_unix_seconds"
        ],
        "sampled_at_monotonic_seconds": query_receipt[
            "query_started_at_monotonic_seconds"
        ],
        "query_started_at_unix_seconds": query_receipt[
            "query_started_at_unix_seconds"
        ],
        "query_finished_at_unix_seconds": query_receipt[
            "query_finished_at_unix_seconds"
        ],
        "query_started_at_monotonic_seconds": query_receipt[
            "query_started_at_monotonic_seconds"
        ],
        "query_finished_at_monotonic_seconds": query_receipt[
            "query_finished_at_monotonic_seconds"
        ],
        "boot_id": _current_boot_id(),
        "nvidia_smi_binary": nvidia_smi_binary,
        "query_receipt": query_receipt,
    }


def _command_output(command):
    environment = os.environ.copy()
    environment.update(GPU_QUERY_LOCALE)
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
        env=environment,
    )


def _as_raw_bytes(value):
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError(f"command output has unsupported type {type(value)!r}")


def _store_raw_stream(receipt, stream_name, value):
    raw = _as_raw_bytes(value)
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        decoded = None
    receipt[f"raw_{stream_name}"] = decoded
    receipt[f"raw_{stream_name}_base64"] = base64.b64encode(raw).decode(
        "ascii"
    )
    receipt[f"raw_{stream_name}_bytes"] = len(raw)
    receipt[f"raw_{stream_name}_sha256"] = hashlib.sha256(raw).hexdigest()


def _execute_command(receipt, command_fn):
    receipt["attempted"] = True
    receipt["started_at_unix_seconds"] = time.time()
    receipt["started_at_monotonic_seconds"] = time.monotonic()
    stdout = b""
    stderr = b""
    exit_code = None
    execution_error = None
    try:
        result = command_fn(list(receipt["command"]))
        if isinstance(result, subprocess.CompletedProcess):
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
        else:
            stdout = result
            exit_code = 0
    except subprocess.CalledProcessError as error:
        stdout = error.output
        stderr = error.stderr
        exit_code = error.returncode
        execution_error = repr(error)
    except subprocess.TimeoutExpired as error:
        stdout = error.output
        stderr = error.stderr
        execution_error = repr(error)
    except Exception as error:
        execution_error = repr(error)
    try:
        _store_raw_stream(receipt, "stdout", stdout)
        _store_raw_stream(receipt, "stderr", stderr)
    except (TypeError, UnicodeError) as error:
        execution_error = repr(error)
        _store_raw_stream(receipt, "stdout", b"")
        _store_raw_stream(receipt, "stderr", b"")
        exit_code = None
    if (
        exit_code == 0
        and receipt.get("raw_stderr_bytes", 0) > 0
        and execution_error is None
    ):
        execution_error = NVIDIA_SMI_STDERR_ERROR
    receipt["exit_code"] = (
        int(exit_code) if _is_json_int(exit_code) else None
    )
    receipt["execution_error"] = execution_error
    receipt["finished_at_unix_seconds"] = time.time()
    receipt["finished_at_monotonic_seconds"] = time.monotonic()
    return receipt["exit_code"] == 0 and execution_error is None


def _parse_identity_output(raw_stdout, expected_device_index):
    if not isinstance(raw_stdout, str):
        raise ValueError("GPU identity stdout is not valid UTF-8")
    rows = [line.strip() for line in raw_stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(
            f"expected one nvidia-smi GPU identity row, found {len(rows)}"
        )
    fields = [field.strip() for field in rows[0].split(",", 2)]
    if len(fields) != 3:
        raise ValueError(f"unexpected nvidia-smi identity row: {rows[0]!r}")
    try:
        observed_index = int(fields[0])
    except ValueError as error:
        raise ValueError(
            f"unparseable nvidia-smi GPU index: {fields[0]!r}"
        ) from error
    device_uuid, device_name = fields[1], fields[2]
    if (
        observed_index != expected_device_index
        or not device_uuid
        or not device_name
    ):
        raise ValueError(
            "nvidia-smi returned an incomplete or mismatched GPU identity"
        )
    return observed_index, device_uuid, device_name


def _parse_process_output(raw_stdout, include_process_name):
    if not isinstance(raw_stdout, str):
        raise ValueError("compute-process stdout is not valid UTF-8")
    processes = []
    for line in raw_stdout.splitlines():
        if not line.strip():
            continue
        expected_fields = 3 if include_process_name else 2
        fields = [
            field.strip()
            for field in line.split(",", expected_fields - 1)
        ]
        if len(fields) != expected_fields:
            raise ValueError(
                f"unexpected nvidia-smi compute-process row: {line!r}"
            )
        try:
            process = {
                "host_pid": int(fields[0]),
                "used_memory_mib": int(fields[-1]),
            }
        except ValueError as error:
            raise ValueError(
                f"unparseable nvidia-smi compute-process row: {line!r}"
            ) from error
        if process["host_pid"] <= 0 or process["used_memory_mib"] <= 0:
            raise ValueError(
                f"invalid nvidia-smi compute-process row: {line!r}"
            )
        if include_process_name:
            if not fields[1]:
                raise ValueError(
                    f"missing nvidia-smi process name: {line!r}"
                )
            process["process_name"] = fields[1]
        processes.append(process)
    return processes


def query_gpu_compute_processes(
    device_index=0,
    command_fn=None,
    *,
    include_process_name=False,
    binary_metadata_fn=None,
):
    """Return identity plus compute processes for one physical GPU index."""
    if not _is_json_int(device_index):
        raise ValueError("GPU query device index must be an integer")
    if not isinstance(include_process_name, bool):
        raise ValueError("include_process_name must be boolean")
    device_index = int(device_index)
    command_fn = command_fn or _command_output
    binary_metadata_fn = binary_metadata_fn or trusted_nvidia_smi_metadata
    query_receipt = _new_query_receipt(
        device_index,
        include_process_name=include_process_name,
    )
    try:
        nvidia_smi_binary = binary_metadata_fn()
    except Exception as error:
        _finish_query_receipt(query_receipt, "binary_metadata_unavailable")
        return _unavailable_snapshot(
            device_index,
            repr(error),
            query_receipt=query_receipt,
        )
    binary_errors = trusted_nvidia_smi_metadata_errors(nvidia_smi_binary)
    if binary_errors:
        _finish_query_receipt(query_receipt, "binary_metadata_invalid")
        return _unavailable_snapshot(
            device_index,
            "; ".join(binary_errors),
            nvidia_smi_binary,
            query_receipt,
        )
    executable = nvidia_smi_binary["resolved_path"]
    query_receipt["resolved_executable"] = executable
    for command_receipt in query_receipt["commands"]:
        command_receipt["resolved_executable"] = executable
        command_receipt["command"][0] = executable
    boot_id = _current_boot_id()
    identity_receipt, process_receipt = query_receipt["commands"]
    if not _execute_command(identity_receipt, command_fn):
        _finish_query_receipt(query_receipt, "identity_command_failed")
        return _unavailable_snapshot(
            device_index,
            identity_receipt.get("execution_error")
            or f"nvidia-smi identity exit code {identity_receipt.get('exit_code')!r}",
            nvidia_smi_binary,
            query_receipt,
        )
    if not _execute_command(process_receipt, command_fn):
        _finish_query_receipt(query_receipt, "process_command_failed")
        return _unavailable_snapshot(
            device_index,
            process_receipt.get("execution_error")
            or f"nvidia-smi process exit code {process_receipt.get('exit_code')!r}",
            nvidia_smi_binary,
            query_receipt,
        )

    try:
        observed_index, device_uuid, device_name = _parse_identity_output(
            identity_receipt["raw_stdout"],
            device_index,
        )
        processes = _parse_process_output(
            process_receipt["raw_stdout"],
            bool(include_process_name),
        )
    except ValueError as error:
        _finish_query_receipt(query_receipt, "parse_failed")
        return _unavailable_snapshot(
            device_index,
            str(error),
            nvidia_smi_binary,
            query_receipt,
        )

    _finish_query_receipt(query_receipt, "ok")
    return {
        "available": True,
        "error": None,
        "device_index": observed_index,
        "device_uuid": device_uuid,
        "device_name": device_name,
        "processes": processes,
        "process_count": len(processes),
        "sampled_at_unix_seconds": query_receipt[
            "query_started_at_unix_seconds"
        ],
        "sampled_at_monotonic_seconds": query_receipt[
            "query_started_at_monotonic_seconds"
        ],
        "query_started_at_unix_seconds": query_receipt[
            "query_started_at_unix_seconds"
        ],
        "query_finished_at_unix_seconds": query_receipt[
            "query_finished_at_unix_seconds"
        ],
        "query_started_at_monotonic_seconds": query_receipt[
            "query_started_at_monotonic_seconds"
        ],
        "query_finished_at_monotonic_seconds": query_receipt[
            "query_finished_at_monotonic_seconds"
        ],
        "boot_id": boot_id,
        "nvidia_smi_binary": dict(nvidia_smi_binary),
        "query_receipt": query_receipt,
    }


def _finite_nonnegative_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _raw_stream_errors(command_receipt, stream_name):
    errors = []
    prefix = f"raw_{stream_name}"
    encoded = command_receipt.get(f"{prefix}_base64")
    if not isinstance(encoded, str):
        return [f"{prefix}_base64 is missing"]
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as error:
        return [f"{prefix}_base64 is invalid: {error}"]
    expected_text = None
    try:
        expected_text = raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if command_receipt.get(prefix) != expected_text:
        errors.append(f"{prefix} differs from exact base64 bytes")
    byte_count = command_receipt.get(f"{prefix}_bytes")
    if not _is_json_int(byte_count) or byte_count != len(raw):
        errors.append(f"{prefix}_bytes is invalid")
    digest = command_receipt.get(f"{prefix}_sha256")
    if digest != hashlib.sha256(raw).hexdigest():
        errors.append(f"{prefix}_sha256 differs from exact bytes")
    return errors


def gpu_compute_snapshot_errors(snapshot):
    """Reparse one raw query receipt and validate every derived field."""

    if not isinstance(snapshot, dict):
        return ["GPU compute snapshot must be a JSON object"]
    errors = []
    boot_id = snapshot.get("boot_id")
    if (
        not isinstance(boot_id, str)
        or re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            boot_id,
        )
        is None
    ):
        errors.append("GPU compute snapshot boot ID is invalid")
    receipt = snapshot.get("query_receipt")
    if not isinstance(receipt, dict):
        return ["GPU compute snapshot query_receipt is missing"]
    if (
        not _is_json_int(receipt.get("schema_version"))
        or receipt.get("schema_version") != GPU_QUERY_RECEIPT_SCHEMA_VERSION
    ):
        errors.append("GPU query receipt schema_version is invalid")
    device_index = receipt.get("device_index")
    if not _is_json_int(device_index) or device_index < 0:
        errors.append("GPU query receipt device_index is invalid")
        device_index = None
    if (
        device_index is not None
        and (
            not _is_json_int(snapshot.get("device_index"))
            or snapshot.get("device_index") != device_index
        )
    ):
        errors.append("GPU snapshot device_index differs from query receipt")
    if not isinstance(receipt.get("include_process_name"), bool):
        errors.append("GPU query receipt include_process_name is invalid")
    include_process_name = receipt.get("include_process_name") is True
    if receipt.get("locale") != GPU_QUERY_LOCALE:
        errors.append("GPU query receipt locale is not fixed to C")
    binary_metadata = snapshot.get("nvidia_smi_binary")
    binary_metadata_errors = trusted_nvidia_smi_metadata_errors(
        binary_metadata
    )
    errors.extend(binary_metadata_errors)
    expected_executable = (
        binary_metadata.get("resolved_path")
        if isinstance(binary_metadata, dict)
        else TRUSTED_NVIDIA_SMI_PATH
    )
    if (
        expected_executable != TRUSTED_NVIDIA_SMI_PATH
        or receipt.get("resolved_executable") != expected_executable
    ):
        errors.append("GPU query receipt resolved executable is not trusted")

    query_times = {}
    for clock in ("unix", "monotonic"):
        start_field = f"query_started_at_{clock}_seconds"
        finish_field = f"query_finished_at_{clock}_seconds"
        start = receipt.get(start_field)
        finish = receipt.get(finish_field)
        if not _finite_nonnegative_number(start):
            errors.append(f"GPU query receipt {start_field} is invalid")
        if not _finite_nonnegative_number(finish):
            errors.append(f"GPU query receipt {finish_field} is invalid")
        if (
            _finite_nonnegative_number(start)
            and _finite_nonnegative_number(finish)
            and float(finish) < float(start)
        ):
            errors.append(f"GPU query receipt {clock} time moved backwards")
        query_times[clock] = (start, finish)
        if (
            not _finite_nonnegative_number(snapshot.get(start_field))
            or not _finite_nonnegative_number(snapshot.get(finish_field))
            or snapshot.get(start_field) != start
            or snapshot.get(finish_field) != finish
        ):
            errors.append(f"GPU snapshot {clock} query times differ from receipt")
    if (
        not _finite_nonnegative_number(
            snapshot.get("sampled_at_unix_seconds")
        )
        or not _finite_nonnegative_number(
            snapshot.get("sampled_at_monotonic_seconds")
        )
        or snapshot.get("sampled_at_unix_seconds")
        != receipt.get("query_started_at_unix_seconds")
        or snapshot.get("sampled_at_monotonic_seconds")
        != receipt.get("query_started_at_monotonic_seconds")
    ):
        errors.append("GPU snapshot sampled times differ from query start")

    commands = receipt.get("commands")
    if not isinstance(commands, list) or len(commands) != 2:
        errors.append("GPU query receipt must contain exactly two commands")
        commands = []
    expected_commands = (
        _new_query_receipt(
            device_index,
            include_process_name=include_process_name,
        )["commands"]
        if device_index is not None
        else []
    )
    for index, command_receipt in enumerate(commands):
        context = f"GPU query command[{index}]"
        if not isinstance(command_receipt, dict):
            errors.append(f"{context} must be a JSON object")
            continue
        if expected_commands:
            expected = expected_commands[index]
            if command_receipt.get("label") != expected["label"]:
                errors.append(f"{context} label is invalid")
            if command_receipt.get("command") != expected["command"]:
                errors.append(f"{context} argv is not the fixed command")
        if command_receipt.get("locale") != GPU_QUERY_LOCALE:
            errors.append(f"{context} locale is not fixed to C")
        if command_receipt.get("resolved_executable") != expected_executable:
            errors.append(f"{context} resolved executable is not trusted")
        attempted = command_receipt.get("attempted")
        if not isinstance(attempted, bool):
            errors.append(f"{context} attempted flag is invalid")
            attempted = False
        errors.extend(
            f"{context} {error}"
            for stream in ("stdout", "stderr")
            for error in _raw_stream_errors(command_receipt, stream)
        )
        exit_code = command_receipt.get("exit_code")
        if exit_code is not None and not _is_json_int(exit_code):
            errors.append(f"{context} exit_code is invalid")
        execution_error = command_receipt.get("execution_error")
        if execution_error is not None and not isinstance(execution_error, str):
            errors.append(f"{context} execution_error is invalid")
        if attempted:
            if exit_code is None and not (
                isinstance(execution_error, str) and execution_error
            ):
                errors.append(
                    f"{context} has neither exit code nor execution error"
                )
            if exit_code == 0:
                stderr_bytes = command_receipt.get("raw_stderr_bytes")
                expected_execution_error = (
                    NVIDIA_SMI_STDERR_ERROR if stderr_bytes else None
                )
                if execution_error != expected_execution_error:
                    errors.append(
                        f"{context} stderr/exit execution status is inconsistent"
                    )
            for clock in ("unix", "monotonic"):
                start = command_receipt.get(
                    f"started_at_{clock}_seconds"
                )
                finish = command_receipt.get(
                    f"finished_at_{clock}_seconds"
                )
                if not _finite_nonnegative_number(start) or not _finite_nonnegative_number(finish):
                    errors.append(f"{context} {clock} times are invalid")
                    continue
                query_start, query_finish = query_times[clock]
                if (
                    _finite_nonnegative_number(query_start)
                    and _finite_nonnegative_number(query_finish)
                    and not (
                        float(query_start)
                        <= float(start)
                        <= float(finish)
                        <= float(query_finish)
                    )
                ):
                    errors.append(f"{context} {clock} times escape query bounds")
        else:
            for field in (
                "started_at_unix_seconds",
                "finished_at_unix_seconds",
                "started_at_monotonic_seconds",
                "finished_at_monotonic_seconds",
                "exit_code",
                "execution_error",
            ):
                if command_receipt.get(field) is not None:
                    errors.append(f"{context} unattempted {field} must be null")
            if (
                command_receipt.get("raw_stdout_bytes") != 0
                or command_receipt.get("raw_stderr_bytes") != 0
            ):
                errors.append(f"{context} unattempted raw output must be empty")

    if len(commands) == 2 and all(
        isinstance(item, dict) and item.get("attempted") is True
        for item in commands
    ):
        for clock in ("unix", "monotonic"):
            first_finished = commands[0].get(f"finished_at_{clock}_seconds")
            second_started = commands[1].get(f"started_at_{clock}_seconds")
            if (
                _finite_nonnegative_number(first_finished)
                and _finite_nonnegative_number(second_started)
                and float(second_started) < float(first_finished)
            ):
                errors.append(
                    f"GPU query commands overlap or reverse in {clock} time"
                )

    status = receipt.get("status")
    allowed_statuses = {
        "ok",
        "not_run",
        "binary_metadata_unavailable",
        "binary_metadata_invalid",
        "identity_command_failed",
        "process_command_failed",
        "parse_failed",
    }
    if status not in allowed_statuses:
        errors.append("GPU query receipt status is invalid")
    parsed = None
    parse_error = None
    if len(commands) == 2:
        identity_receipt, process_receipt = commands
        both_succeeded = all(
            isinstance(item, dict)
            and item.get("attempted") is True
            and _is_json_int(item.get("exit_code"))
            and item.get("exit_code") == 0
            and item.get("execution_error") is None
            for item in commands
        )
        if both_succeeded and device_index is not None:
            try:
                observed_index, device_uuid, device_name = (
                    _parse_identity_output(
                        identity_receipt.get("raw_stdout"),
                        device_index,
                    )
                )
                processes = _parse_process_output(
                    process_receipt.get("raw_stdout"),
                    include_process_name,
                )
                parsed = (
                    observed_index,
                    device_uuid,
                    device_name,
                    processes,
                )
            except ValueError as error:
                parse_error = str(error)

        identity_attempted = identity_receipt.get("attempted") is True
        process_attempted = process_receipt.get("attempted") is True
        identity_succeeded = (
            identity_attempted
            and _is_json_int(identity_receipt.get("exit_code"))
            and identity_receipt.get("exit_code") == 0
            and identity_receipt.get("execution_error") is None
        )
        process_succeeded = (
            process_attempted
            and _is_json_int(process_receipt.get("exit_code"))
            and process_receipt.get("exit_code") == 0
            and process_receipt.get("execution_error") is None
        )
        expected_status = None
        if not identity_attempted and not process_attempted:
            if binary_metadata_errors:
                expected_status = {
                    "binary_metadata_unavailable",
                    "binary_metadata_invalid",
                    "not_run",
                }
            else:
                expected_status = {"not_run"}
        elif not identity_succeeded:
            expected_status = {"identity_command_failed"}
            if process_attempted:
                errors.append(
                    "process query ran after the identity command failed"
                )
        elif not process_attempted or not process_succeeded:
            expected_status = {"process_command_failed"}
        elif parse_error is not None:
            expected_status = {"parse_failed"}
        else:
            expected_status = {"ok"}
        if status not in expected_status:
            errors.append("GPU query receipt status disagrees with execution")

    if parsed is not None:
        observed_index, device_uuid, device_name, processes = parsed
        if status != "ok" or snapshot.get("available") is not True:
            errors.append("parseable successful query is not marked available/ok")
        if snapshot.get("error") is not None:
            errors.append("available GPU snapshot contains an error")
        if (
            not _is_json_int(snapshot.get("device_index"))
            or snapshot.get("device_index") != observed_index
            or snapshot.get("device_uuid") != device_uuid
            or snapshot.get("device_name") != device_name
        ):
            errors.append("GPU identity fields differ from raw stdout")
        if snapshot.get("processes") != processes:
            errors.append("GPU process fields differ from raw stdout")
        snapshot_processes = snapshot.get("processes")
        if isinstance(snapshot_processes, list):
            for process_index, process in enumerate(snapshot_processes):
                if not isinstance(process, dict):
                    errors.append(
                        f"GPU process[{process_index}] is not a JSON object"
                    )
                    continue
                if (
                    not _is_json_int(process.get("host_pid"))
                    or process.get("host_pid") <= 0
                ):
                    errors.append(
                        f"GPU process[{process_index}] host_pid is invalid"
                    )
                if (
                    not _is_json_int(process.get("used_memory_mib"))
                    or process.get("used_memory_mib") <= 0
                ):
                    errors.append(
                        f"GPU process[{process_index}] used_memory_mib is invalid"
                    )
                if include_process_name and (
                    not isinstance(process.get("process_name"), str)
                    or not process.get("process_name")
                ):
                    errors.append(
                        f"GPU process[{process_index}] process_name is invalid"
                    )
        if (
            not _is_json_int(snapshot.get("process_count"))
            or snapshot.get("process_count") != len(processes)
        ):
            errors.append("GPU process_count differs from raw stdout")
    else:
        if snapshot.get("available") is not False:
            errors.append("unparseable/failed GPU query is not unavailable")
        if not isinstance(snapshot.get("error"), str) or not snapshot.get("error"):
            errors.append("unavailable GPU snapshot error is missing")
        if (
            snapshot.get("device_uuid") is not None
            or snapshot.get("device_name") is not None
            or snapshot.get("processes") != []
            or snapshot.get("process_count") is not None
        ):
            errors.append("unavailable GPU snapshot has derived process data")
        if parse_error is not None and status != "parse_failed":
            errors.append("raw parse failure does not match receipt status")
    return errors


def gpu_compute_snapshot_sequence_errors(
    samples,
    maximum_gap_seconds=None,
):
    """Require ordered, non-overlapping, independently timed query samples."""

    if not isinstance(samples, list) or not samples:
        return ["GPU snapshot sequence must be a non-empty list"]
    errors = []
    if maximum_gap_seconds is not None and (
        isinstance(maximum_gap_seconds, bool)
        or not isinstance(maximum_gap_seconds, (int, float))
        or not math.isfinite(float(maximum_gap_seconds))
        or float(maximum_gap_seconds) <= 0.0
    ):
        errors.append("GPU snapshot maximum gap is invalid")
        maximum_gap_seconds = None
    for index in range(1, len(samples)):
        previous = samples[index - 1]
        current = samples[index]
        if not isinstance(previous, dict) or not isinstance(current, dict):
            errors.append(f"GPU snapshot sequence item {index} is invalid")
            continue
        for clock in ("unix", "monotonic"):
            previous_start = previous.get(
                f"query_started_at_{clock}_seconds"
            )
            previous_finish = previous.get(
                f"query_finished_at_{clock}_seconds"
            )
            current_start = current.get(
                f"query_started_at_{clock}_seconds"
            )
            current_finish = current.get(
                f"query_finished_at_{clock}_seconds"
            )
            if not all(
                _finite_nonnegative_number(value)
                for value in (
                    previous_start,
                    previous_finish,
                    current_start,
                    current_finish,
                )
            ):
                errors.append(
                    f"GPU snapshot sequence {clock} times are invalid at {index}"
                )
                continue
            if not (
                float(previous_start) < float(current_start)
                and float(previous_finish) <= float(current_start)
                and float(previous_finish) < float(current_finish)
            ):
                errors.append(
                    f"GPU snapshot sequence is duplicated, overlapping, or "
                    f"reversed in {clock} time at {index}"
                )
            elif (
                maximum_gap_seconds is not None
                and float(current_start) - float(previous_finish)
                > float(maximum_gap_seconds)
            ):
                errors.append(
                    f"GPU snapshot sequence exceeds maximum {clock} gap "
                    f"at {index}"
                )
    return errors


def gpu_compute_snapshot_observation_span_seconds(samples):
    if not isinstance(samples, list) or not samples:
        return None
    first = samples[0]
    last = samples[-1]
    if not isinstance(first, dict) or not isinstance(last, dict):
        return None
    start = first.get("query_started_at_monotonic_seconds")
    finish = last.get("query_finished_at_monotonic_seconds")
    if (
        not _finite_nonnegative_number(start)
        or not _finite_nonnegative_number(finish)
        or float(finish) < float(start)
    ):
        return None
    return float(finish) - float(start)


def gpu_compute_snapshot_maximum_gap_seconds(samples):
    if not isinstance(samples, list) or len(samples) < 2:
        return 0.0 if isinstance(samples, list) and samples else None
    gaps = []
    for index in range(1, len(samples)):
        previous = samples[index - 1]
        current = samples[index]
        if not isinstance(previous, dict) or not isinstance(current, dict):
            return None
        previous_finish = previous.get(
            "query_finished_at_monotonic_seconds"
        )
        current_start = current.get("query_started_at_monotonic_seconds")
        if (
            not _finite_nonnegative_number(previous_finish)
            or not _finite_nonnegative_number(current_start)
            or float(current_start) < float(previous_finish)
        ):
            return None
        gaps.append(float(current_start) - float(previous_finish))
    return max(gaps, default=0.0)


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
    errors.extend(gpu_compute_snapshot_errors(snapshot))
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
        "query_started_at_unix_seconds": snapshot.get(
            "query_started_at_unix_seconds"
        ),
        "query_finished_at_unix_seconds": snapshot.get(
            "query_finished_at_unix_seconds"
        ),
        "query_started_at_monotonic_seconds": snapshot.get(
            "query_started_at_monotonic_seconds"
        ),
        "query_finished_at_monotonic_seconds": snapshot.get(
            "query_finished_at_monotonic_seconds"
        ),
        "boot_id": snapshot.get("boot_id"),
        "run_nonce": secrets.token_hex(16),
        "nvidia_smi_binary": snapshot.get("nvidia_smi_binary"),
        "query_receipt": snapshot.get("query_receipt"),
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
    snapshot_fields = {
        field: report.get(field)
        for field in (
            "available",
            "error",
            "device_index",
            "device_uuid",
            "device_name",
            "processes",
            "process_count",
            "sampled_at_unix_seconds",
            "sampled_at_monotonic_seconds",
            "query_started_at_unix_seconds",
            "query_finished_at_unix_seconds",
            "query_started_at_monotonic_seconds",
            "query_finished_at_monotonic_seconds",
            "boot_id",
            "nvidia_smi_binary",
            "query_receipt",
        )
    }
    errors.extend(
        f"pre-run GPU raw query: {error}"
        for error in gpu_compute_snapshot_errors(snapshot_fields)
    )
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
        if isinstance(interval_seconds, bool):
            raise ValueError("GPU process monitor interval must be a number")
        interval_seconds = float(interval_seconds)
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
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
        first_boot_id = samples[0].get("boot_id") if samples else None
        boot_id_consistent = (
            bool(available)
            and isinstance(first_boot_id, str)
            and all(sample.get("boot_id") == first_boot_id for sample in available)
        )
        first_nvidia_smi_binary = (
            samples[0].get("nvidia_smi_binary") if samples else None
        )
        nvidia_smi_binary_validation_errors = []
        snapshot_validation_errors = []
        for sample_index, sample in enumerate(samples):
            snapshot_validation_errors.extend(
                f"samples[{sample_index}]: {error}"
                for error in gpu_compute_snapshot_errors(sample)
            )
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
        sample_sequence_validation_errors = (
            gpu_compute_snapshot_sequence_errors(
                samples,
                self.interval_seconds
                + GPU_QUERY_CADENCE_TOLERANCE_SECONDS,
            )
        )
        observation_span_seconds = (
            gpu_compute_snapshot_observation_span_seconds(samples)
        )
        maximum_sample_gap_seconds = (
            gpu_compute_snapshot_maximum_gap_seconds(samples)
        )
        return {
            "schema_version": GPU_PROCESS_MONITOR_SCHEMA_VERSION,
            "device_index": identity[0] if identity_consistent else self.device_index,
            "device_uuid": identity[1],
            "device_name": identity[2],
            "identity_consistent": identity_consistent,
            "boot_id": first_boot_id,
            "boot_id_consistent": boot_id_consistent,
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
            "snapshot_validation_errors": snapshot_validation_errors,
            "sample_sequence_validation_errors": (
                sample_sequence_validation_errors
            ),
            "observation_span_seconds": observation_span_seconds,
            "cadence_tolerance_seconds": (
                GPU_QUERY_CADENCE_TOLERANCE_SECONDS
            ),
            "maximum_sample_gap_seconds": maximum_sample_gap_seconds,
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
                and boot_id_consistent
                and nvidia_smi_binary_fixed_valid
                and nvidia_smi_binary_consistent
                and not sample_validation_errors
                and not snapshot_validation_errors
                and not sample_sequence_validation_errors
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
