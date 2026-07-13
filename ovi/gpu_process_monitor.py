"""Low-frequency NVIDIA compute-process monitoring for benchmark evidence."""

import subprocess
import threading
import time


def query_gpu_compute_processes(device_index=0):
    """Return a JSON-serializable snapshot for one physical GPU index."""
    command = [
        "nvidia-smi",
        "--id",
        str(int(device_index)),
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "available": False,
            "error": repr(error),
            "processes": [],
            "process_count": None,
            "sampled_at_unix_seconds": time.time(),
        }

    processes = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            return {
                "available": False,
                "error": f"unexpected nvidia-smi row: {line!r}",
                "processes": [],
                "process_count": None,
                "sampled_at_unix_seconds": time.time(),
            }
        try:
            processes.append(
                {"host_pid": int(fields[0]), "used_memory_mib": int(fields[1])}
            )
        except ValueError:
            return {
                "available": False,
                "error": f"unparseable nvidia-smi row: {line!r}",
                "processes": [],
                "process_count": None,
                "sampled_at_unix_seconds": time.time(),
            }
    return {
        "available": True,
        "error": None,
        "processes": processes,
        "process_count": len(processes),
        "sampled_at_unix_seconds": time.time(),
    }


class GpuProcessMonitor:
    """Poll compute processes in a daemon thread while one generation runs."""

    def __init__(self, device_index=0, interval_seconds=5.0, sample_fn=None):
        interval_seconds = float(interval_seconds)
        if interval_seconds <= 0:
            raise ValueError("GPU process monitor interval must be positive")
        self.device_index = int(device_index)
        self.interval_seconds = interval_seconds
        self.sample_fn = sample_fn or query_gpu_compute_processes
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    def _sample_once(self):
        try:
            sample = self.sample_fn(self.device_index)
        except Exception as error:
            sample = {
                "available": False,
                "error": repr(error),
                "processes": [],
                "process_count": None,
                "sampled_at_unix_seconds": time.time(),
            }
        self._samples.append(sample)

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
        available = [sample for sample in self._samples if sample.get("available")]
        unavailable = [sample for sample in self._samples if not sample.get("available")]
        contaminated = [
            sample for sample in available if int(sample.get("process_count", 0)) > 1
        ]
        max_process_count = max(
            (int(sample.get("process_count", 0)) for sample in available),
            default=None,
        )
        return {
            "device_index": self.device_index,
            "interval_seconds": self.interval_seconds,
            "sample_count": len(self._samples),
            "available_sample_count": len(available),
            "unavailable_sample_count": len(unavailable),
            "max_process_count": max_process_count,
            "contention_detected": bool(contaminated),
            "valid_for_benchmark": bool(available) and not unavailable and not contaminated,
            "first_sample": self._samples[0] if self._samples else None,
            "last_sample": self._samples[-1] if self._samples else None,
            "contention_samples": contaminated[:5],
            "collection_errors": [sample.get("error") for sample in unavailable[:5]],
        }
