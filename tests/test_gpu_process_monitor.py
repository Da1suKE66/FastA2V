import unittest

from ovi.gpu_process_monitor import GpuProcessMonitor


def sample(count, *, available=True):
    return {
        "available": available,
        "error": None if available else "synthetic query failure",
        "processes": [
            {"host_pid": index + 1, "used_memory_mib": 1000}
            for index in range(count)
        ],
        "process_count": count if available else None,
        "sampled_at_unix_seconds": 0.0,
    }


class GpuProcessMonitorTests(unittest.TestCase):
    def test_zero_or_one_process_is_valid(self):
        snapshots = iter((sample(0), sample(1)))
        monitor = GpuProcessMonitor(sample_fn=lambda _device: next(snapshots))
        monitor._sample_once()
        monitor._sample_once()
        summary = monitor.summary()
        self.assertTrue(summary["valid_for_benchmark"])
        self.assertFalse(summary["contention_detected"])
        self.assertEqual(summary["max_process_count"], 1)

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

    def test_collection_failure_is_fail_closed(self):
        snapshots = iter((sample(1), sample(0, available=False)))
        monitor = GpuProcessMonitor(sample_fn=lambda _device: next(snapshots))
        monitor._sample_once()
        monitor._sample_once()
        summary = monitor.summary()
        self.assertFalse(summary["valid_for_benchmark"])
        self.assertEqual(summary["unavailable_sample_count"], 1)

    def test_invalid_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "interval"):
            GpuProcessMonitor(interval_seconds=0)


if __name__ == "__main__":
    unittest.main()
