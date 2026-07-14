import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from ovi.gpu_process_monitor import (
    TRUSTED_NVIDIA_SMI_BYTES,
    TRUSTED_NVIDIA_SMI_PATH,
    TRUSTED_NVIDIA_SMI_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_ovi_eval_csv.py"
SPEC = importlib.util.spec_from_file_location("build_ovi_eval_csv_test", SCRIPT_PATH)
EVAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVAL
SPEC.loader.exec_module(EVAL)

GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"
GPU_NAME = "NVIDIA A100-SXM4-80GB"
PROMPT = "A fixed audiovisual benchmark prompt."


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


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def monitor(*, contention=False):
    process_count = 2 if contention else 1
    processes = [
        {"host_pid": 700 + index, "used_memory_mib": 1000}
        for index in range(process_count)
    ]
    samples = [
        {
            "available": True,
            "error": None,
            "device_index": 0,
            "device_uuid": GPU_UUID,
            "device_name": GPU_NAME,
            "process_count": process_count,
            "processes": processes,
            "sampled_at_unix_seconds": float(index),
            "nvidia_smi_binary": trusted_binary(),
        }
        for index in range(2)
    ]
    return {
        "device_index": 0,
        "device_uuid": GPU_UUID,
        "device_name": GPU_NAME,
        "identity_consistent": True,
        "nvidia_smi_binary": trusted_binary(),
        "nvidia_smi_binary_fixed_valid": True,
        "nvidia_smi_binary_consistent": True,
        "nvidia_smi_binary_validation_errors": [],
        "sample_validation_errors": [],
        "sample_count": 2,
        "available_sample_count": 2,
        "unavailable_sample_count": 0,
        "min_process_count": process_count,
        "max_process_count": process_count,
        "distinct_host_pids": [item["host_pid"] for item in processes],
        "single_distinct_host_pid": not contention,
        "exact_singleton_process_per_sample": not contention,
        "contention_detected": contention,
        "no_process_detected": False,
        "valid_for_benchmark": not contention,
        "collection_errors": [],
        "samples": samples,
    }


class RunFactory:
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.manifest = manifest

    def make(
        self,
        method_id,
        *,
        denoise=(10.0, 30.0, 20.0),
        total=(20.0, 40.0, 30.0),
        allocated_gib=(1.0, 3.0, 2.0),
        reserved_gib=(2.0, 4.0, 3.0),
        dirty=False,
        contention=False,
        commit="a" * 40,
        indices=(0, 1, 2),
        nan_field=None,
        artifact_mismatch=False,
    ):
        method = next(
            item for item in self.manifest["methods"]
            if item["method_id"] == method_id
        )
        run_dir = self.root / f"run-{method_id}-{len(list(self.root.iterdir()))}"
        run_dir.mkdir()

        checkpoint = {
            "schema_version": 1,
            "files": {
                "Ovi/model.safetensors": {
                    "bytes": 123,
                    "sha256": "1" * 64,
                },
                "Wan/vae.pth": {
                    "bytes": 456,
                    "sha256": "2" * 64,
                },
            },
        }
        checkpoint_path = run_dir / "checkpoint_manifest.json"
        write_json(checkpoint_path, checkpoint)

        pre_run_gpu = {
            "schema_version": 1,
            "check_type": "pre_run_idle",
            "physical_device_index": 0,
            "available": True,
            "error": None,
            "device_index": 0,
            "device_uuid": GPU_UUID,
            "device_name": GPU_NAME,
            "processes": [],
            "process_count": 0,
            "idle": True,
            "valid_for_run": True,
            "errors": [],
            "checked_at_utc": "2026-07-14T00:00:00+00:00",
            "cuda_visible_devices": "0",
            "sampled_at_unix_seconds": 1.0,
            "sampled_at_monotonic_seconds": 1.0,
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "run_nonce": "1" * 32,
            "nvidia_smi_binary": trusted_binary(),
        }
        pre_run_gpu_path = run_dir / "pre_run_gpu.json"
        write_json(pre_run_gpu_path, pre_run_gpu)

        environment = {
            **self.manifest["fixed_protocol"],
            **method["expected_environment"],
            "git_commit": commit,
            "git_dirty": dirty,
            "pre_run_gpu_valid": True,
            "gpu_physical_index": 0,
            "gpu_uuid": GPU_UUID,
            "gpu_name": GPU_NAME,
            "gpu": GPU_NAME,
            "cuda_visible_devices": "0",
            "engine_load_seconds": 12.5,
            "expected_measurement_records": 3,
            "expected_warmup_records": 1,
            "run_id": run_dir.name,
            "pre_run_gpu_sha256": sha256(pre_run_gpu_path),
            "evidence_file_sha256": {
                "checkpoint_manifest.json": sha256(checkpoint_path),
                "pre_run_gpu.json": sha256(pre_run_gpu_path),
            },
        }
        write_json(run_dir / "environment.json", environment)

        timings = []
        reports = []
        for record_offset, measurement_index in enumerate(indices):
            artifact = run_dir / f"measurement-{record_offset}.mp4"
            artifact.write_bytes(f"artifact-{method_id}-{record_offset}".encode())
            artifact_hash = sha256(artifact)
            record = {
                "status": "ok",
                "record_type": "measurement",
                "benchmark_candidate": True,
                "benchmark_valid": False,
                "run_id": run_dir.name,
                "measurement_index": measurement_index,
                "prompt_index": 0,
                "sample_index": 0,
                "prompt": PROMPT,
                "seed": 103,
                "sample_steps": 50,
                "attention_method": environment.get("attention_method"),
                "use_cfg_cache": environment.get("use_cfg_cache"),
                "use_block_cache": environment.get("use_block_cache"),
                "requested_video_frame_height_width": [720, 720],
                "actual_video_frame_height_width": [704, 704],
                "generated_video_shape": [3, 121, 704, 704],
                "generated_audio_shape": [80000],
                "denoise_seconds": denoise[record_offset],
                "total_generation_seconds": total[record_offset],
                "save_video_seconds": 2.0,
                "artifact_ready_seconds": total[record_offset] + 2.0,
                "output_hash_seconds": 0.1,
                "peak_memory_allocated_bytes": allocated_gib[record_offset]
                * EVAL.GIB,
                "peak_memory_reserved_bytes": reserved_gib[record_offset]
                * EVAL.GIB,
                "output_path": str(artifact.resolve()),
                "output_sha256": artifact_hash,
                "gpu_process_monitor": monitor(contention=contention),
            }
            if nan_field is not None and record_offset == 1:
                record[nan_field] = float("nan")
            timings.append(record)
            write_json(artifact.with_suffix(".metrics.json"), record)
            reports.append(
                {
                    "path": str(artifact.resolve()),
                    "sha256": (
                        "f" * 64
                        if artifact_mismatch and record_offset == 1
                        else artifact_hash
                    ),
                    "status": "ok",
                    "errors": [],
                }
            )

        (run_dir / "timings.jsonl").write_text(
            "".join(
                json.dumps(record, sort_keys=True, allow_nan=True) + "\n"
                for record in timings
            ),
            encoding="utf-8",
        )
        verification = {
            "status": "ok",
            "benchmark_valid": True,
            "artifact_count": 3,
            "artifacts": reports,
            "protocol": {
                "status": "ok",
                "errors": [],
                "benchmark_candidate": True,
                "benchmark_valid": True,
                "expected_warmup_records": 1,
                "observed_warmup_records": 1,
                "expected_measurement_records": 3,
                "observed_measurement_records": 3,
            },
        }
        write_json(run_dir / "verification.json", verification)
        return run_dir


class EvalCsvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = EVAL.load_manifest()
        self.factory = RunFactory(self.root, self.manifest)

    def method(self, method_id):
        return next(
            item for item in self.manifest["methods"]
            if item["method_id"] == method_id
        )

    def test_manifest_has_seven_required_slots_and_optional_block(self):
        required = [
            item["method_id"] for item in self.manifest["methods"]
            if item["required"]
        ]
        optional = [
            item["method_id"] for item in self.manifest["methods"]
            if not item["required"]
        ]
        self.assertEqual(tuple(required), EVAL.REQUIRED_METHOD_IDS)
        self.assertEqual(tuple(optional), EVAL.OPTIONAL_METHOD_IDS)
        for method_id in (
            "radial_conservative",
            "radial_aggressive",
            "best_sparse_cfg",
        ):
            self.assertEqual(self.method(method_id)["implementation_status"], "pending")

    def test_explicit_mapping_rejects_bare_path_unknown_and_duplicates(self):
        allowed = [item["method_id"] for item in self.manifest["methods"]]
        with self.assertRaisesRegex(EVAL.EvaluationError, "METHOD_ID=RUN_DIR"):
            EVAL.parse_run_mappings(["/some/latest/run"], allowed)
        with self.assertRaisesRegex(EVAL.EvaluationError, "unknown"):
            EVAL.parse_run_mappings(["made_up=/some/run"], allowed)
        with self.assertRaisesRegex(EVAL.EvaluationError, "duplicate"):
            EVAL.parse_run_mappings(["dense=/one", "dense=/two"], allowed)
        parsed = EVAL.parse_run_mappings(["dense=/chosen/exact/run"], allowed)
        self.assertEqual(parsed, {"dense": Path("/chosen/exact/run")})

    def test_dirty_run_is_rejected_even_if_verification_claims_valid(self):
        run_dir = self.factory.make("dense", dirty=True)
        with self.assertRaisesRegex(EVAL.EvaluationError, "git_dirty"):
            EVAL.validate_run(self.method("dense"), run_dir, self.manifest["fixed_protocol"])

    def test_benchmark_valid_must_be_explicitly_true(self):
        run_dir = self.factory.make("dense")
        verification_path = run_dir / "verification.json"
        verification = json.loads(verification_path.read_text())
        verification["benchmark_valid"] = False
        write_json(verification_path, verification)
        with self.assertRaisesRegex(EVAL.EvaluationError, "benchmark_valid=true"):
            EVAL.validate_run(self.method("dense"), run_dir, self.manifest["fixed_protocol"])

    def test_contention_is_rejected_even_if_verification_claims_valid(self):
        run_dir = self.factory.make("dense", contention=True)
        with self.assertRaises(EVAL.EvaluationError):
            EVAL.validate_run(self.method("dense"), run_dir, self.manifest["fixed_protocol"])

    def test_pre_run_untrusted_nvidia_smi_binary_is_rejected(self):
        run_dir = self.factory.make("dense")
        path = run_dir / "pre_run_gpu.json"
        payload = json.loads(path.read_text())
        payload["nvidia_smi_binary"]["sha256"] = "0" * 64
        write_json(path, payload)
        with self.assertRaisesRegex(EVAL.EvaluationError, "nvidia-smi"):
            EVAL.validate_run(
                self.method("dense"),
                run_dir,
                self.manifest["fixed_protocol"],
            )

    def test_sample_nvidia_smi_binary_must_equal_pre_run_evidence(self):
        run_dir = self.factory.make("dense")
        records = [
            json.loads(line)
            for line in (run_dir / "timings.jsonl").read_text().splitlines()
        ]
        records[0]["gpu_process_monitor"]["samples"][1][
            "nvidia_smi_binary"
        ]["inode"] += 1
        (run_dir / "timings.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EVAL.EvaluationError, "nvidia-smi"):
            EVAL.validate_run(
                self.method("dense"),
                run_dir,
                self.manifest["fixed_protocol"],
            )

    def test_pre_run_requires_canonical_fields_and_unambiguous_cvd(self):
        for label, mutate in (
            (
                "missing-run-nonce",
                lambda payload: payload.pop("run_nonce"),
            ),
            (
                "missing-timestamp",
                lambda payload: payload.pop("sampled_at_monotonic_seconds"),
            ),
            (
                "bad-schema",
                lambda payload: payload.__setitem__("schema_version", 0),
            ),
            (
                "bad-check-type",
                lambda payload: payload.__setitem__("check_type", "old"),
            ),
            (
                "ambiguous-cvd",
                lambda payload: payload.__setitem__(
                    "cuda_visible_devices", "1,0"
                ),
            ),
        ):
            with self.subTest(label=label):
                run_dir = self.factory.make("dense")
                path = run_dir / "pre_run_gpu.json"
                payload = json.loads(path.read_text())
                mutate(payload)
                write_json(path, payload)
                if label == "ambiguous-cvd":
                    environment_path = run_dir / "environment.json"
                    environment = json.loads(environment_path.read_text())
                    environment["cuda_visible_devices"] = "1,0"
                    write_json(environment_path, environment)
                with self.assertRaises(EVAL.EvaluationError):
                    EVAL.validate_run(
                        self.method("dense"),
                        run_dir,
                        self.manifest["fixed_protocol"],
                    )

    def test_bool_forged_gpu_monitor_integers_are_rejected(self):
        mutations = (
            ("process-count", ("samples", 0, "process_count")),
            ("device-index", ("samples", 0, "device_index")),
            ("host-pid", ("samples", 0, "processes", 0, "host_pid")),
            (
                "used-memory",
                ("samples", 0, "processes", 0, "used_memory_mib"),
            ),
            ("min-count", ("min_process_count",)),
            ("max-count", ("max_process_count",)),
            ("unavailable-count", ("unavailable_sample_count",)),
        )
        for label, path_parts in mutations:
            with self.subTest(label=label):
                run_dir = self.factory.make("dense")
                timing_path = run_dir / "timings.jsonl"
                records = [
                    json.loads(line)
                    for line in timing_path.read_text().splitlines()
                ]
                target = records[0]["gpu_process_monitor"]
                for part in path_parts[:-1]:
                    target = target[part]
                target[path_parts[-1]] = (
                    False if "unavailable" in label or "device" in label else True
                )
                timing_path.write_text(
                    "".join(
                        json.dumps(record, sort_keys=True) + "\n"
                        for record in records
                    ),
                    encoding="utf-8",
                )
                with self.assertRaises(EVAL.EvaluationError):
                    EVAL.validate_run(
                        self.method("dense"),
                        run_dir,
                        self.manifest["fixed_protocol"],
                    )

    def test_measurement_count_must_be_exactly_three(self):
        run_dir = self.factory.make("dense")
        lines = (run_dir / "timings.jsonl").read_text().splitlines()
        (run_dir / "timings.jsonl").write_text("\n".join(lines[:2]) + "\n")
        with self.assertRaisesRegex(EVAL.EvaluationError, "exactly three"):
            EVAL.validate_run(self.method("dense"), run_dir, self.manifest["fixed_protocol"])

    def test_duplicate_measurement_index_is_rejected(self):
        run_dir = self.factory.make("dense", indices=(0, 0, 2))
        with self.assertRaisesRegex(EVAL.EvaluationError, "duplicated"):
            EVAL.validate_run(self.method("dense"), run_dir, self.manifest["fixed_protocol"])

    def test_nan_timing_is_rejected(self):
        run_dir = self.factory.make("dense", nan_field="denoise_seconds")
        with self.assertRaisesRegex(EVAL.EvaluationError, "must be finite"):
            EVAL.validate_run(self.method("dense"), run_dir, self.manifest["fixed_protocol"])

    def test_timing_values_must_match_the_per_artifact_metrics_sidecar(self):
        run_dir = self.factory.make("dense")
        records = [
            json.loads(line)
            for line in (run_dir / "timings.jsonl").read_text().splitlines()
        ]
        records[1]["denoise_seconds"] = 0.001
        records[1]["total_generation_seconds"] = 0.002
        records[1]["artifact_ready_seconds"] = 2.002
        records[1]["peak_memory_allocated_bytes"] = 1
        records[1]["peak_memory_reserved_bytes"] = 1
        (run_dir / "timings.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
        with self.assertRaisesRegex(EVAL.EvaluationError, "metrics sidecar"):
            EVAL.validate_run(self.method("dense"), run_dir, self.manifest["fixed_protocol"])

    def test_artifact_hash_must_match_verification_and_disk(self):
        run_dir = self.factory.make("dense", artifact_mismatch=True)
        with self.assertRaisesRegex(EVAL.EvaluationError, "verification report"):
            EVAL.validate_run(self.method("dense"), run_dir, self.manifest["fixed_protocol"])

    def test_cross_method_commit_mismatch_is_rejected(self):
        dense = self.factory.make("dense", commit="a" * 40)
        cfg = self.factory.make("dense_cfg_cache", commit="b" * 40)
        with self.assertRaisesRegex(EVAL.EvaluationError, "comparison field git_commit"):
            EVAL.build_rows(
                self.manifest,
                {"dense": dense, "dense_cfg_cache": cfg},
            )

    def test_medians_gib_and_dense_speedup_use_three_measurements(self):
        dense = self.factory.make("dense")
        cfg = self.factory.make(
            "dense_cfg_cache",
            denoise=(5.0, 15.0, 10.0),
            total=(10.0, 20.0, 15.0),
            allocated_gib=(0.5, 1.5, 1.0),
            reserved_gib=(1.0, 2.0, 1.5),
        )
        rows = EVAL.build_rows(
            self.manifest,
            {"dense": dense, "dense_cfg_cache": cfg},
        )
        by_id = {row["method_id"]: row for row in rows}
        self.assertEqual(by_id["dense"]["denoise_seconds_median"], 20.0)
        self.assertEqual(by_id["dense"]["total_generation_seconds_median"], 30.0)
        self.assertEqual(by_id["dense"]["peak_memory_allocated_gib_median"], 2.0)
        self.assertEqual(by_id["dense"]["peak_memory_reserved_gib_median"], 3.0)
        self.assertEqual(by_id["dense"]["total_speedup_vs_dense"], 1.0)
        self.assertEqual(by_id["dense_cfg_cache"]["denoise_seconds_median"], 10.0)
        self.assertEqual(by_id["dense_cfg_cache"]["total_generation_seconds_median"], 15.0)
        self.assertEqual(by_id["dense_cfg_cache"]["denoise_speedup_vs_dense"], 2.0)
        self.assertEqual(by_id["dense_cfg_cache"]["total_speedup_vs_dense"], 2.0)

    def test_missing_quality_and_manual_fields_stay_blank_and_pending(self):
        dense = self.factory.make("dense")
        rows = EVAL.build_rows(self.manifest, {"dense": dense})
        by_id = {row["method_id"]: row for row in rows}
        self.assertEqual(by_id["dense"]["timing_status"], "valid")
        self.assertEqual(by_id["dense"]["status"], "pending")
        self.assertEqual(by_id["dense"]["quality_score"], "")
        self.assertEqual(by_id["dense"]["manual_review"], "")
        self.assertEqual(by_id["radial_conservative"]["status"], "pending")
        self.assertEqual(by_id["radial_conservative"]["measurement_count"], "")

    def test_pending_method_cannot_be_mapped_to_a_fabricated_run(self):
        # Build a structurally plausible run without changing the checked-in
        # pending status.  The aggregator must stop before accepting it.
        pending = self.method("radial_conservative")
        pending["implementation_status"] = "ready"
        try:
            run_dir = self.factory.make("radial_conservative")
        finally:
            pending["implementation_status"] = "pending"
        with self.assertRaisesRegex(EVAL.EvaluationError, "still marked pending"):
            EVAL.build_rows(self.manifest, {"radial_conservative": run_dir})

    def test_csv_writer_keeps_pending_numeric_fields_empty_not_zero(self):
        rows = EVAL.build_rows(self.manifest, {})
        output = self.root / "eval.csv"
        EVAL.write_csv(rows, output)
        with output.open(newline="", encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        self.assertEqual(len(parsed), 8)
        self.assertEqual(parsed[0]["status"], "pending")
        self.assertEqual(parsed[0]["total_generation_seconds_median"], "")
        self.assertEqual(parsed[0]["quality_score"], "")

    def test_existing_run_is_not_discovered_without_an_explicit_mapping(self):
        self.factory.make("dense")
        rows = EVAL.build_rows(self.manifest, {})
        dense = next(row for row in rows if row["method_id"] == "dense")
        self.assertEqual(dense["timing_status"], "pending")
        self.assertEqual(dense["run_dir"], "")


if __name__ == "__main__":
    unittest.main()
