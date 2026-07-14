import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from ovi.gpu_process_monitor import (
    GPU_PROCESS_MONITOR_SCHEMA_VERSION,
    GPU_QUERY_CADENCE_TOLERANCE_SECONDS,
    TRUSTED_NVIDIA_SMI_BYTES,
    TRUSTED_NVIDIA_SMI_PATH,
    TRUSTED_NVIDIA_SMI_SHA256,
    build_pre_run_gpu_report,
    gpu_compute_snapshot_maximum_gap_seconds,
    gpu_compute_snapshot_observation_span_seconds,
    gpu_compute_snapshot_sequence_errors,
    query_gpu_compute_processes,
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
BOOT_ID = "11111111-2222-3333-4444-555555555555"


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


def shift_snapshot_times(snapshot, delta):
    for field in (
        "sampled_at_unix_seconds",
        "sampled_at_monotonic_seconds",
        "query_started_at_unix_seconds",
        "query_finished_at_unix_seconds",
        "query_started_at_monotonic_seconds",
        "query_finished_at_monotonic_seconds",
    ):
        snapshot[field] += delta
    receipt = snapshot["query_receipt"]
    for field in (
        "query_started_at_unix_seconds",
        "query_finished_at_unix_seconds",
        "query_started_at_monotonic_seconds",
        "query_finished_at_monotonic_seconds",
    ):
        receipt[field] += delta
    for command in receipt["commands"]:
        for field in (
            "started_at_unix_seconds",
            "finished_at_unix_seconds",
            "started_at_monotonic_seconds",
            "finished_at_monotonic_seconds",
        ):
            command[field] += delta


def monitor(*, contention=False):
    process_count = 2 if contention else 1
    process_output = "".join(
        f"{700 + index}, 1000\n" for index in range(process_count)
    )
    samples = []
    for sample_index in range(10):
        outputs = iter((
            f"0, {GPU_UUID}, {GPU_NAME}\n",
            process_output,
        ))
        snapshot = query_gpu_compute_processes(
            0,
            command_fn=lambda _command, output=outputs: next(output),
            binary_metadata_fn=trusted_binary,
        )
        snapshot["boot_id"] = BOOT_ID
        shift_snapshot_times(snapshot, sample_index * 5.0)
        samples.append(snapshot)
    processes = samples[0]["processes"]
    sequence_errors = gpu_compute_snapshot_sequence_errors(samples, 6.0)
    return {
        "schema_version": GPU_PROCESS_MONITOR_SCHEMA_VERSION,
        "device_index": 0,
        "device_uuid": GPU_UUID,
        "device_name": GPU_NAME,
        "identity_consistent": True,
        "boot_id": BOOT_ID,
        "boot_id_consistent": True,
        "nvidia_smi_binary": trusted_binary(),
        "nvidia_smi_binary_fixed_valid": True,
        "nvidia_smi_binary_consistent": True,
        "nvidia_smi_binary_validation_errors": [],
        "sample_validation_errors": [],
        "snapshot_validation_errors": [],
        "sample_sequence_validation_errors": sequence_errors,
        "observation_span_seconds": (
            gpu_compute_snapshot_observation_span_seconds(samples)
        ),
        "interval_seconds": 5.0,
        "cadence_tolerance_seconds": GPU_QUERY_CADENCE_TOLERANCE_SECONDS,
        "maximum_sample_gap_seconds": (
            gpu_compute_snapshot_maximum_gap_seconds(samples)
        ),
        "sample_count": len(samples),
        "available_sample_count": len(samples),
        "unavailable_sample_count": 0,
        "min_process_count": process_count,
        "max_process_count": process_count,
        "distinct_host_pids": [item["host_pid"] for item in processes],
        "single_distinct_host_pid": not contention,
        "exact_singleton_process_per_sample": not contention,
        "contention_detected": contention,
        "no_process_detected": False,
        "valid_for_benchmark": not contention and not sequence_errors,
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

        idle_output = iter((f"0, {GPU_UUID}, {GPU_NAME}\n", ""))
        idle_snapshot = query_gpu_compute_processes(
            0,
            command_fn=lambda _command: next(idle_output),
            binary_metadata_fn=trusted_binary,
        )
        idle_snapshot["boot_id"] = BOOT_ID
        pre_run_gpu = build_pre_run_gpu_report(
            idle_snapshot,
            cuda_visible_devices="0",
        )
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
            "gpu_process_monitor_interval_seconds": 5.0,
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

    def make_radial_csv_fixture(self):
        run_dir = self.factory.make("radial_conservative")
        checkpoint_path = run_dir / "checkpoint_manifest.json"
        checkpoint_manifest = json.loads(checkpoint_path.read_text())
        for index, relative_path in enumerate(
            EVAL.REQUIRED_PREFLIGHT_CHECKPOINTS,
            start=10,
        ):
            checkpoint_manifest["files"][relative_path] = {
                "bytes": index,
                "sha256": f"{index:064x}",
            }
        write_json(checkpoint_path, checkpoint_manifest)
        copied_artifacts = {
            "source_module": run_dir / "radial-attention-source.py",
            "derived_module": run_dir / "radial-attention-derived.py",
            "optional_imports_patch": (
                run_dir / "radial-attention-optional-imports.patch"
            ),
        }
        for field, path in copied_artifacts.items():
            path.write_bytes(f"fixture-{field}\n".encode())
        flashinfer_manifest_path = run_dir / "radial-flashinfer-manifest.json"
        write_json(flashinfer_manifest_path, {"fixture": "manifest"})
        receipt = {
            field: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for field, path in copied_artifacts.items()
        }
        receipt["commit"] = EVAL.RADIAL_COMMIT
        receipt["flashinfer_version"] = EVAL.FLASHINFER_VERSION
        receipt["runtime_loaded_dependencies"] = {
            "fixture_alias": [{"path": "/fixture/libfixture.so"}],
        }
        receipt["flashinfer_manifest"] = {
            "path": str(flashinfer_manifest_path),
            "bytes": flashinfer_manifest_path.stat().st_size,
            "sha256": sha256(flashinfer_manifest_path),
        }
        write_json(run_dir / "radialattn-install.json", receipt)
        runtime_dependencies = EVAL._expected_radial_runtime_dependencies(
            receipt
        )
        binding = {
            "pmon_observation_mode": (
                "pmon_reported_all_idle_during_audited_window"
            ),
            "binding_method": (
                "sampled_temporal_association_after_idle_guard"
            ),
            "claim_scope": (
                "sampled_temporal_association_not_pid_ownership_or_"
                "continuous_exclusivity"
            ),
            "host_pid_ownership": (
                "unknown_sampled_temporal_association_only"
            ),
            "mps": {
                "mps_status": "unknown",
                "pmon": {"status": "degraded"},
            },
        }
        preflight_path = run_dir / "preflight.json"
        write_json(
            preflight_path,
            {
                "errors": [],
                "attention_method": "radial",
                "python_executable": "/cache/liluchen/FastA2V/envs/ovi/bin/python",
                "cuda_available": True,
                "gpu": GPU_NAME,
                "compute_capability": [8, 0],
                "ffmpeg": "/usr/bin/ffmpeg",
                "ffprobe": "/usr/bin/ffprobe",
                "packages": {
                    "torch": "2.6.0",
                    "torchvision": "0.21.0",
                    "torchaudio": "2.6.0",
                    "flash-attn": "2.7.4.post1",
                    "transformers": "4.49.0",
                    "diffusers": "0.32.2",
                    "omegaconf": "2.3.0",
                },
                "checkpoints": {
                    relative_path: {
                        "exists": True,
                        "bytes": checkpoint_manifest["files"][relative_path][
                            "bytes"
                        ],
                    }
                    for relative_path in EVAL.REQUIRED_PREFLIGHT_CHECKPOINTS
                },
                "checkpoint_manifest": (
                    "/cache/liluchen/FastA2V/checkpoint_manifest.json"
                ),
                "flash_attn_microtest": {
                    "status": "ok",
                    "device": GPU_NAME,
                    "compute_capability": [8, 0],
                    "torch": "2.6.0+cu124",
                    "torch_cuda": "12.4",
                    "torch_cxx11_abi": False,
                    "dtype": "torch.bfloat16",
                    "shape": [1, 128, 24, 128],
                    "max_abs_difference": 0.0078125,
                },
                "radialattn": {
                    "pinned_commit": EVAL.RADIAL_COMMIT,
                    "mask_api": EVAL.RADIAL_MASK_API,
                    "install_receipt_contents": receipt,
                    "source_files_verified": True,
                    "flashinfer_files_verified": True,
                    "flashinfer_manifest_verified": True,
                    "runtime_loader_environment_verified": True,
                    "runtime_dependencies_before_optional_imports": (
                        runtime_dependencies
                    ),
                    "optional_import_loader_evidence": {
                        "status": "ok",
                        "restored": True,
                        "removed_prepend_paths": [
                            EVAL.RADIAL_OPTIONAL_IMPORT_LIB64
                        ],
                        "runtime_dependencies": runtime_dependencies,
                    },
                    "cpu_mask_audits_verified": True,
                    "flashinfer_version": EVAL.FLASHINFER_VERSION,
                    "flashinfer_apis": {
                        "BlockSparseAttentionWrapper": True,
                        "single_prefill_with_kv_cache": True,
                        "merge_state": True,
                    },
                    "derived_mask_api_callable": True,
                    "install_cuda_kernel_launched": False,
                    "preflight_cuda_microtest_required": True,
                },
                "radialattn_microtest": {
                    "gpu_process_binding": binding,
                    "runtime_dependencies_before_cuda": runtime_dependencies,
                    "runtime_dependencies_after_cuda": runtime_dependencies,
                },
            },
        )
        environment_path = run_dir / "environment.json"
        environment = json.loads(environment_path.read_text())
        environment.update(
            {
                "radial_decay_factor": 4.0,
                "radial_block_size": 128,
                "radial_model_type": "wan",
                "radial_loader_bootstrap": {
                    "status": "ok",
                    "receipt_path": EVAL.RADIAL_INSTALL_RECEIPT_PATH,
                    "before_optional_imports": runtime_dependencies,
                    "after_optional_imports": {
                        "status": "ok",
                        "restored": True,
                        "removed_prepend_paths": [
                            EVAL.RADIAL_OPTIONAL_IMPORT_LIB64
                        ],
                        "runtime_dependencies": runtime_dependencies,
                    },
                },
            }
        )
        evidence_paths = (
            preflight_path,
            checkpoint_path,
            run_dir / "radialattn-install.json",
            flashinfer_manifest_path,
            *copied_artifacts.values(),
        )
        for evidence_path in evidence_paths:
            environment["evidence_file_sha256"][evidence_path.name] = sha256(
                evidence_path
            )
        write_json(environment_path, environment)

        calls = 2950
        dispatcher = {
            "configured_method": "radial",
            "active_method": "radial",
            "backend_ready": True,
            "calls_total": calls,
            "calls_by_method": {
                "dense": 0,
                "sparge": 0,
                "radial": calls,
                "svg": 0,
            },
            "errors_by_method": {
                "dense": 0,
                "sparge": 0,
                "radial": 0,
                "svg": 0,
            },
            "fallback_allowed": False,
            "fallback_used": False,
            "fallback_count": 0,
            "fallback_reason": None,
            "expected_calls_without_block_cache": calls,
            "expected_calls": calls,
            "calls_match_expected": True,
            "backend_details": {
                "backend": "official_radial_attention_flashinfer",
                "repository": EVAL.RADIAL_REPOSITORY,
                "pinned_commit": EVAL.RADIAL_COMMIT,
                "mask_api": EVAL.RADIAL_MASK_API,
                "profile": "conservative",
                "decay_factor": 4.0,
                "model_type": EVAL.RADIAL_MODEL_TYPE,
                "block_size": EVAL.RADIAL_BLOCK_SIZE,
                "sequence": EVAL.RADIAL_SEQUENCE,
                "prefix_sequence": EVAL.RADIAL_PREFIX_SEQUENCE,
                "tail_sequence": EVAL.RADIAL_TAIL_SEQUENCE,
                "tail_strategy": "dense_lse_merge_no_padding",
                "empty_row_policy": "dense_row",
                "empty_rows": list(EVAL.RADIAL_EMPTY_ROWS),
                "fallback_allowed": False,
                "calls": calls,
                "plan_cache_entries": 1,
                "plan_cache_hits": calls,
                "plan_cache_misses": 0,
                "last_shape": [1, EVAL.RADIAL_SEQUENCE, 24, 128],
                "last_grid": list(EVAL.RADIAL_GRID),
                "last_device": "cuda:0",
                "last_dtype": "torch.bfloat16",
                "last_mask_audit": EVAL.RADIAL_PROFILE_AUDITS[
                    "conservative"
                ],
                "install_receipt": {
                    "path": EVAL.RADIAL_INSTALL_RECEIPT_PATH,
                    "commit": EVAL.RADIAL_COMMIT,
                    "derived_module_sha256": receipt["derived_module"][
                        "sha256"
                    ],
                    "flashinfer_version": EVAL.FLASHINFER_VERSION,
                    "runtime_dependencies": runtime_dependencies,
                },
                "runtime_dependencies_after_first_cuda": runtime_dependencies,
            },
        }
        timings_path = run_dir / "timings.jsonl"
        timings = [json.loads(line) for line in timings_path.read_text().splitlines()]
        for record in timings:
            record["video_self_attention_dispatcher"] = dispatcher
            write_json(Path(record["output_path"]).with_suffix(".metrics.json"), record)
        timings_path.write_text(
            "".join(
                json.dumps(record, sort_keys=True, allow_nan=True) + "\n"
                for record in timings
            ),
            encoding="utf-8",
        )
        method = dict(self.method("radial_conservative"))
        method["implementation_status"] = "ready"
        return method, run_dir, preflight_path

    def validate_radial_fixture(self, method, run_dir):
        with (
            mock.patch.object(
                EVAL, "radial_microtest_evidence_errors", return_value=[]
            ),
            mock.patch.object(
                EVAL, "radial_receipt_evidence_errors", return_value=[]
            ),
            mock.patch.object(
                EVAL, "flashinfer_manifest_evidence_errors", return_value=[]
            ),
        ):
            return EVAL.validate_run(
                method,
                run_dir,
                self.manifest["fixed_protocol"],
            )

    def rebind_environment_hash(self, run_dir, path):
        environment_path = run_dir / "environment.json"
        environment = json.loads(environment_path.read_text())
        environment["evidence_file_sha256"][path.name] = sha256(path)
        write_json(environment_path, environment)

    def rewrite_radial_timings(self, run_dir, mutate):
        timings_path = run_dir / "timings.jsonl"
        records = [
            json.loads(line) for line in timings_path.read_text().splitlines()
        ]
        mutate(records)
        timings_path.write_text(
            "".join(
                json.dumps(record, sort_keys=True, allow_nan=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        for record in records:
            write_json(
                Path(record["output_path"]).with_suffix(".metrics.json"),
                record,
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

    def test_radial_csv_revalidates_hashed_preflight_and_exposes_scope(self):
        method, run_dir, preflight_path = self.make_radial_csv_fixture()
        valid_payload = json.loads(preflight_path.read_text())
        valid_preflight_sha256 = sha256(preflight_path)
        with (
            mock.patch.object(
                EVAL,
                "radial_microtest_evidence_errors",
                return_value=[],
            ) as validator,
            mock.patch.object(
                EVAL,
                "radial_receipt_evidence_errors",
                return_value=[],
            ) as receipt_validator,
            mock.patch.object(
                EVAL,
                "flashinfer_manifest_evidence_errors",
                return_value=[],
            ) as manifest_validator,
        ):
            summary = EVAL.validate_run(
                method,
                run_dir,
                self.manifest["fixed_protocol"],
            )

        payload = dict(valid_payload)
        payload["errors"] = ["dependency failed"]
        write_json(preflight_path, payload)
        environment_path = run_dir / "environment.json"
        environment = json.loads(environment_path.read_text())
        environment["evidence_file_sha256"]["preflight.json"] = sha256(
            preflight_path
        )
        write_json(environment_path, environment)
        with (
            mock.patch.object(
                EVAL, "radial_microtest_evidence_errors", return_value=[]
            ),
            mock.patch.object(
                EVAL, "radial_receipt_evidence_errors", return_value=[]
            ),
            mock.patch.object(
                EVAL, "flashinfer_manifest_evidence_errors", return_value=[]
            ),
            self.assertRaisesRegex(EVAL.EvaluationError, "preflight errors"),
        ):
            EVAL.validate_run(
                method,
                run_dir,
                self.manifest["fixed_protocol"],
            )

        payload = json.loads(json.dumps(valid_payload))
        payload["radialattn"]["source_files_verified"] = 1
        write_json(preflight_path, payload)
        environment = json.loads(environment_path.read_text())
        environment["evidence_file_sha256"]["preflight.json"] = sha256(
            preflight_path
        )
        write_json(environment_path, environment)
        with (
            mock.patch.object(
                EVAL, "radial_microtest_evidence_errors", return_value=[]
            ),
            mock.patch.object(
                EVAL, "radial_receipt_evidence_errors", return_value=[]
            ),
            mock.patch.object(
                EVAL, "flashinfer_manifest_evidence_errors", return_value=[]
            ),
            self.assertRaisesRegex(EVAL.EvaluationError, "source_files_verified"),
        ):
            EVAL.validate_run(
                method,
                run_dir,
                self.manifest["fixed_protocol"],
            )
        self.assertEqual(summary["preflight_sha256"], valid_preflight_sha256)
        self.assertEqual(
            summary["radial_evidence_mode"],
            "pmon_reported_all_idle_during_audited_window",
        )
        self.assertEqual(summary["radial_pmon_status"], "degraded")
        self.assertEqual(summary["radial_mps_status"], "unknown")
        self.assertEqual(
            summary["radial_host_pid_ownership"],
            "unknown_sampled_temporal_association_only",
        )
        validator.assert_called_once()
        receipt_validator.assert_called_once()
        manifest_validator.assert_called_once()
        self.assertEqual(
            receipt_validator.call_args.args[0],
            valid_payload["radialattn"]["install_receipt_contents"],
        )
        self.assertEqual(
            manifest_validator.call_args.args[1],
            valid_payload["radialattn"]["install_receipt_contents"],
        )
        kwargs = validator.call_args.kwargs
        self.assertEqual(kwargs["expected_pre_run_gpu_path"], str(
            (run_dir / "pre_run_gpu.json").resolve()
        ))

        payload = json.loads(preflight_path.read_text())
        payload["radialattn_microtest"]["gpu_process_binding"][
            "pmon_observation_mode"
        ] = "direct_c_observed"
        write_json(preflight_path, payload)
        with self.assertRaisesRegex(EVAL.EvaluationError, "preflight hash"):
            EVAL.validate_run(
                method,
                run_dir,
                self.manifest["fixed_protocol"],
            )

    def test_radial_csv_requires_exact_loader_bootstrap(self):
        mutations = {
            "missing": lambda payload: payload.pop("radial_loader_bootstrap"),
            "bad-status": lambda payload: payload[
                "radial_loader_bootstrap"
            ].__setitem__("status", "failed"),
            "bad-receipt": lambda payload: payload[
                "radial_loader_bootstrap"
            ].__setitem__("receipt_path", "/tmp/untrusted.json"),
            "bad-runtime": lambda payload: payload[
                "radial_loader_bootstrap"
            ]["before_optional_imports"].__setitem__("aliases", 999),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                method, run_dir, _ = self.make_radial_csv_fixture()
                environment_path = run_dir / "environment.json"
                environment = json.loads(environment_path.read_text())
                mutate(environment)
                write_json(environment_path, environment)
                with self.assertRaisesRegex(
                    EVAL.EvaluationError,
                    "loader bootstrap",
                ):
                    self.validate_radial_fixture(method, run_dir)

    def test_radial_csv_requires_each_real_measurement_dispatcher(self):
        mutations = {
            "missing": lambda records: records[0].pop(
                "video_self_attention_dispatcher"
            ),
            "zero-calls": lambda records: records[1][
                "video_self_attention_dispatcher"
            ].__setitem__("calls_total", 0),
            "boolean-backend-calls": lambda records: records[1][
                "video_self_attention_dispatcher"
            ]["backend_details"].__setitem__("calls", True),
            "fallback": lambda records: records[2][
                "video_self_attention_dispatcher"
            ].__setitem__("fallback_used", True),
            "wrong-receipt": lambda records: records[0][
                "video_self_attention_dispatcher"
            ]["backend_details"]["install_receipt"].__setitem__(
                "derived_module_sha256", "0" * 64
            ),
            "wrong-mask": lambda records: records[0][
                "video_self_attention_dispatcher"
            ]["backend_details"]["last_mask_audit"].__setitem__(
                "repaired_true_blocks", 1
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                method, run_dir, _ = self.make_radial_csv_fixture()
                self.rewrite_radial_timings(run_dir, mutate)
                with self.assertRaisesRegex(
                    EVAL.EvaluationError,
                    "dispatcher",
                ):
                    self.validate_radial_fixture(method, run_dir)

    def test_radial_csv_rejects_forged_originals_even_when_environment_rehashed(self):
        cases = (
            ("radial-attention-source.py", b"forged-source\n"),
            ("radial-attention-derived.py", b"forged-derived\n"),
            ("radial-attention-optional-imports.patch", b"forged-patch\n"),
        )
        for filename, replacement in cases:
            with self.subTest(filename=filename):
                method, run_dir, _ = self.make_radial_csv_fixture()
                evidence_path = run_dir / filename
                evidence_path.write_bytes(replacement)
                self.rebind_environment_hash(run_dir, evidence_path)
                with self.assertRaisesRegex(
                    EVAL.EvaluationError,
                    "differs from install receipt",
                ):
                    self.validate_radial_fixture(method, run_dir)

        for filename, field in (
            ("radialattn-install.json", "receipt"),
            ("radial-flashinfer-manifest.json", "manifest"),
        ):
            with self.subTest(filename=filename):
                method, run_dir, _ = self.make_radial_csv_fixture()
                evidence_path = run_dir / filename
                payload = json.loads(evidence_path.read_text())
                payload["forged"] = True
                write_json(evidence_path, payload)
                self.rebind_environment_hash(run_dir, evidence_path)
                expected = (
                    "receipt differs" if field == "receipt" else "manifest differs"
                )
                with self.assertRaisesRegex(EVAL.EvaluationError, expected):
                    self.validate_radial_fixture(method, run_dir)

    def test_radial_csv_cross_binds_fixed_preflight_inventory(self):
        mutations = {
            "compute": lambda payload: payload.__setitem__(
                "compute_capability", [9, 0]
            ),
            "package": lambda payload: payload["packages"].pop("flash-attn"),
            "checkpoint": lambda payload: payload["checkpoints"].pop(
                EVAL.REQUIRED_PREFLIGHT_CHECKPOINTS[0]
            ),
            "checkpoint-size": lambda payload: payload["checkpoints"][
                EVAL.REQUIRED_PREFLIGHT_CHECKPOINTS[1]
            ].__setitem__("bytes", 999),
            "flash-shape": lambda payload: payload["flash_attn_microtest"].__setitem__(
                "shape", [1, 64, 24, 128]
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                method, run_dir, preflight_path = self.make_radial_csv_fixture()
                payload = json.loads(preflight_path.read_text())
                mutate(payload)
                write_json(preflight_path, payload)
                self.rebind_environment_hash(run_dir, preflight_path)
                with self.assertRaisesRegex(
                    EVAL.EvaluationError,
                    "preflight static evidence",
                ):
                    self.validate_radial_fixture(method, run_dir)

    def test_checkpoint_manifest_is_cross_bound_to_preflight_bytes(self):
        method, run_dir, _ = self.make_radial_csv_fixture()
        manifest_path = run_dir / "checkpoint_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][EVAL.REQUIRED_PREFLIGHT_CHECKPOINTS[0]][
            "bytes"
        ] += 1
        write_json(manifest_path, manifest)
        self.rebind_environment_hash(run_dir, manifest_path)
        with self.assertRaisesRegex(
            EVAL.EvaluationError,
            "checkpoint inventory differs",
        ):
            self.validate_radial_fixture(method, run_dir)

    def test_stable_snapshot_rejects_replacement_after_read(self):
        method, run_dir, preflight_path = self.make_radial_csv_fixture()
        original_snapshot = EVAL._stable_file_snapshot
        replaced = False

        def replace_after_snapshot(path, context):
            nonlocal replaced
            snapshot = original_snapshot(path, context)
            if Path(path).name == preflight_path.name and not replaced:
                replacement = preflight_path.with_suffix(".replacement")
                replacement.write_bytes(snapshot.data)
                os.replace(replacement, preflight_path)
                replaced = True
            return snapshot

        with (
            mock.patch.object(
                EVAL,
                "_stable_file_snapshot",
                side_effect=replace_after_snapshot,
            ),
            self.assertRaisesRegex(
                EVAL.EvaluationError,
                "changed after its stable byte snapshot",
            ),
        ):
            self.validate_radial_fixture(method, run_dir)
        self.assertTrue(replaced)

    def test_stable_snapshot_rejects_symlinked_critical_evidence(self):
        method, run_dir, preflight_path = self.make_radial_csv_fixture()
        target = run_dir / "preflight-target.json"
        target.write_bytes(preflight_path.read_bytes())
        preflight_path.unlink()
        preflight_path.symlink_to(target)
        with self.assertRaisesRegex(EVAL.EvaluationError, "must not be a symlink"):
            self.validate_radial_fixture(method, run_dir)

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

    def test_sample_raw_query_receipt_is_reparsed(self):
        run_dir = self.factory.make("dense")
        records = [
            json.loads(line)
            for line in (run_dir / "timings.jsonl").read_text().splitlines()
        ]
        records[0]["gpu_process_monitor"]["samples"][0]["query_receipt"][
            "commands"
        ][1]["raw_stdout"] = "forged\n"
        (run_dir / "timings.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EVAL.EvaluationError, "raw GPU snapshot"):
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
                "missing-query-receipt",
                lambda payload: payload.pop("query_receipt"),
            ),
            (
                "bad-schema",
                lambda payload: payload.__setitem__("schema_version", 0),
            ),
            (
                "legacy-schema",
                lambda payload: payload.__setitem__("schema_version", 1),
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

    def test_legacy_gpu_monitor_schema_is_rejected(self):
        run_dir = self.factory.make("dense")
        timing_path = run_dir / "timings.jsonl"
        records = [
            json.loads(line) for line in timing_path.read_text().splitlines()
        ]
        records[0]["gpu_process_monitor"]["schema_version"] = 1
        timing_path.write_text(
            "".join(
                json.dumps(record, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EVAL.EvaluationError, "schema"):
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
