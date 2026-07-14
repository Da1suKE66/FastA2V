import importlib.util
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from ovi.gpu_process_monitor import (
    GPU_PROCESS_MONITOR_SCHEMA_VERSION,
    GPU_QUERY_CADENCE_TOLERANCE_SECONDS,
    TRUSTED_NVIDIA_SMI_BYTES,
    TRUSTED_NVIDIA_SMI_PATH,
    TRUSTED_NVIDIA_SMI_SHA256,
    build_pre_run_gpu_report,
    gpu_compute_snapshot_observation_span_seconds,
    gpu_compute_snapshot_maximum_gap_seconds,
    gpu_compute_snapshot_sequence_errors,
    query_gpu_compute_processes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "scripts" / "verify_ovi_output.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_ovi_output_gpu_evidence_test", VERIFIER_PATH
)
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
with mock.patch.dict(sys.modules, {"numpy": SimpleNamespace()}):
    SPEC.loader.exec_module(VERIFIER)

GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"
GPU_NAME = "NVIDIA A100-SXM4-80GB"
IDENTITY = (0, GPU_UUID, GPU_NAME)
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


def raw_sample(count=1, uuid=GPU_UUID):
    process_output = "".join(
        f"{700 + index}, 1000\n" for index in range(count)
    )
    outputs = iter((f"0, {uuid}, {GPU_NAME}\n", process_output))
    snapshot = query_gpu_compute_processes(
        0,
        command_fn=lambda _command: next(outputs),
        binary_metadata_fn=trusted_binary,
    )
    snapshot["boot_id"] = BOOT_ID
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
            receipt[f"{prefix}_{clock}_seconds"] += delta
        for command in receipt["commands"]:
            for prefix in ("started_at", "finished_at"):
                command[f"{prefix}_{clock}_seconds"] += delta
    return shifted


def pre_run_report():
    return build_pre_run_gpu_report(raw_sample(0), cuda_visible_devices="0")


def monitor(samples):
    counts = [item["process_count"] for item in samples]
    pids = sorted(
        {
            process["host_pid"]
            for item in samples
            for process in item["processes"]
        }
    )
    singleton = all(count == 1 for count in counts)
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
        "sample_sequence_validation_errors": (
            sequence_errors
        ),
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
        "min_process_count": min(counts),
        "max_process_count": max(counts),
        "distinct_host_pids": pids,
        "single_distinct_host_pid": singleton and len(pids) == 1,
        "exact_singleton_process_per_sample": singleton,
        "contention_detected": any(count > 1 for count in counts),
        "no_process_detected": any(count == 0 for count in counts),
        "valid_for_benchmark": singleton and not sequence_errors,
        "collection_errors": [],
        "samples": samples,
    }


class GpuEvidenceVerifierTests(unittest.TestCase):
    def test_candidate_accepts_only_cross_bound_singleton_samples(self):
        errors = []
        VERIFIER.validate_gpu_monitor(
            monitor([raw_sample(), raw_sample()]),
            IDENTITY,
            trusted_binary(),
            BOOT_ID,
            5.0,
            1e-12,
            True,
            "measurement[0]",
            errors,
        )
        self.assertEqual(errors, [])

    def test_candidate_rejects_legacy_monitor_schema(self):
        evidence = monitor([raw_sample(), raw_sample()])
        evidence["schema_version"] = 1
        errors = []
        VERIFIER.validate_gpu_monitor(
            evidence,
            IDENTITY,
            trusted_binary(),
            BOOT_ID,
            5.0,
            1e-12,
            True,
            "measurement[0]",
            errors,
        )
        self.assertTrue(any("schema" in error for error in errors))

    def test_candidate_rejects_zero_process_sample(self):
        errors = []
        VERIFIER.validate_gpu_monitor(
            monitor([raw_sample(), raw_sample(0)]),
            IDENTITY,
            trusted_binary(),
            BOOT_ID,
            5.0,
            1e-12,
            True,
            "measurement[0]",
            errors,
        )
        self.assertTrue(any("exactly one" in error for error in errors))

    def test_monitor_uuid_drift_is_rejected(self):
        evidence = monitor([raw_sample(), raw_sample(uuid="GPU-different")])
        errors = []
        VERIFIER.validate_gpu_monitor(
            evidence,
            IDENTITY,
            trusted_binary(),
            BOOT_ID,
            5.0,
            1e-12,
            True,
            "warmup[0]",
            errors,
        )
        self.assertTrue(any("does not match pre-run" in error for error in errors))

    def test_monitor_binary_metadata_drift_is_rejected(self):
        evidence = monitor([raw_sample(), raw_sample()])
        evidence["samples"][1]["nvidia_smi_binary"]["inode"] += 1
        errors = []
        VERIFIER.validate_gpu_monitor(
            evidence,
            IDENTITY,
            trusted_binary(),
            BOOT_ID,
            5.0,
            1e-12,
            True,
            "measurement[0]",
            errors,
        )
        self.assertTrue(
            any("exactly match pre-run" in error for error in errors)
        )

    def test_monitor_raw_receipt_tampering_is_rejected(self):
        evidence = monitor([raw_sample(), raw_sample()])
        evidence["samples"][1]["query_receipt"]["commands"][1][
            "raw_stdout"
        ] = "forged\n"
        errors = []
        VERIFIER.validate_gpu_monitor(
            evidence,
            IDENTITY,
            trusted_binary(),
            BOOT_ID,
            5.0,
            1e-12,
            True,
            "measurement[0]",
            errors,
        )
        self.assertTrue(any("raw_stdout" in error for error in errors))

    def test_monitor_boot_drift_is_rejected(self):
        second = raw_sample()
        second["boot_id"] = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        evidence = monitor([raw_sample(), second])
        errors = []
        VERIFIER.validate_gpu_monitor(
            evidence,
            IDENTITY,
            trusted_binary(),
            BOOT_ID,
            5.0,
            1e-12,
            True,
            "measurement[0]",
            errors,
        )
        self.assertTrue(any("boot" in error.lower() for error in errors))

    def test_duplicate_snapshot_receipt_cannot_count_as_two_samples(self):
        snapshot = raw_sample()
        evidence = monitor([snapshot, snapshot])
        errors = []
        VERIFIER.validate_gpu_monitor(
            evidence,
            IDENTITY,
            trusted_binary(),
            BOOT_ID,
            5.0,
            1e-12,
            True,
            "measurement[0]",
            errors,
        )
        self.assertTrue(any("sequence" in error for error in errors))

    def test_environment_is_bound_to_idle_pre_run_identity(self):
        report = pre_run_report()
        environment = {
            "gpu_physical_index": 0,
            "gpu_uuid": GPU_UUID,
            "gpu_name": GPU_NAME,
            "gpu": GPU_NAME,
            "pre_run_gpu_valid": True,
            "cuda_visible_devices": "0",
        }
        errors = []
        identity = VERIFIER.validate_pre_run_gpu(report, environment, errors)
        self.assertEqual(identity, IDENTITY)
        self.assertEqual(errors, [])

    def test_pre_run_untrusted_binary_is_rejected(self):
        report = pre_run_report()
        report["nvidia_smi_binary"]["sha256"] = "0" * 64
        environment = {
            "gpu_physical_index": 0,
            "gpu_uuid": GPU_UUID,
            "gpu_name": GPU_NAME,
            "gpu": GPU_NAME,
            "pre_run_gpu_valid": True,
            "cuda_visible_devices": "0",
        }
        errors = []
        VERIFIER.validate_pre_run_gpu(report, environment, errors)
        self.assertTrue(any("nvidia-smi sha256" in error for error in errors))

    def test_pre_run_requires_canonical_fields_and_mapping(self):
        base = pre_run_report()
        environment = {
            "gpu_physical_index": 0,
            "gpu_uuid": GPU_UUID,
            "gpu_name": GPU_NAME,
            "gpu": GPU_NAME,
            "pre_run_gpu_valid": True,
            "cuda_visible_devices": "0",
        }
        mutations = []
        for field in (
            "checked_at_utc",
            "sampled_at_unix_seconds",
            "sampled_at_monotonic_seconds",
            "query_started_at_unix_seconds",
            "query_finished_at_unix_seconds",
            "query_started_at_monotonic_seconds",
            "query_finished_at_monotonic_seconds",
            "boot_id",
            "run_nonce",
            "query_receipt",
        ):
            report = dict(base)
            del report[field]
            mutations.append((f"missing-{field}", report, environment))
        for field, value in (
            ("schema_version", 0),
            ("schema_version", 1),
            ("check_type", "old"),
        ):
            report = dict(base)
            report[field] = value
            mutations.append((field, report, environment))
        report = dict(base)
        report["cuda_visible_devices"] = "1,0"
        bad_environment = dict(environment)
        bad_environment["cuda_visible_devices"] = "1,0"
        mutations.append(("ambiguous-cvd", report, bad_environment))
        for label, report, current_environment in mutations:
            with self.subTest(label=label):
                errors = []
                VERIFIER.validate_pre_run_gpu(
                    report,
                    current_environment,
                    errors,
                )
                self.assertTrue(errors)

    def test_consumer_recomputes_and_rejects_large_cadence_gap(self):
        samples = [
            raw_sample(),
            shift_snapshot_times(raw_sample(), 100.0),
        ]
        evidence = monitor(samples)
        evidence["sample_sequence_validation_errors"] = []
        evidence["valid_for_benchmark"] = True
        errors = []
        VERIFIER.validate_gpu_monitor(
            evidence,
            IDENTITY,
            trusted_binary(),
            BOOT_ID,
            5.0,
            1e-12,
            True,
            "measurement[0]",
            errors,
        )
        self.assertTrue(
            any("sequence" in error or "cadence" in error for error in errors)
        )

    def test_bool_forged_monitor_integer_fields_are_rejected(self):
        mutations = (
            ("device_index", lambda evidence: evidence["samples"][0].__setitem__("device_index", False)),
            ("process_count", lambda evidence: evidence["samples"][0].__setitem__("process_count", True)),
            ("host_pid", lambda evidence: evidence["samples"][0]["processes"][0].__setitem__("host_pid", True)),
            ("used_memory", lambda evidence: evidence["samples"][0]["processes"][0].__setitem__("used_memory_mib", True)),
            ("min_process_count", lambda evidence: evidence.__setitem__("min_process_count", True)),
            ("max_process_count", lambda evidence: evidence.__setitem__("max_process_count", True)),
            ("unavailable_sample_count", lambda evidence: evidence.__setitem__("unavailable_sample_count", False)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                evidence = monitor([raw_sample(), raw_sample()])
                mutate(evidence)
                errors = []
                VERIFIER.validate_gpu_monitor(
                    evidence,
                    IDENTITY,
                    trusted_binary(),
                    BOOT_ID,
                    5.0,
                    1e-12,
                    True,
                    "measurement[0]",
                    errors,
                )
                self.assertTrue(errors)


class RunnerOrderingTests(unittest.TestCase):
    RUNNERS = (
        "run_ovi_smoke.sh",
        "run_ovi_dense_baseline.sh",
        "run_ovi_cfg_cache_smoke.sh",
        "run_ovi_cfg_cache_baseline.sh",
        "run_ovi_block_cache_smoke.sh",
        "run_ovi_block_cache_baseline.sh",
        "run_ovi_sparge_smoke.sh",
        "run_ovi_sparge_baseline.sh",
        "run_ovi_sparge_topk75_smoke.sh",
        "run_ovi_sparge_topk75_baseline.sh",
        "run_ovi_radial_conservative_smoke.sh",
        "run_ovi_radial_conservative_baseline.sh",
        "run_ovi_radial_aggressive_smoke.sh",
        "run_ovi_radial_aggressive_baseline.sh",
        "run_ovi_sparse_combo_baseline.sh",
    )

    SPARGE_CUDA_BOUND_SCRIPTS = (
        "install_sparge_attn.sh",
        "run_ovi_sparge_smoke.sh",
        "run_ovi_sparge_baseline.sh",
        "run_ovi_sparge_topk75_smoke.sh",
        "run_ovi_sparge_topk75_baseline.sh",
        "run_ovi_sparse_combo_baseline.sh",
    )

    def test_sparge_scripts_bind_physical_zero_uuid_before_any_python(self):
        for filename in self.SPARGE_CUDA_BOUND_SCRIPTS:
            source = (REPO_ROOT / "scripts" / filename).read_text(
                encoding="utf-8"
            )
            with self.subTest(filename=filename):
                query_offset = source.index(
                    "/usr/bin/nvidia-smi --id 0 --query-gpu=uuid"
                )
                case_offset = source.index(
                    'case "${CUDA_VISIBLE_DEVICES:-}" in'
                )
                export_offset = source.index(
                    'export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_ZERO_UUID}"'
                )
                first_python_offset = source.index(
                    '"${FASTA2V_OVI_ENV}/bin/python"'
                )
                self.assertLess(query_offset, case_offset)
                self.assertLess(case_offset, export_offset)
                self.assertLess(export_offset, first_python_offset)
                self.assertIn(
                    '  ""|"0"|"${PHYSICAL_GPU_ZERO_UUID}") ;;',
                    source,
                )
                self.assertIn(
                    "CUDA_VISIBLE_DEVICES does not select physical GPU 0",
                    source,
                )
                self.assertIn(
                    "^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-"
                    "[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
                    "[0-9A-Fa-f]{12}$",
                    source,
                )
                self.assertEqual(
                    source.count(
                        'export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_ZERO_UUID}"'
                    ),
                    1,
                )

        combo_source = (
            REPO_ROOT / "scripts" / "run_ovi_sparse_combo_baseline.sh"
        ).read_text(encoding="utf-8")
        self.assertLess(
            combo_source.index(
                'export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_ZERO_UUID}"'
            ),
            combo_source.index('source "${REPO_ROOT}/scripts/radial_env.sh"'),
        )
        self.assertNotIn(
            'if [[ "${ATTENTION_METHOD}" == "radial" ]]; then',
            combo_source,
        )

    def test_idle_check_precedes_all_cuda_preflight_and_inference(self):
        for filename in self.RUNNERS:
            with self.subTest(filename=filename):
                source = (REPO_ROOT / "scripts" / filename).read_text(
                    encoding="utf-8"
                )
                check_offset = source.index("scripts/check_pre_run_gpu.py")
                preflight_offset = source.index("scripts/preflight_ovi.py")
                inference_offset = source.index("inference.py")
                self.assertLess(check_offset, preflight_offset)
                self.assertLess(check_offset, inference_offset)
                self.assertIn('"${RUN_DIR}/pre_run_gpu.json"', source)

    def test_environment_hashes_pre_run_gpu_evidence(self):
        source = (REPO_ROOT / "inference.py").read_text(encoding="utf-8")
        self.assertIn('"pre_run_gpu.json"', source)
        self.assertIn('"pre_run_gpu_sha256"', source)
        self.assertIn("validate_pre_run_gpu_report", source)

    def test_idle_check_path_does_not_import_torch_or_initialize_cuda(self):
        source = (
            REPO_ROOT / "ovi" / "gpu_process_monitor.py"
        ).read_text(encoding="utf-8")
        cli_source = (
            REPO_ROOT / "scripts" / "check_pre_run_gpu.py"
        ).read_text(encoding="utf-8")
        sparge_evidence_source = (
            REPO_ROOT / "ovi" / "sparge_evidence.py"
        ).read_text(encoding="utf-8")
        for evidence_source in (source, cli_source, sparge_evidence_source):
            self.assertNotIn("import torch", evidence_source)
            self.assertNotIn("torch.cuda", evidence_source)

    def test_sparge_installer_checks_idle_gpu_before_build_and_microtest(self):
        source = (REPO_ROOT / "scripts" / "install_sparge_attn.sh").read_text(
            encoding="utf-8"
        )
        check_offset = source.index("scripts/check_pre_run_gpu.py")
        build_offset = source.index('"${FASTA2V_OVI_ENV}/bin/python" -m pip install')
        microtest_offset = source.index("from scripts.sparge_attn_microtest")
        self.assertLess(check_offset, build_offset)
        self.assertLess(check_offset, microtest_offset)

    def test_sparge_runners_copy_hashed_build_and_install_gpu_evidence(self):
        for filename in (
            "run_ovi_sparge_smoke.sh",
            "run_ovi_sparge_baseline.sh",
            "run_ovi_sparge_topk75_smoke.sh",
            "run_ovi_sparge_topk75_baseline.sh",
        ):
            source = (REPO_ROOT / "scripts" / filename).read_text(encoding="utf-8")
            self.assertIn("spargeattn-build.log", source)
            self.assertIn("spargeattn-install-pre_run_gpu.json", source)


if __name__ == "__main__":
    unittest.main()
