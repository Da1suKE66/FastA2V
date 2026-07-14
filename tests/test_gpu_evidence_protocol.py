import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from ovi.gpu_process_monitor import (
    TRUSTED_NVIDIA_SMI_BYTES,
    TRUSTED_NVIDIA_SMI_PATH,
    TRUSTED_NVIDIA_SMI_SHA256,
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
    return {
        "available": True,
        "error": None,
        "device_index": 0,
        "device_uuid": uuid,
        "device_name": GPU_NAME,
        "process_count": count,
        "processes": [
            {"host_pid": 700 + index, "used_memory_mib": 1000}
            for index in range(count)
        ],
        "sampled_at_unix_seconds": 0.0,
        "nvidia_smi_binary": trusted_binary(),
    }


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
        "valid_for_benchmark": singleton,
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
            True,
            "measurement[0]",
            errors,
        )
        self.assertEqual(errors, [])

    def test_candidate_rejects_zero_process_sample(self):
        errors = []
        VERIFIER.validate_gpu_monitor(
            monitor([raw_sample(), raw_sample(0)]),
            IDENTITY,
            trusted_binary(),
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
            True,
            "measurement[0]",
            errors,
        )
        self.assertTrue(
            any("exactly match pre-run" in error for error in errors)
        )

    def test_environment_is_bound_to_idle_pre_run_identity(self):
        report = {
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
            "cuda_visible_devices": "0",
            "nvidia_smi_binary": trusted_binary(),
            "checked_at_utc": "2026-07-14T00:00:00+00:00",
            "sampled_at_unix_seconds": 1.0,
            "sampled_at_monotonic_seconds": 1.0,
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "run_nonce": "1" * 32,
        }
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
        report = {
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
            "cuda_visible_devices": "0",
            "nvidia_smi_binary": trusted_binary(),
            "checked_at_utc": "2026-07-14T00:00:00+00:00",
            "sampled_at_unix_seconds": 1.0,
            "sampled_at_monotonic_seconds": 1.0,
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "run_nonce": "1" * 32,
        }
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
        base = {
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
            "cuda_visible_devices": "0",
            "nvidia_smi_binary": trusted_binary(),
            "checked_at_utc": "2026-07-14T00:00:00+00:00",
            "sampled_at_unix_seconds": 1.0,
            "sampled_at_monotonic_seconds": 1.0,
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "run_nonce": "1" * 32,
        }
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
            "boot_id",
            "run_nonce",
        ):
            report = dict(base)
            del report[field]
            mutations.append((f"missing-{field}", report, environment))
        for field, value in (("schema_version", 0), ("check_type", "old")):
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
