import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


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
            True,
            "warmup[0]",
            errors,
        )
        self.assertTrue(any("does not match pre-run" in error for error in errors))

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
