import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from ovi.eval_protocol import materialize_run_protocol, run_protocol_errors
from ovi.radial_evidence import (
    FLASHINFER_WHEEL_SHA256,
    RADIAL_COMMIT,
    RADIAL_DERIVED_MODULE_SHA256,
    RADIAL_OPTIONAL_IMPORTS_PATCH_SHA256,
    RADIAL_SOURCE_MODULE_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "scripts" / "verify_ovi_output.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_radial_protocols_under_test", VERIFIER_PATH
)
VERIFIER = importlib.util.module_from_spec(SPEC)
with mock.patch.dict(sys.modules, {"numpy": SimpleNamespace()}):
    SPEC.loader.exec_module(VERIFIER)


def fixed_environment(profile, *, smoke):
    suffix = "diagnostic_smoke" if smoke else "baseline"
    return materialize_run_protocol(f"radial_{profile}_{suffix}")


class RadialRunProtocolTests(unittest.TestCase):
    def test_all_four_fixed_profiles_and_run_tiers_are_accepted(self):
        for profile in ("conservative", "aggressive"):
            for smoke in (False, True):
                with self.subTest(profile=profile, smoke=smoke):
                    environment = fixed_environment(profile, smoke=smoke)
                    self.assertEqual(run_protocol_errors(environment), [])

    def test_mixed_cache_or_profile_drift_is_rejected(self):
        environment = fixed_environment("conservative", smoke=False)
        environment["use_cfg_cache"] = True
        environment["radial_decay_factor"] = 1.0
        errors = run_protocol_errors(environment)
        self.assertTrue(any("use_cfg_cache" in error for error in errors))
        self.assertTrue(any("radial_decay_factor" in error for error in errors))

        environment["run_kind"] = "radial_unreviewed"
        errors = run_protocol_errors(environment)
        self.assertTrue(any("not an audited immutable" in error for error in errors))

    def test_configs_are_pure_radial_and_bind_profile_decay(self):
        cases = {
            "ovi_720x720_5s_radial_conservative.yaml": (
                'run_kind: "radial_conservative_baseline"',
                'radial_profile: "conservative"',
                "radial_decay_factor: 4.0",
                "sample_steps: 50",
                "warmup_runs: 1",
                "measurement_runs: 3",
                "benchmark_eligible: true",
                "debug_forward: false",
            ),
            "ovi_720x720_5s_radial_conservative_smoke.yaml": (
                'run_kind: "radial_conservative_diagnostic_smoke"',
                'radial_profile: "conservative"',
                "radial_decay_factor: 4.0",
                "sample_steps: 20",
                "warmup_runs: 0",
                "measurement_runs: 1",
                "benchmark_eligible: false",
                "debug_forward: true",
            ),
            "ovi_720x720_5s_radial_aggressive.yaml": (
                'run_kind: "radial_aggressive_baseline"',
                'radial_profile: "aggressive"',
                "radial_decay_factor: 1.0",
                "sample_steps: 50",
                "warmup_runs: 1",
                "measurement_runs: 3",
                "benchmark_eligible: true",
                "debug_forward: false",
            ),
            "ovi_720x720_5s_radial_aggressive_smoke.yaml": (
                'run_kind: "radial_aggressive_diagnostic_smoke"',
                'radial_profile: "aggressive"',
                "radial_decay_factor: 1.0",
                "sample_steps: 20",
                "warmup_runs: 0",
                "measurement_runs: 1",
                "benchmark_eligible: false",
                "debug_forward: true",
            ),
        }
        common = (
            'attention_method: "radial"',
            "radial_block_size: 128",
            'radial_model_type: "wan"',
            "sp_size: 1",
            "use_cfg_cache: false",
            "use_block_cache: false",
        )
        for filename, expected_lines in cases.items():
            with self.subTest(filename=filename):
                source = (REPO_ROOT / "configs" / filename).read_text()
                for line in (*common, *expected_lines):
                    self.assertIn(line, source)

    def test_runners_have_unique_parents_and_copy_all_source_evidence(self):
        names = (
            "run_ovi_radial_conservative_baseline.sh",
            "run_ovi_radial_conservative_smoke.sh",
            "run_ovi_radial_aggressive_baseline.sh",
            "run_ovi_radial_aggressive_smoke.sh",
        )
        parents = set()
        for filename in names:
            source = (REPO_ROOT / "scripts" / filename).read_text()
            parent_line = next(
                line for line in source.splitlines() if line.startswith("RUN_PARENT=")
            )
            parents.add(parent_line)
            for evidence in (
                "radialattn-install.json",
                "radial-flashinfer-manifest.json",
                "radial-attention-source.py",
                "radial-attention-derived.py",
                "radial-attention-optional-imports.patch",
            ):
                self.assertIn(evidence, source)
            self.assertIn("--query-gpu=uuid", source)
            self.assertIn(
                'export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_ZERO_UUID}"',
                source,
            )
            self.assertIn("--attention-method radial", source)
            radial_env_offset = source.index("scripts/radial_env.sh")
            preflight_offset = source.index("scripts/preflight_ovi.py")
            self.assertLess(radial_env_offset, preflight_offset)
        self.assertEqual(len(parents), len(names))

    def test_dispatcher_validator_requires_real_radial_calls_and_tail_audit(self):
        calls = 2950
        dispatcher = {
            "calls_total": calls,
            "calls_by_method": {
                "dense": 0,
                "sparge": 0,
                "radial": calls,
                "svg": 0,
            },
            "backend_details": {
                "backend": "official_radial_attention_flashinfer",
                "repository": (
                    "https://github.com/mit-han-lab/radial-attention.git"
                ),
                "pinned_commit": RADIAL_COMMIT,
                "mask_api": "gen_log_mask_shrinked",
                "profile": "conservative",
                "decay_factor": 4.0,
                "model_type": "wan",
                "block_size": 128,
                "sequence": 15004,
                "prefix_sequence": 14976,
                "tail_sequence": 28,
                "tail_strategy": "dense_lse_merge_no_padding",
                "empty_row_policy": "dense_row",
                "empty_rows": [22, 56, 90],
                "fallback_allowed": False,
                "calls": calls,
                "plan_cache_entries": 1,
                "plan_cache_hits": calls,
                "plan_cache_misses": 0,
                "last_shape": [1, 15004, 24, 128],
                "last_grid": [31, 22, 22],
                "last_device": "cuda:0",
                "last_dtype": "torch.bfloat16",
                "last_mask_audit": dict(
                    VERIFIER.RADIAL_PROFILE_AUDITS["conservative"]
                ),
                "install_receipt": {},
            },
        }
        errors = []
        VERIFIER.validate_radial_dispatcher(dispatcher, errors)
        self.assertEqual(errors, [])

        dispatcher["backend_details"]["tail_strategy"] = "padded_sparse"
        dispatcher["calls_by_method"]["dense"] = 1
        errors = []
        VERIFIER.validate_radial_dispatcher(dispatcher, errors)
        self.assertTrue(any("tail_strategy" in error for error in errors))
        self.assertTrue(any("calls_by_method" in error for error in errors))


class RadialPinAndInstallerTests(unittest.TestCase):
    def test_pin_and_optional_import_patch_hashes_are_exact(self):
        pin = (
            REPO_ROOT / "third_party" / "radial-attention.commit"
        ).read_text().strip()
        patch_path = (
            REPO_ROOT
            / "third_party"
            / "radial-attention-optional-imports.patch"
        )
        self.assertEqual(pin, RADIAL_COMMIT)
        self.assertEqual(
            hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            RADIAL_OPTIONAL_IMPORTS_PATCH_SHA256,
        )
        patch_source = patch_path.read_text()
        changed_files = [
            line[6:]
            for line in patch_source.splitlines()
            if line.startswith("+++ b/")
        ]
        self.assertEqual(changed_files, ["radial_attn/attn_mask.py"])
        self.assertFalse(
            any(path.endswith((".cu", ".cuh")) for path in changed_files)
        )
        self.assertNotIn("triton", patch_source.lower())
        self.assertIn("+def _load_sage_backends():", patch_source)
        self.assertIn(
            '+    plt = importlib.import_module("matplotlib.pyplot")',
            patch_source,
        )
        self.assertIn(
            '+            importlib.import_module("spas_sage_attn")',
            patch_source,
        )

    def test_installer_binds_source_derived_and_flashinfer_candidate(self):
        source = (
            REPO_ROOT / "scripts" / "install_radial_attention.sh"
        ).read_text()
        for value in (
            RADIAL_SOURCE_MODULE_SHA256,
            RADIAL_DERIVED_MODULE_SHA256,
            RADIAL_OPTIONAL_IMPORTS_PATCH_SHA256,
            'UPSTREAM_CLONE_URL="ssh://git@ssh.github.com:443/',
            'GITHUB_SSH_KEY="/home/ma-user/.ssh/id_ed25519_github"',
            'FLASHINFER_VERSION="0.2.5+cu124torch2.6"',
            'FLASHINFER_WHEEL_DIR="${FASTA2V_CACHE_ROOT}/wheels"',
            FLASHINFER_WHEEL_SHA256,
            'FLASHINFER_MANIFEST_PATH="${FASTA2V_CACHE_ROOT}/',
            'metadata_value["ldd_output"] = ldd_output',
            'FIXED_CUDA_HOME="/usr/local/cuda-12.1"',
            'FIXED_LDD_EXECUTABLE="/usr/bin/ldd"',
            "ldd_env = deterministic_ldd_environment(ldd_search_paths)",
            '[str(ldd_executable), str(installed_path)]',
            'source "${REPO_ROOT}/scripts/radial_env.sh"',
            'metadata_value["ldd_dependency_paths"] = dependency_paths',
            '"ldd_dependencies": ldd_dependencies',
            '"runtime_loader_environment": runtime_loader_environment',
            '"flashinfer_manifest": fingerprint(flashinfer_manifest_path)',
            '"cuda_kernel_launched": False',
        ):
            self.assertIn(value, source)
        self.assertNotIn("scripts/check_pre_run_gpu.py", source)
        self.assertNotIn("torch.cuda", source)
        self.assertNotIn(".cuda()", source)
        self.assertNotIn("ldd_env = os.environ.copy()", source)

        radial_env = (REPO_ROOT / "scripts" / "radial_env.sh").read_text()
        self.assertIn("compgen -e", radial_env)
        self.assertIn('== LD_*', radial_env)
        self.assertIn("unset GLIBC_TUNABLES", radial_env)
        self.assertIn(
            'export LD_LIBRARY_PATH="${FASTA2V_OVI_ENV}/lib/python3.11/',
            radial_env,
        )

    def test_backend_has_no_custom_cuda_or_dense_fallback(self):
        source = (
            REPO_ROOT / "ovi" / "modules" / "radial_attention_backend.py"
        ).read_text()
        self.assertIn("BlockSparseAttentionWrapper", source)
        self.assertIn("single_prefill_with_kv_cache", source)
        self.assertIn("merge_state", source)
        self.assertIn("gen_log_mask_shrinked", source)
        self.assertNotIn("scaled_dot_product_attention", source)
        self.assertNotIn("import triton", source)
        self.assertNotIn("fallback_used", source)
        microtest_source = (
            REPO_ROOT / "scripts" / "radial_flashinfer_microtest.py"
        ).read_text()
        self.assertIn("RadialVideoSelfAttentionBackend", microtest_source)
        self.assertIn("torch.isfinite", microtest_source)
        self.assertLess(
            microtest_source.index("verify_radial_runtime_loader_environment(receipt)"),
            microtest_source.index("    import torch"),
        )
        self.assertIn("os.getpid()", microtest_source)
        self.assertIn('gpu_identity.get("process_count") != 1', microtest_source)
        self.assertNotIn("import triton", microtest_source)
        self.assertNotIn("scaled_dot_product_attention", microtest_source)
        verifier_source = (
            REPO_ROOT / "scripts" / "verify_ovi_output.py"
        ).read_text()
        self.assertIn('"runtime_loader_environment_verified": True', verifier_source)


if __name__ == "__main__":
    unittest.main()
