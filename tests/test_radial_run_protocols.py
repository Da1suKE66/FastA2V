import hashlib
import importlib.util
from copy import deepcopy
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


def radial_dispatcher_fixture(calls=2950):
    runtime_dependencies = {
        "status": "ok",
        "aliases": 26,
        "mapped_files": 31,
        "inventory_sha256": "a" * 64,
    }
    return {
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
            "install_receipt": {
                "runtime_dependencies": dict(runtime_dependencies)
            },
            "runtime_dependencies_after_first_cuda": dict(
                runtime_dependencies
            ),
        },
    }


class RadialRunProtocolTests(unittest.TestCase):
    def test_verified_degraded_pmon_claims_are_explicit_in_summary(self):
        microtest = {
            "pmon_observation_mode": (
                "pmon_reported_all_idle_during_audited_window"
            ),
            "mps_status": "unknown",
            "pid_binding_method": (
                "sampled_temporal_association_after_idle_guard"
            ),
            "gpu_process_claim_scope": (
                "sampled_temporal_association_not_pid_ownership_or_"
                "continuous_exclusivity"
            ),
            "host_pid_ownership": (
                "unknown_sampled_temporal_association_only"
            ),
            "gpu_process_binding": {
                "mps": {
                    "mps_status": "unknown",
                    "host_pid_observed_by_pmon": False,
                    "pmon": {
                        "status": "degraded",
                        "collection_status": "ok",
                        "direct_compute_type_observed": False,
                        "continuous_exclusivity_proven": False,
                    },
                }
            },
        }
        claims = VERIFIER.validated_radial_preflight_claims(microtest, [])
        self.assertEqual(
            claims,
            {
                "status": "validated",
                "source": "preflight.json.radialattn_microtest",
                "pmon_status": "degraded",
                "pmon_collection_status": "ok",
                "pmon_observation_mode": (
                    "pmon_reported_all_idle_during_audited_window"
                ),
                "mps_status": "unknown",
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
                "direct_compute_type_observed": False,
                "host_pid_observed_by_pmon": False,
                "continuous_exclusivity_proven": False,
            },
        )
        protocol = {
            "status": "ok",
            "errors": [],
            "benchmark_valid": True,
            "radial_evidence": claims,
        }
        summary = VERIFIER.build_verification_summary(
            [{"errors": []}], protocol
        )
        self.assertEqual(summary["radial_evidence"], claims)
        self.assertEqual(summary["protocol"]["radial_evidence"], claims)
        self.assertTrue(summary["benchmark_valid"])

    def test_unvalidated_radial_preflight_claims_are_not_republished(self):
        microtest = {
            "gpu_process_binding": {
                "mps": {"pmon": {"status": "degraded"}}
            }
        }
        self.assertIsNone(
            VERIFIER.validated_radial_preflight_claims(
                microtest,
                ["canonical evidence validation failed"],
            )
        )

    def test_runtime_loader_evidence_is_bound_to_receipt_and_opencv_path(self):
        receipt = {
            "runtime_loaded_dependencies": {
                "flashinfer_kernels.abi3.so": [
                    {
                        "path": "/fixed/flashinfer_kernels.abi3.so",
                        "bytes": 2,
                        "sha256": "a" * 64,
                    }
                ],
                "libcudart.so.12": [
                    {
                        "path": "/fixed/libcudart.so.12",
                        "bytes": 3,
                        "sha256": "b" * 64,
                    },
                    {
                        "path": "/other/libcudart.so.12.8.90",
                        "bytes": 4,
                        "sha256": "c" * 64,
                    },
                ],
            }
        }
        expected = VERIFIER.expected_radial_runtime_dependency_evidence(receipt)
        self.assertEqual(expected["aliases"], 2)
        self.assertEqual(expected["mapped_files"], 3)
        optional_imports = {
            "status": "ok",
            "restored": True,
            "removed_prepend_paths": [
                "/cache/liluchen/FastA2V/envs/ovi/lib/python3.11/"
                "site-packages/cv2/../../lib64"
            ],
            "runtime_dependencies": dict(expected),
        }
        errors = []
        VERIFIER.validate_radial_optional_import_loader_evidence(
            optional_imports,
            expected,
            "/cache/liluchen/FastA2V/envs/ovi/lib/python3.11/lib64",
            "test",
            errors,
        )
        self.assertEqual(errors, [])

        optional_imports["runtime_dependencies"]["inventory_sha256"] = "0" * 64
        optional_imports["removed_prepend_paths"] = [
            "/cache/liluchen/FastA2V/envs/ovi/lib64"
        ]
        errors = []
        VERIFIER.validate_radial_optional_import_loader_evidence(
            optional_imports,
            expected,
            "/cache/liluchen/FastA2V/envs/ovi/lib/python3.11/lib64",
            "test",
            errors,
        )
        self.assertTrue(any("copied receipt" in error for error in errors))
        self.assertTrue(any("fixed env lib64" in error for error in errors))

    def test_runtime_dependency_counts_reject_json_booleans(self):
        expected = {
            "status": "ok",
            "aliases": 2,
            "mapped_files": 3,
            "inventory_sha256": "a" * 64,
        }
        for field in ("aliases", "mapped_files"):
            with self.subTest(field=field):
                evidence = dict(expected)
                evidence[field] = True
                errors = []
                VERIFIER.validate_radial_runtime_dependency_evidence(
                    evidence,
                    expected,
                    "test",
                    errors,
                )
                self.assertTrue(
                    any(
                        field in error and "JSON integer" in error
                        for error in errors
                    ),
                    errors,
                )

    def test_radial_preflight_static_flags_and_apis_require_json_booleans(self):
        evidence = {
            "pinned_commit": RADIAL_COMMIT,
            "mask_api": "gen_log_mask_shrinked",
            "source_files_verified": True,
            "flashinfer_files_verified": True,
            "flashinfer_manifest_verified": True,
            "runtime_loader_environment_verified": True,
            "cpu_mask_audits_verified": True,
            "flashinfer_version": VERIFIER.FLASHINFER_VERSION,
            "flashinfer_apis": {
                "BlockSparseAttentionWrapper": True,
                "single_prefill_with_kv_cache": True,
                "merge_state": True,
            },
            "derived_mask_api_callable": True,
            "install_cuda_kernel_launched": False,
            "preflight_cuda_microtest_required": True,
        }
        errors = []
        VERIFIER.validate_radial_preflight_static_evidence(evidence, errors)
        self.assertEqual(errors, [])

        boolean_fields = (
            "source_files_verified",
            "flashinfer_files_verified",
            "flashinfer_manifest_verified",
            "runtime_loader_environment_verified",
            "cpu_mask_audits_verified",
            "derived_mask_api_callable",
            "install_cuda_kernel_launched",
            "preflight_cuda_microtest_required",
        )
        for field in boolean_fields:
            with self.subTest(field=field):
                mutation = deepcopy(evidence)
                mutation[field] = 1 if evidence[field] is True else 0
                errors = []
                VERIFIER.validate_radial_preflight_static_evidence(
                    mutation,
                    errors,
                )
                self.assertTrue(any(field in error for error in errors), errors)

        for api in evidence["flashinfer_apis"]:
            with self.subTest(api=api):
                mutation = deepcopy(evidence)
                mutation["flashinfer_apis"][api] = 1
                errors = []
                VERIFIER.validate_radial_preflight_static_evidence(
                    mutation,
                    errors,
                )
                self.assertTrue(
                    any("flashinfer_apis" in error for error in errors),
                    errors,
                )

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
                if "_smoke" in filename:
                    self.assertIn("text_prompt: prompts/ovi_smoke.csv", source)
                    self.assertIn("each_example_n_times: 1", source)
                else:
                    self.assertIn("text_prompt: prompts/ovi_formal8.csv", source)
                    self.assertIn("each_example_n_times: 3", source)

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
        dispatcher = radial_dispatcher_fixture()
        errors = []
        VERIFIER.validate_radial_dispatcher(dispatcher, errors)
        self.assertEqual(errors, [])

        dispatcher["backend_details"]["tail_strategy"] = "padded_sparse"
        dispatcher["calls_by_method"]["dense"] = 1
        errors = []
        VERIFIER.validate_radial_dispatcher(dispatcher, errors)
        self.assertTrue(any("tail_strategy" in error for error in errors))
        self.assertTrue(any("calls_by_method" in error for error in errors))

        dispatcher["backend_details"]["tail_strategy"] = (
            "dense_lse_merge_no_padding"
        )
        dispatcher["calls_by_method"]["dense"] = 0
        dispatcher["backend_details"][
            "runtime_dependencies_after_first_cuda"
        ] = None
        errors = []
        VERIFIER.validate_radial_dispatcher(dispatcher, errors)
        self.assertTrue(any("after first CUDA" in error for error in errors))

    def test_dispatcher_validator_rejects_bool_int_substitution_per_field(self):
        mutations = (
            (
                "fallback_allowed",
                lambda item: item["backend_details"].__setitem__(
                    "fallback_allowed", 0
                ),
            ),
            (
                "calls_total_bool",
                lambda item: item.__setitem__("calls_total", True),
            ),
            (
                "calls_total_zero",
                lambda item: item.__setitem__("calls_total", 0),
            ),
            (
                "backend_calls_bool",
                lambda item: item["backend_details"].__setitem__("calls", True),
            ),
            (
                "backend_calls_zero",
                lambda item: item["backend_details"].__setitem__("calls", 0),
            ),
            *(
                (
                    f"calls_by_method_{method}",
                    lambda item, method=method: item["calls_by_method"].__setitem__(
                        method,
                        True if method == "radial" else False,
                    ),
                )
                for method in ("dense", "sparge", "radial", "svg")
            ),
            (
                "plan_cache_entries",
                lambda item: item["backend_details"].__setitem__(
                    "plan_cache_entries", True
                ),
            ),
            (
                "plan_cache_hits",
                lambda item: item["backend_details"].__setitem__(
                    "plan_cache_hits", True
                ),
            ),
            (
                "plan_cache_misses",
                lambda item: item["backend_details"].__setitem__(
                    "plan_cache_misses", False
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                dispatcher = radial_dispatcher_fixture()
                mutate(dispatcher)
                errors = []
                VERIFIER.validate_radial_dispatcher(dispatcher, errors)
                self.assertTrue(errors, label)

    def test_dispatcher_runtime_receipts_use_strict_json_comparison(self):
        dispatcher = radial_dispatcher_fixture()
        dispatcher["backend_details"][
            "runtime_dependencies_after_first_cuda"
        ]["aliases"] = True
        errors = []
        VERIFIER.validate_radial_dispatcher(dispatcher, errors)
        self.assertTrue(
            any(
                "after first CUDA" in error and "aliases" in error
                for error in errors
            ),
            errors,
        )


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
            'metadata_value["ldd_dependency_libraries"] = dependency_libraries',
            '"ldd_dependencies": ldd_dependencies',
            '"runtime_loaded_dependencies": runtime_loaded_dependencies',
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
        self.assertIn("readlink -f", radial_env)
        self.assertIn(
            'export LD_LIBRARY_PATH="${RADIAL_TORCH_LIB}:${RADIAL_CUDA_LIB}"',
            radial_env,
        )
        self.assertIn('export FASTA2V_ATTENTION_METHOD="radial"', radial_env)

    def test_radial_runtime_bootstrap_precedes_optional_native_imports(self):
        inference_source = (REPO_ROOT / "inference.py").read_text()
        bootstrap = inference_source.index(
            'if os.environ.get("FASTA2V_ATTENTION_METHOD") == "radial":'
        )
        runtime_check = inference_source.index(
            '"before_optional_imports": verify_radial_runtime_loaded_dependencies',
            bootstrap,
        )
        third_party_import = inference_source.index("    import torch\n", bootstrap)
        self.assertLess(bootstrap, runtime_check)
        self.assertLess(runtime_check, third_party_import)
        self.assertIn(
            "restore_radial_loader_after_preloaded_optional_imports",
            inference_source,
        )
        self.assertIn('"radial_loader_bootstrap": (', inference_source)

        preflight_source = (
            REPO_ROOT / "scripts" / "preflight_ovi.py"
        ).read_text()
        preflight_bootstrap = preflight_source.index(
            'if attention_method == "radial":'
        )
        preflight_runtime_check = preflight_source.index(
            "verify_radial_runtime_loaded_dependencies(receipt)",
            preflight_bootstrap,
        )
        generic_torch_import = preflight_source.index(
            "            import torch\n", preflight_bootstrap
        )
        self.assertLess(preflight_runtime_check, generic_torch_import)
        self.assertIn(
            '"runtime_dependencies_before_optional_imports"',
            preflight_source,
        )
        self.assertIn('"optional_import_loader_evidence"', preflight_source)

        verifier_source = (
            REPO_ROOT / "scripts" / "verify_ovi_output.py"
        ).read_text()
        for marker in (
            "expected_radial_runtime_dependency_evidence",
            "validate_radial_optional_import_loader_evidence",
            'f"runtime_dependencies_{phase}"',
            'environment.get("radial_loader_bootstrap")',
        ):
            self.assertIn(marker, verifier_source)

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
