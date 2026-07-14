import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from ovi.eval_protocol import materialize_run_protocol, run_protocol_errors


REPO_ROOT = Path(__file__).resolve().parents[1]

COMBO_CONFIGS = {
    "sparge_topk50_cfg_benchmark": "ovi_720x720_5s_sparge_topk50_cfg.yaml",
    "sparge_topk75_cfg_benchmark": "ovi_720x720_5s_sparge_topk75_cfg.yaml",
    "radial_conservative_cfg_benchmark": (
        "ovi_720x720_5s_radial_conservative_cfg.yaml"
    ),
    "radial_aggressive_cfg_benchmark": (
        "ovi_720x720_5s_radial_aggressive_cfg.yaml"
    ),
    "sparge_topk50_block_cache_benchmark": (
        "ovi_720x720_5s_sparge_topk50_block_cache.yaml"
    ),
    "sparge_topk75_block_cache_benchmark": (
        "ovi_720x720_5s_sparge_topk75_block_cache.yaml"
    ),
    "radial_conservative_block_cache_benchmark": (
        "ovi_720x720_5s_radial_conservative_block_cache.yaml"
    ),
    "radial_aggressive_block_cache_benchmark": (
        "ovi_720x720_5s_radial_aggressive_block_cache.yaml"
    ),
}

CFG_RUN_KINDS = (
    "sparge_topk50_cfg_benchmark",
    "sparge_topk75_cfg_benchmark",
    "radial_conservative_cfg_benchmark",
    "radial_aggressive_cfg_benchmark",
)
BLOCK_RUN_KINDS = (
    "sparge_topk50_block_cache_benchmark",
    "sparge_topk75_block_cache_benchmark",
    "radial_conservative_block_cache_benchmark",
    "radial_aggressive_block_cache_benchmark",
)

DERIVED_FIELDS = {
    "prompt_count",
    "prompts_sha256",
    "expected_warmup_records",
    "expected_measurement_records",
    "cfg_cache_window_inclusive",
    "block_cache_window_inclusive",
}


def simple_yaml_values(path):
    values = {}
    for source_line in path.read_text(encoding="utf-8").splitlines():
        line = source_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        try:
            values[key.strip()] = json.loads(raw_value.strip())
        except json.JSONDecodeError:
            continue
    return values


class SparseComboProtocolTests(unittest.TestCase):
    def test_all_eight_combinations_are_immutable_formal_protocols(self):
        self.assertEqual(set(COMBO_CONFIGS), set(CFG_RUN_KINDS + BLOCK_RUN_KINDS))
        for run_kind in COMBO_CONFIGS:
            with self.subTest(run_kind=run_kind):
                protocol = materialize_run_protocol(run_kind)
                self.assertEqual(run_protocol_errors(protocol), [])
                self.assertEqual(protocol["sample_steps"], 50)
                self.assertEqual(protocol["prompt_count"], 6)
                self.assertEqual(protocol["warmup_runs"], 1)
                self.assertEqual(protocol["measurement_runs"], 3)
                self.assertEqual(protocol["expected_measurement_records"], 18)
                self.assertIs(protocol["benchmark_eligible"], True)
                self.assertIs(protocol["debug_forward"], False)
                self.assertEqual(
                    protocol["use_cfg_cache"], run_kind in CFG_RUN_KINDS
                )
                self.assertEqual(
                    protocol["use_block_cache"], run_kind in BLOCK_RUN_KINDS
                )
                self.assertEqual(protocol["block_cache_policy"], "fixed")

                drifted = dict(protocol)
                drifted["use_cfg_cache"] = not protocol["use_cfg_cache"]
                self.assertTrue(run_protocol_errors(drifted))

    def test_sparse_profile_identity_is_fixed_per_run_kind(self):
        for run_kind in COMBO_CONFIGS:
            protocol = materialize_run_protocol(run_kind)
            with self.subTest(run_kind=run_kind):
                if run_kind.startswith("sparge_topk50_"):
                    self.assertEqual(protocol["attention_method"], "sparge")
                    self.assertEqual(protocol["sparge_topk"], 0.5)
                elif run_kind.startswith("sparge_topk75_"):
                    self.assertEqual(protocol["attention_method"], "sparge")
                    self.assertEqual(protocol["sparge_topk"], 0.75)
                elif run_kind.startswith("radial_conservative_"):
                    self.assertEqual(protocol["attention_method"], "radial")
                    self.assertEqual(protocol["radial_profile"], "conservative")
                    self.assertEqual(protocol["radial_decay_factor"], 4.0)
                else:
                    self.assertEqual(protocol["attention_method"], "radial")
                    self.assertEqual(protocol["radial_profile"], "aggressive")
                    self.assertEqual(protocol["radial_decay_factor"], 1.0)

    def test_eight_configs_match_their_complete_protocols(self):
        output_dirs = set()
        for run_kind, filename in COMBO_CONFIGS.items():
            path = REPO_ROOT / "configs" / filename
            config = simple_yaml_values(path)
            protocol = materialize_run_protocol(run_kind)
            with self.subTest(run_kind=run_kind, filename=filename):
                for field, expected in protocol.items():
                    if field in DERIVED_FIELDS:
                        continue
                    self.assertIn(field, config)
                    self.assertEqual(config[field], expected)
                self.assertIn("text_prompt: prompts/ovi_dev6.csv", path.read_text())
                output_line = next(
                    line
                    for line in path.read_text().splitlines()
                    if line.startswith("output_dir:")
                )
                self.assertNotIn(output_line, output_dirs)
                output_dirs.add(output_line)
        self.assertEqual(len(output_dirs), 8)

    def test_ordinary_radial_protocols_still_forbid_both_caches(self):
        for run_kind in (
            "radial_conservative_baseline",
            "radial_conservative_diagnostic_smoke",
            "radial_aggressive_baseline",
            "radial_aggressive_diagnostic_smoke",
        ):
            protocol = materialize_run_protocol(run_kind)
            with self.subTest(run_kind=run_kind):
                self.assertIs(protocol["use_cfg_cache"], False)
                self.assertIs(protocol["use_block_cache"], False)
                protocol["use_cfg_cache"] = True
                self.assertTrue(run_protocol_errors(protocol))

    def test_matrix_registers_ready_candidates_with_explicit_selection(self):
        matrix = json.loads(
            (REPO_ROOT / "configs" / "ovi_eval_matrix.json").read_text()
        )
        methods = {method["method_id"]: method for method in matrix["methods"]}
        cfg = methods["best_sparse_cfg"]
        block = methods["block_cache"]
        self.assertEqual(cfg["implementation_status"], "ready")
        self.assertIs(cfg["selection_required"], True)
        self.assertEqual(tuple(cfg["allowed_run_kinds"]), CFG_RUN_KINDS)
        self.assertNotIn("run_kind", cfg["expected_environment"])
        self.assertEqual(block["implementation_status"], "ready")
        self.assertIs(block["selection_required"], True)
        self.assertEqual(tuple(block["allowed_run_kinds"]), BLOCK_RUN_KINDS)
        self.assertNotIn("run_kind", block["expected_environment"])
        self.assertEqual(block["expected_environment"]["block_cache_policy"], "fixed")


class SparseComboRunnerTests(unittest.TestCase):
    def test_generic_runner_rejects_missing_profile_before_creating_cache(self):
        path = REPO_ROOT / "scripts" / "run_ovi_sparse_combo_baseline.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache-must-not-be-created"
            env = {
                **os.environ,
                "FASTA2V_CACHE_ROOT": str(cache_root),
                "FASTA2V_SPARSE_COMBO": "cfg",
            }
            env.pop("FASTA2V_SPARSE_PROFILE", None)
            result = subprocess.run(
                ["bash", str(path)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must be set explicitly", result.stderr)
            self.assertFalse(cache_root.exists())

    def test_generic_runner_is_fresh_fail_closed_and_copies_all_evidence(self):
        path = REPO_ROOT / "scripts" / "run_ovi_sparse_combo_baseline.sh"
        source = path.read_text(encoding="utf-8")
        self.assertTrue(os.access(path, os.X_OK))
        self.assertIn('SPARSE_PROFILE="${FASTA2V_SPARSE_PROFILE:-}"', source)
        self.assertIn('SPARSE_COMBO="${FASTA2V_SPARSE_COMBO:-}"', source)
        self.assertIn(
            'RUN_PARENT="${FASTA2V_CACHE_ROOT}/runs/ovi_720ckpt_${SPARSE_PROFILE}_${SPARSE_COMBO}_50step"',
            source,
        )
        self.assertIn('if ! mkdir "${RUN_DIR}"; then', source)
        for filename in COMBO_CONFIGS.values():
            self.assertIn(f'CONFIG_FILE="configs/{filename}"', source)
        for evidence in (
            "ovi-environment.freeze.txt",
            "checkpoint_manifest.json",
            "spargeattn-install.json",
            "spargeattn-build.log",
            "spargeattn-install-pre_run_gpu.json",
            "radialattn-install.json",
            "radial-flashinfer-manifest.json",
            "radial-attention-source.py",
            "radial-attention-derived.py",
            "radial-attention-optional-imports.patch",
            "pre_run_gpu.json",
            "preflight.json",
        ):
            self.assertIn(evidence, source)
        self.assertIn('--attention-method "${ATTENTION_METHOD}"', source)
        self.assertIn('scripts/verify_ovi_output.py "${RUN_DIR}"', source)
        subprocess.run(["bash", "-n", str(path)], check=True)

    def test_user_runners_fix_combo_and_reject_external_override(self):
        cases = {
            "run_ovi_best_sparse_cfg_baseline.sh": "cfg",
            "run_ovi_best_sparse_block_cache_baseline.sh": "block_cache",
        }
        for filename, combo in cases.items():
            path = REPO_ROOT / "scripts" / filename
            source = path.read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertTrue(os.access(path, os.X_OK))
                self.assertIn('if [[ "${FASTA2V_SPARSE_COMBO+x}" == "x" ]]', source)
                self.assertIn(f'export FASTA2V_SPARSE_COMBO="{combo}"', source)
                self.assertIn("run_ovi_sparse_combo_baseline.sh", source)
                subprocess.run(["bash", "-n", str(path)], check=True)
                result = subprocess.run(
                    ["bash", str(path)],
                    cwd=REPO_ROOT,
                    env={
                        **os.environ,
                        "FASTA2V_SPARSE_PROFILE": "sparge_topk50",
                        "FASTA2V_SPARSE_COMBO": "external_override",
                    },
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("external override is forbidden", result.stderr)


if __name__ == "__main__":
    unittest.main()
