import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from ovi.cfg_ablation_v2_protocol import (
    CANDIDATE_FREEZE_RULE,
    EXPECTED_CONFIG_IDS,
    FROZEN_CONFIG,
    ProtocolError,
    STAGE0_ORDER,
    STAGE3_BALANCED_ORDER,
    STAGE4_FIXED,
    cell_filename,
    filter_cells,
    load_and_validate_matrix,
    protocol_summary,
    validate_frozen_base_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX = REPO_ROOT / "configs/matrix/ovi_cfg_cache_ablation_v2_matrix.csv"
BASE = REPO_ROOT / "configs/ovi_720x720_5s_cfg_cache_late_window_ablation.yaml"
GENERATOR_PATH = REPO_ROOT / "scripts/generate_ovi_cfg_ablation_v2_configs.py"

SPEC = importlib.util.spec_from_file_location("ovi_cfg_v2_generator", GENERATOR_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GENERATOR)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompts(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [row["text_prompt"] for row in csv.DictReader(handle)]


class OviCfgAblationV2ProtocolTests(unittest.TestCase):
    def test_authoritative_bundle_and_development_prompt_hashes_are_bound(self):
        binding_path = (
            REPO_ROOT / "configs/matrix/ovi_cfg_cache_ablation_v2_inputs.json"
        )
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        expected_bundle = {
            "generate_ovi_cfg_ablation_v2_configs.py": "9af786a2bfeb41d855fa45f426d73597c7b91944b4d078f01aa25f5697d28e52",
            "ovi_cfg_cache_ablation_v2.md": "5f989734cfea3b20a43e2944b9b15118207b609f980e7a20f5d6691194db9364",
            "ovi_cfg_cache_ablation_v2_matrix.csv": "b317f02e66e8e356bd80218b24182607a8e1a686878cdb0a32cfa55595843524",
            "ovi_cfg_cache_heldout_prompt_manifest.csv": "a360c2d00109d242deb9ef5f290591a113d178ebc0a4e1b279f8b7269b289018",
            "ovi_cfg_cache_heldout_prompts.csv": "be05f7c411dbe9c1e1a1a2ef8aa6b8f7f6b9a0247f0550198ea66dfb19675631",
        }
        self.assertEqual(
            {
                name: item["sha256"]
                for name, item in binding["authoritative_bundle_files"].items()
            },
            expected_bundle,
        )
        for relative, item in binding["repo_files"].items():
            self.assertEqual(sha256(REPO_ROOT / relative), item["sha256"])

        dev5 = prompts(REPO_ROOT / "prompts/ovi_cfg_ablation_v2_dev5.csv")
        dev3 = prompts(REPO_ROOT / "prompts/ovi_cfg_ablation_v2_dev3.csv")
        stage0 = prompts(REPO_ROOT / "prompts/ovi_cfg_ablation_v2_stage0.csv")
        self.assertEqual(len(dev5), 5)
        self.assertEqual(dev3, [dev5[0], dev5[3], dev5[4]])
        self.assertEqual(stage0, [dev5[0]])

    def test_matrix_has_all_authoritative_cells_and_exact_workloads(self):
        cells = load_and_validate_matrix(MATRIX)
        self.assertEqual(tuple(cell.config_id for cell in cells), EXPECTED_CONFIG_IDS)
        by_id = {cell.config_id: cell for cell in cells}
        expected = {
            "dense": (0, 0, 30, 1770, 0),
            "late_12_29_r1_null": (18, 0, 30, 1770, 0),
            "current_9_26_r5_anchor": (4, 14, 16, 1364, 4),
            "new_12_29_r5_repeat": (4, 14, 16, 1364, 4),
            "late_12_29_r2": (9, 9, 21, 1509, 1),
            "late_12_29_r3": (6, 12, 18, 1422, 2),
            "late_12_29_r4": (5, 13, 17, 1393, 3),
            "late_12_29_r5": (4, 14, 16, 1364, 4),
            "late_15_29_r5": (3, 12, 18, 1422, 4),
            "late_14_29_r8": (2, 14, 16, 1364, 7),
            "late_15_29_r15": (1, 14, 16, 1364, 14),
            "current_6_23_r3": (6, 12, 18, 1422, 2),
        }
        for config_id, values in expected.items():
            cell = by_id[config_id]
            self.assertEqual(
                (
                    cell.refreshes,
                    cell.cache_hits,
                    cell.negative_forwards,
                    cell.expected_video_self_attention_calls,
                    cell.max_cache_age,
                ),
                values,
            )
        for config_id in (
            "bin_00_04_r5",
            "bin_05_09_r5",
            "bin_10_14_r5",
            "bin_15_19_r5",
            "bin_20_24_r5",
            "bin_25_29_r5",
        ):
            cell = by_id[config_id]
            self.assertEqual(
                (
                    cell.eligible_steps,
                    cell.refreshes,
                    cell.cache_hits,
                    cell.negative_forwards,
                    cell.expected_video_self_attention_calls,
                    cell.max_cache_age,
                ),
                (5, 1, 4, 26, 1654, 4),
            )

    def test_matrix_rejects_formula_drift_and_duplicate_config_id(self):
        original = MATRIX.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            formula_path = Path(temporary) / "formula.csv"
            formula_path.write_text(
                original.replace(
                    "0,dense,0,,,,0,0,0,30,1770,0,",
                    "0,dense,0,,,,0,0,1,30,1770,0,",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProtocolError, "workload formula mismatch"):
                load_and_validate_matrix(formula_path)

            duplicate_path = Path(temporary) / "duplicate.csv"
            duplicate_path.write_text(
                original + original.splitlines()[1] + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ProtocolError, "duplicate config_id"):
                load_and_validate_matrix(duplicate_path)

    def test_frozen_constants_and_all_stage_rules_are_machine_readable(self):
        base = GENERATOR.load_flat_yaml(BASE)
        validate_frozen_base_config(base)
        broken = dict(base)
        broken["sample_steps"] = 50
        with self.assertRaisesRegex(ProtocolError, "sample_steps"):
            validate_frozen_base_config(broken)

        summary = protocol_summary()
        self.assertEqual(summary["frozen_config"], FROZEN_CONFIG)
        self.assertEqual(len(STAGE0_ORDER), 6)
        self.assertEqual(
            [item["config_id"] for item in STAGE0_ORDER],
            [
                "dense",
                "late_12_29_r1_null",
                "current_9_26_r5_anchor",
                "new_12_29_r5_repeat",
                "new_12_29_r5_repeat",
                "dense",
            ],
        )
        self.assertEqual(summary["stage1"]["seeds"], [103, 211])
        self.assertEqual(summary["stage1"]["independent_prompt_seed_units"], 6)
        self.assertEqual(CANDIDATE_FREEZE_RULE["required_candidate_count"], 2)
        self.assertEqual(CANDIDATE_FREEZE_RULE["max_stage2_cells_advanced_to_dev5"], 3)
        self.assertEqual(tuple(STAGE3_BALANCED_ORDER), (503, 887, 1291))
        self.assertEqual(STAGE4_FIXED["minimum_warmup_runs"], 3)
        self.assertEqual(STAGE4_FIXED["minimum_measurement_runs"], 5)
        self.assertEqual(summary["stage3"]["forbidden_substitute"], "prompts/ovi_formal8.csv")

    def test_filter_and_filenames_are_explicitly_zero_based_inclusive(self):
        cells = load_and_validate_matrix(MATRIX)
        selected = filter_cells(cells, {"1"}, {"bin_00_04_r5"})
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            cell_filename(selected[0], 211),
            "ovi_cfg_v2_s1_bin_00_04_r5_steps00-04_inclusive_r5_seed211.yaml",
        )
        with self.assertRaisesRegex(ProtocolError, "unknown stages"):
            filter_cells(cells, {"9"}, None)
        with self.assertRaisesRegex(ProtocolError, "unknown or stage-filtered"):
            filter_cells(cells, {"1"}, {"late_12_29_r3"})

    def test_heldout_bundle_is_eight_rows_and_formal8_is_rejected(self):
        heldout = REPO_ROOT / "prompts/ovi_cfg_cache_heldout_prompts.csv"
        manifest = REPO_ROOT / "prompts/ovi_cfg_cache_heldout_prompt_manifest.csv"
        heldout_prompts = GENERATOR.load_prompt_csv(heldout)
        rows = GENERATOR.load_prompt_manifest(manifest, len(heldout_prompts))
        self.assertEqual(len(heldout_prompts), 8)
        self.assertEqual([row["prompt_id"] for row in rows], [f"H{i:02d}" for i in range(1, 9)])
        with self.assertRaisesRegex(ProtocolError, "must not be used"):
            GENERATOR.load_prompt_csv(REPO_ROOT / "prompts/ovi_formal8.csv")

    def test_materializer_filters_stage_config_seed_and_binds_every_input_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "materialized"
            args = [
                "materialize-config",
                "--base-config",
                str(BASE),
                "--prompt-csv",
                str(REPO_ROOT / "prompts/ovi_cfg_ablation_v2_dev3.csv"),
                "--output-dir",
                str(output),
                "--stages",
                "1",
                "--config-ids",
                "bin_00_04_r5",
                "--seeds",
                "103,211",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(GENERATOR.main(args), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(len(manifest["materializations"]), 2)
            self.assertEqual(
                {entry["seed"] for entry in manifest["materializations"]},
                {103, 211},
            )
            self.assertEqual(
                set(manifest["input_files"]),
                {
                    "base_config",
                    "matrix",
                    "prompt_csv",
                    "protocol_doc",
                    "source_binding",
                    "generator",
                    "protocol_module",
                },
            )
            for entry in manifest["materializations"]:
                self.assertEqual(entry["indexing"], "zero_based_inclusive")
                self.assertEqual(entry["refreshes"], 1)
                self.assertEqual(entry["cache_hits"], 4)
                self.assertEqual(entry["negative_forwards"], 26)
                self.assertEqual(entry["expected_video_self_attention_calls"], 1654)
                self.assertEqual(entry["max_cache_age"], 4)
                self.assertIn("steps00-04_inclusive_r5", Path(entry["config_path"]).name)
                self.assertEqual(sha256(entry["config_path"]), entry["config_sha256"])
                config = GENERATOR.load_flat_yaml(Path(entry["config_path"]))
                self.assertEqual(config["sample_steps"], 30)
                self.assertEqual(config["solver_name"], "euler")
                self.assertEqual(config["cfg_cache_start_step"], 0)
                self.assertEqual(config["cfg_cache_end_step"], 4)
                self.assertEqual(config["cfg_cache_refresh_interval"], 5)
            copied = manifest["copied_inputs"]["prompt_csv"]
            self.assertEqual(copied["sha256"], sha256(REPO_ROOT / "prompts/ovi_cfg_ablation_v2_dev3.csv"))

            with self.assertRaises(SystemExit) as reused:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    GENERATOR.main(args)
            self.assertEqual(reused.exception.code, 2)

    def test_stage3_materializes_mixed_source_stages_with_heldout_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "stage3"
            selected = (
                "dense,current_6_23_r3,late_12_29_r3,"
                "current_9_26_r5_anchor,late_12_29_r5"
            )
            args = [
                "materialize-config",
                "--base-config",
                str(BASE),
                "--prompt-csv",
                str(REPO_ROOT / "prompts/ovi_cfg_cache_heldout_prompts.csv"),
                "--prompt-manifest",
                str(REPO_ROOT / "prompts/ovi_cfg_cache_heldout_prompt_manifest.csv"),
                "--output-dir",
                str(output),
                "--execution-stage",
                "3",
                "--config-ids",
                selected,
                "--seeds",
                "503,887,1291",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(GENERATOR.main(args), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["materializations"]), 15)
            self.assertEqual(manifest["filters"]["execution_stage"], "3")
            self.assertEqual(
                {entry["source_matrix_stage"] for entry in manifest["materializations"]},
                {0, 2},
            )
            for entry in manifest["materializations"]:
                self.assertEqual(entry["execution_stage"], 3)
                self.assertEqual(entry["stage"], 3)
                self.assertIn("ovi_cfg_v2_s3_", Path(entry["config_path"]).name)
                config = GENERATOR.load_flat_yaml(Path(entry["config_path"]))
                self.assertEqual(config["cfg_ablation_execution_stage"], 3)
                self.assertEqual(
                    config["cfg_ablation_source_matrix_stage"],
                    entry["source_matrix_stage"],
                )
                self.assertTrue(config["run_kind"].startswith("cfg_cache_ablation_v2_s3_"))
            self.assertEqual(
                manifest["copied_inputs"]["prompt_manifest"]["sha256"],
                sha256(REPO_ROOT / "prompts/ovi_cfg_cache_heldout_prompt_manifest.csv"),
            )

    def test_benchmark_materialization_enforces_three_warmups_and_five_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "benchmark"
            common = [
                "materialize-config",
                "--base-config",
                str(BASE),
                "--prompt-csv",
                str(REPO_ROOT / "prompts/ovi_cfg_ablation_v2_stage0.csv"),
                "--output-dir",
                str(output),
                "--stages",
                "0",
                "--config-ids",
                "dense",
                "--execution-stage",
                "4",
                "--seeds",
                "777",
                "--benchmark-eligible",
            ]
            with self.assertRaises(SystemExit) as invalid:
                with redirect_stderr(io.StringIO()):
                    GENERATOR.main(common + ["--warmup-runs", "2", "--measurement-runs", "5"])
            self.assertEqual(invalid.exception.code, 2)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    GENERATOR.main(
                        common + ["--warmup-runs", "3", "--measurement-runs", "5"]
                    ),
                    0,
                )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["warmup_runs"], 3)
            self.assertEqual(manifest["measurement_runs"], 5)
            self.assertTrue(manifest["benchmark_eligible"])
            self.assertEqual(manifest["materializations"][0]["execution_stage"], 4)
            self.assertEqual(manifest["materializations"][0]["source_matrix_stage"], 0)
            self.assertEqual(manifest["materializations"][0]["seed"], 777)


if __name__ == "__main__":
    unittest.main()
