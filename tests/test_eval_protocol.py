import ast
import csv
import json
from pathlib import Path
import unittest

from ovi.eval_protocol import (
    AUDITED_RUN_KINDS,
    RUN_KIND_PROTOCOLS,
    materialize_run_protocol,
    prompt_sequence_sha256,
    run_protocol_errors,
    validate_run_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_RUN_KINDS = {
    "dense_baseline",
    "diagnostic_smoke",
    "official_reference",
    "cfg_cache_benchmark",
    "cfg_cache_diagnostic_smoke",
    "block_cache_benchmark",
    "block_cache_diagnostic_smoke",
    "sparge_baseline",
    "sparge_diagnostic_smoke",
    "sparge_topk75_baseline",
    "sparge_topk75_diagnostic_smoke",
    "radial_conservative_baseline",
    "radial_conservative_diagnostic_smoke",
    "radial_aggressive_baseline",
    "radial_aggressive_diagnostic_smoke",
    "sparge_topk50_cfg_benchmark",
    "sparge_topk75_cfg_benchmark",
    "radial_conservative_cfg_benchmark",
    "radial_aggressive_cfg_benchmark",
    "sparge_topk50_block_cache_benchmark",
    "sparge_topk75_block_cache_benchmark",
    "radial_conservative_block_cache_benchmark",
    "radial_aggressive_block_cache_benchmark",
}

RUN_KIND_CONFIGS = {
    "dense_baseline": "ovi_720x720_5s_dense.yaml",
    "diagnostic_smoke": "ovi_720x720_5s_smoke.yaml",
    "official_reference": "ovi_720x720_5s_official_smoke.yaml",
    "cfg_cache_benchmark": "ovi_720x720_5s_cfg_cache.yaml",
    "cfg_cache_diagnostic_smoke": "ovi_720x720_5s_cfg_cache_smoke.yaml",
    "block_cache_benchmark": "ovi_720x720_5s_block_cache.yaml",
    "block_cache_diagnostic_smoke": "ovi_720x720_5s_block_cache_smoke.yaml",
    "sparge_baseline": "ovi_720x720_5s_sparge.yaml",
    "sparge_diagnostic_smoke": "ovi_720x720_5s_sparge_smoke.yaml",
    "sparge_topk75_baseline": "ovi_720x720_5s_sparge_topk75.yaml",
    "sparge_topk75_diagnostic_smoke": (
        "ovi_720x720_5s_sparge_topk75_smoke.yaml"
    ),
    "radial_conservative_baseline": (
        "ovi_720x720_5s_radial_conservative.yaml"
    ),
    "radial_conservative_diagnostic_smoke": (
        "ovi_720x720_5s_radial_conservative_smoke.yaml"
    ),
    "radial_aggressive_baseline": "ovi_720x720_5s_radial_aggressive.yaml",
    "radial_aggressive_diagnostic_smoke": (
        "ovi_720x720_5s_radial_aggressive_smoke.yaml"
    ),
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

FORMAL8_RUN_KINDS = {
    "dense_baseline",
    "cfg_cache_benchmark",
    "block_cache_benchmark",
    "sparge_baseline",
    "sparge_topk75_baseline",
    "radial_conservative_baseline",
    "radial_aggressive_baseline",
    "sparge_topk50_cfg_benchmark",
    "sparge_topk75_cfg_benchmark",
    "radial_conservative_cfg_benchmark",
    "radial_aggressive_cfg_benchmark",
    "sparge_topk50_block_cache_benchmark",
    "sparge_topk75_block_cache_benchmark",
    "radial_conservative_block_cache_benchmark",
    "radial_aggressive_block_cache_benchmark",
}

DERIVED_ENVIRONMENT_FIELDS = {
    "prompt_count",
    "prompts_sha256",
    "expected_warmup_records",
    "expected_measurement_records",
    "cfg_cache_window_inclusive",
    "block_cache_window_inclusive",
}


def _mutated(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, str):
        return value + "__mutated"
    if isinstance(value, list):
        return [*value, 1]
    raise AssertionError(f"test has no mutation for {type(value).__name__}")


def _simple_json_yaml_values(path):
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


class ImmutableEvalProtocolTests(unittest.TestCase):
    def test_registry_contains_every_existing_ovi_run_kind(self):
        self.assertEqual(set(AUDITED_RUN_KINDS), EXPECTED_RUN_KINDS)
        self.assertEqual(set(RUN_KIND_PROTOCOLS), EXPECTED_RUN_KINDS)

    def test_every_exact_protocol_is_accepted(self):
        for run_kind in sorted(EXPECTED_RUN_KINDS):
            with self.subTest(run_kind=run_kind):
                environment = materialize_run_protocol(run_kind)
                self.assertEqual(run_protocol_errors(environment), [])
                errors = []
                validate_run_protocol(environment, errors)
                self.assertEqual(errors, [])

    def test_every_protocol_matches_its_checked_in_run_config(self):
        self.assertEqual(set(RUN_KIND_CONFIGS), EXPECTED_RUN_KINDS)
        for run_kind, filename in RUN_KIND_CONFIGS.items():
            with self.subTest(run_kind=run_kind, filename=filename):
                config = _simple_json_yaml_values(
                    REPO_ROOT / "configs" / filename
                )
                protocol = materialize_run_protocol(run_kind)
                for field, expected in protocol.items():
                    if field in DERIVED_ENVIRONMENT_FIELDS:
                        continue
                    self.assertIn(field, config)
                    self.assertEqual(config[field], expected)

    def test_fixed_prompt_hash_matches_each_checked_in_ordered_prompt_set(self):
        prompt_sets = {}
        for filename, expected_count in (
            ("ovi_smoke.csv", 1),
            ("ovi_dev6.csv", 6),
            ("ovi_formal8.csv", 8),
        ):
            with (REPO_ROOT / "prompts" / filename).open(
                newline="", encoding="utf-8"
            ) as handle:
                prompts = [
                    row["text_prompt"] for row in csv.DictReader(handle)
                ]
            self.assertEqual(len(prompts), expected_count)
            self.assertEqual(len(set(prompts)), expected_count)
            self.assertTrue(all(prompt.strip() for prompt in prompts))
            prompt_sets[filename] = (
                expected_count,
                prompt_sequence_sha256(prompts),
                prompts,
            )

        self.assertEqual(
            prompt_sets["ovi_formal8.csv"][2][:6],
            prompt_sets["ovi_dev6.csv"][2],
        )

        for run_kind, protocol in RUN_KIND_PROTOCOLS.items():
            with self.subTest(run_kind=run_kind):
                filename = (
                    "ovi_formal8.csv"
                    if run_kind in FORMAL8_RUN_KINDS
                    else "ovi_smoke.csv"
                )
                expected_count, expected_hash, _prompts = prompt_sets[filename]
                self.assertEqual(protocol["prompt_count"], expected_count)
                self.assertEqual(protocol["prompts_sha256"], expected_hash)

    def test_formal_configs_use_formal8_and_smoke_configs_stay_single_prompt(self):
        self.assertEqual(
            FORMAL8_RUN_KINDS,
            {
                run_kind
                for run_kind, protocol in RUN_KIND_PROTOCOLS.items()
                if protocol["sample_steps"] == 50
            },
        )
        for run_kind, filename in RUN_KIND_CONFIGS.items():
            if run_kind == "official_reference":
                continue
            expected_prompt_file = (
                "ovi_formal8.csv"
                if run_kind in FORMAL8_RUN_KINDS
                else "ovi_smoke.csv"
            )
            source = (REPO_ROOT / "configs" / filename).read_text(
                encoding="utf-8"
            )
            with self.subTest(run_kind=run_kind, filename=filename):
                self.assertIn(
                    f"text_prompt: prompts/{expected_prompt_file}", source
                )

    def test_expected_measurement_records_cover_prompts_samples_and_repeats(self):
        for run_kind, protocol in RUN_KIND_PROTOCOLS.items():
            with self.subTest(run_kind=run_kind):
                self.assertEqual(
                    protocol["expected_measurement_records"],
                    protocol["measurement_runs"]
                    * protocol["prompt_count"]
                    * protocol["each_example_n_times"],
                )
                expected_count = 8 if run_kind in FORMAL8_RUN_KINDS else 1
                expected_samples = 3 if run_kind in FORMAL8_RUN_KINDS else 1
                expected_records = 72 if run_kind in FORMAL8_RUN_KINDS else 1
                self.assertEqual(protocol["prompt_count"], expected_count)
                self.assertEqual(
                    protocol["each_example_n_times"], expected_samples
                )
                self.assertEqual(
                    protocol["expected_measurement_records"], expected_records
                )

    def test_evaluation_matrix_uses_formal8_and_a_to_f_method_order(self):
        matrix = json.loads(
            (REPO_ROOT / "configs" / "ovi_eval_matrix.json").read_text(
                encoding="utf-8"
            )
        )
        fixed = matrix["fixed_protocol"]
        formal = RUN_KIND_PROTOCOLS["dense_baseline"]
        self.assertEqual(
            matrix["matrix_id"],
            "ovi_720x720_5s_a100_bf16_formal8x3_v2",
        )
        self.assertEqual(fixed["prompt_count"], formal["prompt_count"])
        self.assertEqual(fixed["prompts_sha256"], formal["prompts_sha256"])
        self.assertEqual(
            fixed["each_example_n_times"], formal["each_example_n_times"]
        )
        formal_methods = matrix["methods"][:6]
        self.assertEqual(
            [method["method_id"] for method in formal_methods],
            [
                "dense",
                "dense_cfg_cache",
                "sparge_topk75",
                "sparge_topk50",
                "radial_conservative",
                "radial_aggressive",
            ],
        )
        self.assertEqual(
            [method["formal_slot"] for method in formal_methods],
            list("ABCDEF"),
        )
        methods = {method["method_id"]: method for method in matrix["methods"]}
        for method_id in ("radial_conservative", "radial_aggressive"):
            self.assertEqual(methods[method_id]["implementation_status"], "ready")
            self.assertNotIn("pending_reason", methods[method_id])
        selected_cfg = methods["best_sparse_cfg"]
        self.assertEqual(selected_cfg["implementation_status"], "ready")
        self.assertIs(selected_cfg["selection_required"], True)
        self.assertEqual(
            selected_cfg["allowed_run_kinds"],
            [
                "sparge_topk50_cfg_benchmark",
                "sparge_topk75_cfg_benchmark",
                "radial_conservative_cfg_benchmark",
                "radial_aggressive_cfg_benchmark",
            ],
        )
        self.assertNotIn("run_kind", selected_cfg["expected_environment"])
        selected_block = methods["block_cache"]
        self.assertEqual(
            selected_block["label"], "Best sparse + simple block cache"
        )
        self.assertEqual(selected_block["implementation_status"], "ready")
        self.assertIs(selected_block["selection_required"], True)
        self.assertEqual(
            selected_block["allowed_run_kinds"],
            [
                "sparge_topk50_block_cache_benchmark",
                "sparge_topk75_block_cache_benchmark",
                "radial_conservative_block_cache_benchmark",
                "radial_aggressive_block_cache_benchmark",
            ],
        )
        self.assertNotIn("run_kind", selected_block["expected_environment"])
        self.assertEqual(
            selected_block["expected_environment"]["block_cache_policy"],
            "fixed",
        )

    def test_every_fixed_field_mutation_is_rejected(self):
        for run_kind in sorted(EXPECTED_RUN_KINDS):
            pristine = materialize_run_protocol(run_kind)
            for field, value in pristine.items():
                with self.subTest(run_kind=run_kind, field=field):
                    environment = dict(pristine)
                    environment[field] = _mutated(value)
                    self.assertTrue(run_protocol_errors(environment))

    def test_every_missing_fixed_field_is_rejected(self):
        for run_kind in sorted(EXPECTED_RUN_KINDS):
            pristine = materialize_run_protocol(run_kind)
            for field in pristine:
                with self.subTest(run_kind=run_kind, field=field):
                    environment = dict(pristine)
                    del environment[field]
                    errors = run_protocol_errors(environment)
                    self.assertTrue(any(field in error for error in errors), errors)

    def test_unknown_run_kind_is_rejected(self):
        environment = materialize_run_protocol("dense_baseline")
        environment["run_kind"] = "unreviewed_experiment"
        errors = run_protocol_errors(environment)
        self.assertEqual(len(errors), 1)
        self.assertIn("not an audited immutable", errors[0])

        environment["run_kind"] = []
        errors = run_protocol_errors(environment)
        self.assertEqual(len(errors), 1)
        self.assertIn("not an audited immutable", errors[0])

    def test_equal_python_values_with_wrong_json_types_are_rejected(self):
        environment = materialize_run_protocol("dense_baseline")
        environment["benchmark_eligible"] = 1
        environment["shift"] = 5
        errors = run_protocol_errors(environment)
        self.assertTrue(any("benchmark_eligible" in error for error in errors))
        self.assertTrue(any("shift" in error for error in errors))

    def test_registry_and_nested_protocol_values_are_immutable(self):
        with self.assertRaises(TypeError):
            RUN_KIND_PROTOCOLS["new"] = {}
        protocol = RUN_KIND_PROTOCOLS["dense_baseline"]
        with self.assertRaises(TypeError):
            protocol["seed"] = 999
        self.assertIsInstance(protocol["video_frame_height_width"], tuple)
        with self.assertRaises(TypeError):
            protocol["video_frame_height_width"][0] = 1

    def test_module_has_no_accelerator_or_config_runtime_imports(self):
        source = (REPO_ROOT / "ovi" / "eval_protocol.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertTrue(
            imported_roots.isdisjoint({"torch", "numpy", "omegaconf"}),
            imported_roots,
        )

    def test_inference_records_every_protocol_field(self):
        source = (REPO_ROOT / "inference.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        collector = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_collect_environment"
        )
        returned_dict = next(
            node.value
            for node in ast.walk(collector)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        )
        recorded_fields = {
            key.value
            for key in returned_dict.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        protocol_fields = {
            field
            for protocol in RUN_KIND_PROTOCOLS.values()
            for field in protocol
        }
        self.assertTrue(protocol_fields <= recorded_fields)

    def test_verifier_calls_general_protocol_validator_unconditionally(self):
        source = (REPO_ROOT / "scripts" / "verify_ovi_output.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        verifier = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "verify_run_protocol"
        )
        direct_calls = [
            statement.value.func.id
            for statement in verifier.body
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
        ]
        self.assertIn("validate_run_protocol", direct_calls)
        self.assertNotIn("validate_sparge_run_protocol", source)
        self.assertNotIn("validate_radial_run_protocol", source)

    def test_official_reference_config_declares_its_protocol_identity(self):
        protocol = RUN_KIND_PROTOCOLS["official_reference"]
        self.assertIs(protocol["benchmark_eligible"], False)
        self.assertEqual(protocol["warmup_runs"], 0)
        self.assertEqual(protocol["measurement_runs"], 1)

        source = (
            REPO_ROOT / "configs" / "ovi_720x720_5s_official_smoke.yaml"
        ).read_text(encoding="utf-8")
        for line in (
            'run_kind: "official_reference"',
            "sample_steps: 20",
            "warmup_runs: 0",
            "measurement_runs: 1",
            'attention_method: "dense"',
            "use_cfg_cache: false",
            "use_block_cache: false",
            "benchmark_eligible: false",
            "debug_forward: false",
        ):
            self.assertIn(line, source)
        self.assertIn(
            "text_prompt: ${oc.env:FASTA2V_PROMPT_FILE}", source
        )
        runner = (
            REPO_ROOT / "scripts" / "run_ovi_official_reference.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'FASTA2V_PROMPT_FILE="${REPO_ROOT}/prompts/ovi_smoke.csv"',
            runner,
        )


if __name__ == "__main__":
    unittest.main()
