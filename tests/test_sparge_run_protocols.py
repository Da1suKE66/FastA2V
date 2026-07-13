from pathlib import Path
import re
import unittest

from ovi.eval_protocol import materialize_run_protocol, validate_run_protocol


REPO_ROOT = Path(__file__).resolve().parents[1]


class SpargeRunProtocolTests(unittest.TestCase):
    def test_each_run_kind_is_strictly_bound_to_its_topk_and_mode(self):
        cases = (
            ("sparge_baseline", 0.50),
            ("sparge_diagnostic_smoke", 0.50),
            ("sparge_topk75_baseline", 0.75),
            ("sparge_topk75_diagnostic_smoke", 0.75),
        )
        for run_kind, topk in cases:
            with self.subTest(run_kind=run_kind):
                environment = materialize_run_protocol(run_kind)
                errors = []
                validate_run_protocol(environment, errors)
                self.assertEqual(errors, [])

                environment["sparge_topk"] = 0.75 if topk == 0.50 else 0.50
                errors = []
                validate_run_protocol(environment, errors)
                self.assertTrue(
                    any("sparge_topk" in error for error in errors), errors
                )

    def test_protocol_rejects_unknown_run_kind_and_mixed_acceleration(self):
        environment = materialize_run_protocol("sparge_topk75_baseline")
        environment["use_cfg_cache"] = True
        errors = []
        validate_run_protocol(environment, errors)
        self.assertTrue(any("use_cfg_cache" in error for error in errors))

        environment["run_kind"] = "sparge_unreviewed"
        errors = []
        validate_run_protocol(environment, errors)
        self.assertTrue(any("not an audited" in error for error in errors))

    def test_topk75_configs_preserve_pure_sparge_settings(self):
        expected = {
            "ovi_720x720_5s_sparge_topk75.yaml": (
                'run_kind: "sparge_topk75_baseline"',
                "sample_steps: 50",
                "warmup_runs: 1",
                "measurement_runs: 3",
                "benchmark_eligible: true",
                "debug_forward: false",
            ),
            "ovi_720x720_5s_sparge_topk75_smoke.yaml": (
                'run_kind: "sparge_topk75_diagnostic_smoke"',
                "sample_steps: 20",
                "warmup_runs: 0",
                "measurement_runs: 1",
                "benchmark_eligible: false",
                "debug_forward: true",
            ),
        }
        common = (
            'attention_method: "sparge"',
            "sparge_topk: 0.75",
            "sparge_pvthreshd: 50",
            "sparge_smooth_k: true",
            "sp_size: 1",
            "use_cfg_cache: false",
            "use_block_cache: false",
        )
        for filename, mode_lines in expected.items():
            with self.subTest(filename=filename):
                source = (REPO_ROOT / "configs" / filename).read_text(
                    encoding="utf-8"
                )
                for line in (*common, *mode_lines):
                    self.assertIn(line, source)

    def test_topk75_runners_cannot_share_topk50_run_parent(self):
        runner_pairs = (
            (
                "run_ovi_sparge_baseline.sh",
                "run_ovi_sparge_topk75_baseline.sh",
                "ovi_720x720_5s_sparge_topk75.yaml",
            ),
            (
                "run_ovi_sparge_smoke.sh",
                "run_ovi_sparge_topk75_smoke.sh",
                "ovi_720x720_5s_sparge_topk75_smoke.yaml",
            ),
        )
        run_parent_pattern = re.compile(r'^RUN_PARENT="([^"]+)"$', re.MULTILINE)
        for topk50_name, topk75_name, config_name in runner_pairs:
            with self.subTest(topk75_name=topk75_name):
                topk50_source = (REPO_ROOT / "scripts" / topk50_name).read_text(
                    encoding="utf-8"
                )
                topk75_source = (REPO_ROOT / "scripts" / topk75_name).read_text(
                    encoding="utf-8"
                )
                topk50_parent = run_parent_pattern.search(topk50_source)
                topk75_parent = run_parent_pattern.search(topk75_source)
                self.assertIsNotNone(topk50_parent)
                self.assertIsNotNone(topk75_parent)
                self.assertNotEqual(
                    topk50_parent.group(1), topk75_parent.group(1)
                )
                self.assertIn("sparge_topk75", topk75_parent.group(1))
                self.assertIn(
                    f"--config-file configs/{config_name}", topk75_source
                )


if __name__ == "__main__":
    unittest.main()
