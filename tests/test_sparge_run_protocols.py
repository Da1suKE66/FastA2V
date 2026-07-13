import importlib.util
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "scripts" / "verify_ovi_output.py"
VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "verify_sparge_protocols_under_test", VERIFIER_PATH
)
VERIFIER_MODULE = importlib.util.module_from_spec(VERIFIER_SPEC)
with mock.patch.dict(sys.modules, {"numpy": SimpleNamespace()}):
    VERIFIER_SPEC.loader.exec_module(VERIFIER_MODULE)


def _fixed_environment(run_kind, topk, *, smoke):
    return {
        "run_kind": run_kind,
        "model_name": "720x720_5s",
        "mode": "t2v",
        "video_frame_height_width": [720, 720],
        "solver_name": "unipc",
        "shift": 5.0,
        "seed": 103,
        "video_guidance_scale": 4.0,
        "audio_guidance_scale": 3.0,
        "fp8": False,
        "qint8": False,
        "cpu_offload": False,
        "sp_size": 1,
        "slg_layer": 11,
        "prompt_count": 1,
        "each_example_n_times": 1,
        "sparge_topk": topk,
        "sparge_pvthreshd": 50.0,
        "sparge_smooth_k": True,
        "use_cfg_cache": False,
        "use_block_cache": False,
        "sample_steps": 20 if smoke else 50,
        "warmup_runs": 0 if smoke else 1,
        "measurement_runs": 1 if smoke else 3,
        "benchmark_eligible": not smoke,
        "debug_forward": smoke,
    }


class SpargeRunProtocolTests(unittest.TestCase):
    def test_each_run_kind_is_strictly_bound_to_its_topk_and_mode(self):
        cases = (
            ("sparge_baseline", 0.50, False),
            ("sparge_diagnostic_smoke", 0.50, True),
            ("sparge_topk75_baseline", 0.75, False),
            ("sparge_topk75_diagnostic_smoke", 0.75, True),
        )
        for run_kind, topk, smoke in cases:
            with self.subTest(run_kind=run_kind):
                environment = _fixed_environment(run_kind, topk, smoke=smoke)
                errors = []
                VERIFIER_MODULE.validate_sparge_run_protocol(
                    environment, errors
                )
                self.assertEqual(errors, [])

                environment["sparge_topk"] = 0.75 if topk == 0.50 else 0.50
                errors = []
                VERIFIER_MODULE.validate_sparge_run_protocol(
                    environment, errors
                )
                self.assertTrue(
                    any("sparge_topk" in error for error in errors), errors
                )

    def test_protocol_rejects_unknown_run_kind_and_mixed_acceleration(self):
        environment = _fixed_environment(
            "sparge_topk75_baseline", 0.75, smoke=False
        )
        environment["use_cfg_cache"] = True
        errors = []
        VERIFIER_MODULE.validate_sparge_run_protocol(environment, errors)
        self.assertTrue(any("use_cfg_cache" in error for error in errors))

        environment["run_kind"] = "sparge_unreviewed"
        errors = []
        VERIFIER_MODULE.validate_sparge_run_protocol(environment, errors)
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
