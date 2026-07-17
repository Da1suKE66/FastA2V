import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/analyze_ovi_cfg_ablation_v2_stage1.py"
SPEC = importlib.util.spec_from_file_location("stage1_analysis", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Stage1PositionAnalysisTests(unittest.TestCase):
    def make_fixture(self, root):
        stage_tag = "stage1-test"
        run_root = root / "runs"
        comparison_dir = root / "comparisons"
        run_root.mkdir()
        comparison_dir.mkdir()
        runs = ANALYSIS.expected_runs(stage_tag, run_root)
        prompts = (("P01", "prompt one"), ("P02", "prompt two"), ("P03", "prompt three"))
        media = {}
        for (seed, config_id), run_dir in runs.items():
            run_dir.mkdir()
            for index, (prompt_id, _prompt) in enumerate(prompts):
                path = run_dir / f"prompt-{index:03d}.mp4"
                path.write_bytes(f"{seed}:{config_id}:{prompt_id}".encode())
                media[(seed, config_id, prompt_id)] = path
            receipt = {
                "status": "passed",
                "cell_id": config_id,
                "seed": seed,
                "run_dir": str(run_dir.resolve()),
                "inputs": {"prompt_csv": {"sha256": "a" * 64}},
                "validation": {
                    "cell": {
                        "config_id": config_id,
                        "stage": "0" if config_id == "dense" else "1",
                    },
                    "git_commit": "b" * 40,
                    "gpu_uuid": "GPU-test",
                    "checkpoint": {"model_sha256": "c" * 64},
                    "record_counts": {"measurements": 3, "warmups": 0},
                    "decoded_streams": {prompt_id: {} for prompt_id, _ in prompts},
                },
            }
            (run_dir / "protocol_validation.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )

        for seed in ANALYSIS.SEEDS:
            for config_id in ANALYSIS.ALL_BINS:
                pairs = []
                for prompt_id, prompt in prompts:
                    early = config_id in ANALYSIS.EARLY_BINS
                    is_single_failure = seed == 211 and prompt_id == "P03"
                    ssim = 0.80 if early else (0.70 if is_single_failure else 0.90)
                    dense = media[(seed, "dense", prompt_id)]
                    candidate = media[(seed, config_id, prompt_id)]
                    pairs.append(
                        {
                            "split": "development",
                            "prompt_id": prompt_id,
                            "prompt": prompt,
                            "seed": seed,
                            "candidate_id": config_id,
                            "dense": {"path": str(dense.resolve()), "sha256": sha256(dense)},
                            "candidate": {
                                "path": str(candidate.resolve()),
                                "sha256": sha256(candidate),
                            },
                            "metrics": {"video": {"ssim": ssim}},
                        }
                    )
                payload = {
                    "record_type": "ovi_cfg_ablation_v2_media_comparison",
                    "pair_count": 3,
                    "pairs": pairs,
                }
                (comparison_dir / ANALYSIS.comparison_filename(seed, config_id)).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
        return stage_tag, run_root, comparison_dir, runs

    def test_exactly_five_of_six_units_supports_position_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_tag, run_root, comparison_dir, _runs = self.make_fixture(root)
            output_dir = root / "output"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = ANALYSIS.main(
                    [
                        "--stage-tag", stage_tag,
                        "--run-root", str(run_root),
                        "--comparison-dir", str(comparison_dir),
                        "--output-dir", str(output_dir),
                        "--reuse-comparisons-only",
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(
                (output_dir / f"{stage_tag}_stage1_position_analysis.json").read_text()
            )
            self.assertEqual(report["unit_count"], 6)
            self.assertEqual(report["late_less_damaging_unit_count"], 5)
            self.assertTrue(report["position_claim_supported"])
            csv_lines = (
                output_dir / f"{stage_tag}_stage1_position_units.csv"
            ).read_text().splitlines()
            self.assertEqual(len(csv_lines), 7)

    def test_any_nonpassed_protocol_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_tag, run_root, comparison_dir, runs = self.make_fixture(root)
            receipt_path = runs[(103, "bin_00_04_r5")] / "protocol_validation.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["status"] = "failed"
            receipt_path.write_text(json.dumps(receipt))
            errors = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(errors):
                status = ANALYSIS.main(
                    [
                        "--stage-tag", stage_tag,
                        "--run-root", str(run_root),
                        "--comparison-dir", str(comparison_dir),
                        "--output-dir", str(root / "output"),
                        "--reuse-comparisons-only",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("status is not passed", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
