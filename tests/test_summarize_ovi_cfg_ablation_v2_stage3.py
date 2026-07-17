import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/summarize_ovi_cfg_ablation_v2_stage3.py"
SPEC = importlib.util.spec_from_file_location("stage3_summary_test_module", SCRIPT)
TOOL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TOOL)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Stage3SummaryTests(unittest.TestCase):
    def setUp(self):
        with TOOL.DEFAULT_PROMPTS.open(newline="", encoding="utf-8") as handle:
            self.prompts = [row["text_prompt"] for row in csv.DictReader(handle)]

    def make_fixture(self, root):
        root = Path(root)
        configs = {
            "dense": ("dense", 0, 1770),
            "old_12": ("current_6_23_r3", 12, 1422),
            "new_12": ("late_12_29_r3", 12, 1422),
            "old_14": ("current_9_26_r5_anchor", 14, 1364),
            "new_14": ("late_12_29_r5", 14, 1364),
        }
        planned = []
        ordinal = 0
        run_tags = {}
        for seed in (503, 887, 1291):
            for label in ("dense", "old_12", "new_12", "old_14", "new_14"):
                ordinal += 1
                tag = f"fixture-{ordinal:02d}-seed{seed}-{label}"
                run_tags[(seed, label)] = tag
                planned.append(
                    {
                        "ordinal": ordinal,
                        "seed": seed,
                        "label": label,
                        "config_id": configs[label][0],
                        "run_tag": tag,
                    }
                )
        receipt = {
            "record_type": "ovi_cfg_ablation_v2_frozen_stage3_candidates",
            "protocol_id": TOOL.PROTOCOL_ID,
            "status": "frozen",
            "inputs": {
                "heldout_prompt_csv": {"sha256": sha256(TOOL.DEFAULT_PROMPTS)},
                "heldout_prompt_manifest": {
                    "sha256": sha256(TOOL.DEFAULT_PROMPT_MANIFEST)
                },
            },
            "configurations": {
                label: {
                    "config_id": config_id,
                    "cache_hits": hits,
                    "expected_video_self_attention_calls": calls,
                }
                for label, (config_id, hits, calls) in configs.items()
            },
            "planned_runs": planned,
        }
        receipt_path = root / "frozen.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        comparisons = []
        for label in ("old_12", "new_12", "old_14", "new_14"):
            config_id, hits, calls = configs[label]
            is_new = label.startswith("new")
            for seed in (503, 887, 1291):
                dense_dir = root / run_tags[(seed, "dense")]
                candidate_dir = root / run_tags[(seed, label)]
                dense_dir.mkdir(exist_ok=True)
                candidate_dir.mkdir(exist_ok=True)
                pairs = []
                for prompt_index, prompt in enumerate(self.prompts):
                    filename = f"p{prompt_index:03d}_fixture.mp4"
                    dense_path = dense_dir / filename
                    candidate_path = candidate_dir / filename
                    candidate_path.with_suffix(".metrics.json").write_text(
                        json.dumps(
                            {
                                "status": "ok",
                                "record_type": "measurement",
                                "seed": seed,
                                "prompt_index": prompt_index,
                                "cfg_cache_hits": hits,
                                "denoise_seconds": 9.95 if is_new else 10.0,
                                "video_self_attention_dispatcher": {
                                    "calls_total": calls,
                                    "fallback_used": False,
                                    "fallback_count": 0,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    pairs.append(
                        {
                            "split": "heldout",
                            "prompt": prompt,
                            "seed": seed,
                            "candidate_id": config_id,
                            "comparison_id": f"{label}_vs_dense",
                            "dense": {"path": str(dense_path)},
                            "candidate": {"path": str(candidate_path)},
                            "metrics": {
                                "video": {
                                    "psnr_db": 30.5 if is_new else 30.0,
                                    "ssim": 0.91 if is_new else 0.90,
                                    "temporal_frame_difference_rmse": (
                                        0.09 if is_new else 0.10
                                    ),
                                },
                                "audio": {
                                    "aligned_correlation": 0.91 if is_new else 0.90,
                                    "si_sdr_db": 12.0 if is_new else 11.0,
                                    "aligned_rmse": 0.09 if is_new else 0.10,
                                    "log_mel_l1_distance": 0.19 if is_new else 0.20,
                                    "speech_activity_coverage_difference": (
                                        0.01 if is_new else 0.02
                                    ),
                                    "silence_ratio_difference": (
                                        -0.01 if is_new else -0.02
                                    ),
                                },
                            },
                        }
                    )
                comparison = root / f"{label}-seed{seed}.json"
                comparison.write_text(
                    json.dumps(
                        {
                            "record_type": "ovi_cfg_ablation_v2_media_comparison",
                            "pair_count": 8,
                            "pairs": pairs,
                        }
                    ),
                    encoding="utf-8",
                )
                comparisons.append(comparison)
        return receipt_path, comparisons

    @staticmethod
    def fake_compare_module():
        class FakeCompare:
            @staticmethod
            def analyze_rows(rows, *, bootstrap_replicates, bootstrap_seed):
                del bootstrap_seed
                metrics = {}
                for metric in sorted({row["metric"] for row in rows}):
                    selected = [row for row in rows if row["metric"] == metric]
                    values = [row["oriented_difference"] for row in selected]
                    overall = {
                        "pair_count": len(values),
                        "prompt_cluster_count": len(
                            {row["prompt"] for row in selected}
                        ),
                        "mean": statistics.fmean(values),
                        "median": statistics.median(values),
                        "p10": min(values),
                        "worst": min(values),
                        "win_rate": sum(value > 0 for value in values) / len(values),
                        "tie_rate": sum(value == 0 for value in values) / len(values),
                        "mean_cluster_bootstrap_ci95": [min(values), max(values)],
                        "median_cluster_bootstrap_ci95": [min(values), max(values)],
                    }
                    metrics[metric] = {
                        "higher_is_better": selected[0]["higher_is_better"],
                        "reported_quantity": "candidate_minus_comparator",
                        "overall": overall,
                        "categories": {},
                    }
                return {
                    "record_type": "ovi_cfg_ablation_v2_clustered_analysis",
                    "splits": {
                        "development": {"status": "no_records", "metrics": {}},
                        "heldout": {
                            "status": "ok",
                            "prompt_cluster_count": len(
                                {row["prompt"] for row in rows}
                            ),
                            "record_count": len(rows),
                            "metrics": metrics,
                        },
                    },
                    "bootstrap": {
                        "unit": "prompt",
                        "replicates": bootstrap_replicates,
                    },
                    "cross_split_aggregation": "forbidden",
                    "pending_evaluations": {
                        "asr": {"status": "pending"},
                        "syncnet": {"status": "pending"},
                        "human_blind_review": {"status": "pending"},
                    },
                }

        return FakeCompare

    def test_long_rows_cluster_bootstrap_and_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt, comparisons = self.make_fixture(temporary)
            with mock.patch.object(
                TOOL, "_load_compare_module", return_value=self.fake_compare_module()
            ):
                rows, analyses, summary = TOOL.summarize(
                    frozen_receipt=receipt,
                    comparison_paths=comparisons,
                    bootstrap_replicates=50,
                    bootstrap_seed=17,
                )
        self.assertEqual(len(rows["12_hit"]), 3 * 8 * len(TOOL.METRICS))
        self.assertEqual(len(rows["14_hit"]), 3 * 8 * len(TOOL.METRICS))
        for tier in ("12_hit", "14_hit"):
            heldout = analyses[tier]["splits"]["heldout"]
            self.assertEqual(heldout["prompt_cluster_count"], 8)
            primary = heldout["metrics"]["video_ssim"]["overall"]
            self.assertAlmostEqual(primary["median"], 0.01)
            self.assertEqual(primary["win_rate"], 1.0)
            equivalence = summary["tiers"][tier]["workload_equivalence"]
            self.assertTrue(equivalence["exact_workload_counters_passed"])
            self.assertTrue(
                equivalence["paired_median_within_plus_minus_1_percent"]
            )
        self.assertEqual(summary["pending_evaluations"]["asr"]["status"], "pending")
        self.assertEqual(
            summary["final_acceptance"]["status"], "pending_external_evaluations"
        )

    def test_requires_all_twelve_comparison_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt, comparisons = self.make_fixture(temporary)
            with self.assertRaisesRegex(TOOL.SummaryError, "exactly 12"):
                TOOL.summarize(
                    frozen_receipt=receipt,
                    comparison_paths=comparisons[:-1],
                    bootstrap_replicates=10,
                )


if __name__ == "__main__":
    unittest.main()
