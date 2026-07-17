import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "compare_ovi_cfg_ablation_v2.py"
SPEC = importlib.util.spec_from_file_location("compare_ovi_cfg_ablation_v2", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


class AudioMetricTests(unittest.TestCase):
    def test_alignment_recovers_positive_candidate_delay(self):
        rng = np.random.default_rng(17)
        reference = rng.normal(size=4096)
        delay = 13
        candidate = np.concatenate([np.zeros(delay), reference[:-delay]])

        left, right, selected, correlation = TOOL.align_audio(
            reference,
            candidate,
            max_lag_samples=20,
        )

        self.assertEqual(selected, delay)
        self.assertGreater(correlation, 0.999999)
        np.testing.assert_allclose(left, right)

    def test_identical_audio_has_zero_rmse_and_log_mel_distance(self):
        time = np.arange(3200, dtype=np.float64) / TOOL.SAMPLE_RATE
        audio = 0.2 * np.sin(2.0 * np.pi * 440.0 * time)

        metrics = TOOL.aligned_audio_metrics(
            audio,
            audio.copy(),
            max_lag_samples=8,
            activity_threshold_dbfs=-45.0,
        )

        self.assertEqual(metrics["selected_lag_samples"], 0)
        self.assertAlmostEqual(metrics["aligned_correlation"], 1.0, places=12)
        self.assertAlmostEqual(metrics["aligned_rmse"], 0.0, places=12)
        self.assertAlmostEqual(metrics["log_mel_l1_distance"], 0.0, places=12)
        self.assertGreater(metrics["si_sdr_db"], 100.0)
        self.assertEqual(
            metrics["dense_activity"]["speech_activity_coverage"],
            metrics["candidate_activity"]["speech_activity_coverage"],
        )


class VideoAndLatentMetricTests(unittest.TestCase):
    def test_temporal_frame_difference_error(self):
        reference = np.asarray([[[0]], [[10]], [[20]]], dtype=np.uint8)
        identical = reference.copy()
        changed = np.asarray([[[0]], [[20]], [[20]]], dtype=np.uint8)

        self.assertEqual(TOOL.temporal_frame_difference_error(reference, identical), 0.0)
        self.assertGreater(TOOL.temporal_frame_difference_error(reference, changed), 0.0)

    def test_latent_relative_l2_and_cosine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense = root / "dense.npy"
            candidate = root / "candidate.npy"
            np.save(dense, np.asarray([[1.0, 0.0], [0.0, 1.0]]))
            np.save(candidate, np.asarray([[1.0, 0.0], [0.0, 1.0]]))

            result = TOOL.latent_similarity(dense, candidate)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["relative_l2"], 0.0)
        self.assertAlmostEqual(result["cosine_similarity"], 1.0)


class PairResolutionAndLpipsTests(unittest.TestCase):
    def test_run_pair_requires_exact_relative_mp4_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense = root / "dense"
            candidate = root / "candidate"
            dense.mkdir()
            candidate.mkdir()
            (dense / "sample.mp4").write_bytes(b"dense")
            (candidate / "sample.mp4").write_bytes(b"candidate")
            args = argparse.Namespace(
                dense=None,
                candidate=None,
                dense_run=dense,
                candidate_run=candidate,
            )

            pairs = TOOL._resolve_video_pairs(args)

            self.assertEqual(pairs, [(dense / "sample.mp4", candidate / "sample.mp4")])
            (candidate / "extra.mp4").write_bytes(b"extra")
            with self.assertRaisesRegex(TOOL.ComparisonError, "same relative MP4"):
                TOOL._resolve_video_pairs(args)

    def test_required_lpips_fails_closed_and_is_lazily_loaded(self):
        self.assertNotIn("compare_ovi_quality", TOOL._MODULE_CACHE)
        with mock.patch.object(
            TOOL,
            "_load_sibling_module",
            side_effect=ModuleNotFoundError("lpips unavailable"),
        ):
            with self.assertRaisesRegex(TOOL.ComparisonError, "required pinned LPIPS"):
                TOOL.PinnedLpipsEvaluator(Path("protocol.json"), None)


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def _write_rows(path: Path) -> None:
        rows = [
            {
                "split": "development",
                "prompt_id": "dev-a",
                "category": "speech",
                "seed": 103,
                "metric": "video_ssim",
                "candidate_value": 0.94,
                "comparator_value": 0.90,
            },
            {
                "split": "development",
                "prompt_id": "dev-a",
                "category": "speech",
                "seed": 211,
                "metric": "video_ssim",
                "candidate_value": 0.93,
                "comparator_value": 0.91,
            },
            {
                "split": "development",
                "prompt_id": "dev-b",
                "category": "motion",
                "seed": 103,
                "metric": "video_ssim",
                "candidate_value": 0.89,
                "comparator_value": 0.90,
            },
            {
                "split": "heldout",
                "prompt_id": "held-a",
                "category": "music",
                "seed": 503,
                "metric": "video_ssim",
                "candidate_value": 0.95,
                "comparator_value": 0.92,
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_cluster_bootstrap_is_deterministic_and_splits_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.csv"
            self._write_rows(path)
            rows = TOOL.load_analysis_rows([path])

        first = TOOL.analyze_rows(rows, bootstrap_replicates=200, bootstrap_seed=77)
        second = TOOL.analyze_rows(rows, bootstrap_replicates=200, bootstrap_seed=77)

        self.assertEqual(first, second)
        development = first["splits"]["development"]
        heldout = first["splits"]["heldout"]
        self.assertEqual(development["prompt_cluster_count"], 2)
        self.assertEqual(heldout["prompt_cluster_count"], 1)
        self.assertEqual(
            set(development["metrics"]["video_ssim"]["categories"]),
            {"speech", "motion"},
        )
        self.assertAlmostEqual(
            development["metrics"]["video_ssim"]["overall"]["win_rate"],
            2.0 / 3.0,
        )
        self.assertEqual(first["cross_split_aggregation"], "forbidden")
        self.assertEqual(first["pending_evaluations"]["asr"]["status"], "pending")

    def test_lower_is_better_is_oriented_toward_candidate_win(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "split": "development",
                            "prompt": "dev-a",
                            "category": "speech",
                            "seed": 103,
                            "metric": "lpips_mean",
                            "candidate_value": 0.1,
                            "comparator_value": 0.2,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            rows = TOOL.load_analysis_rows([path])

        self.assertAlmostEqual(rows[0]["raw_difference"], -0.1)
        self.assertAlmostEqual(rows[0]["oriented_difference"], 0.1)

    def test_prompt_overlap_between_splits_fails_closed(self):
        common = {
            "prompt": "same-prompt",
            "category": "speech",
            "seed": 103,
            "metric": "video_ssim",
            "higher_is_better": True,
            "oriented_difference": 0.1,
        }
        rows = [
            {**common, "split": "development"},
            {**common, "split": "heldout", "seed": 503},
        ]

        with self.assertRaisesRegex(TOOL.ComparisonError, "overlap"):
            TOOL.analyze_rows(rows, bootstrap_replicates=10, bootstrap_seed=1)


if __name__ == "__main__":
    unittest.main()
