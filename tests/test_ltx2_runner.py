import csv
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ltx2.inference import (
    DISTILLED_STEPS,
    PromptCase,
    TimedDiffusionStage,
    append_result,
    build_parser,
    load_prompt_cases,
    validate_sparse_metrics,
)
from ltx2 import inference as ltx_inference


class _FakeCuda:
    def __init__(self):
        self.syncs = 0

    def is_available(self):
        return True

    def synchronize(self):
        self.syncs += 1


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()


class LTX2RunnerTests(unittest.TestCase):
    def test_official_distilled_schedule_is_eleven_steps(self):
        self.assertEqual(DISTILLED_STEPS, 11)

    def test_inline_prompt(self):
        self.assertEqual(
            load_prompt_cases(prompt="  a bell rings  ", prompt_id="speech 1", prompts_csv=None),
            [PromptCase("speech-1", "a bell rings")],
        )

    def test_prompt_csv_accepts_prompt_id_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.csv"
            path.write_text("prompt_id,prompt\na,first\nb,second\n", encoding="utf-8")
            self.assertEqual(
                load_prompt_cases(prompt=None, prompt_id="unused", prompts_csv=path),
                [PromptCase("a", "first"), PromptCase("b", "second")],
            )
            path.write_text("id,prompt\na,first\na,second\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_prompt_cases(prompt=None, prompt_id="unused", prompts_csv=path)

    def test_prompt_csv_accepts_setup_script_text_prompt_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.csv"
            path.write_text(
                "prompt_id,text_prompt\nspeech,A woman says hello.\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_prompt_cases(prompt=None, prompt_id="unused", prompts_csv=path),
                [PromptCase("speech", "A woman says hello.")],
            )

    def test_requires_exactly_one_prompt_source(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_prompt_cases(prompt=None, prompt_id="x", prompts_csv=None)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_prompt_cases(prompt="x", prompt_id="x", prompts_csv="prompts.csv")

    def test_timed_stage_delegates_and_synchronizes(self):
        stage_calls = []
        loop_calls = []

        def official_loop(value, *, scale):
            loop_calls.append((value, scale))
            return value * scale

        class Stage:
            marker = "official"

            def __call__(self, value, *, scale, loop):
                stage_calls.append((value, scale))
                return loop(value, scale=scale)

        fake_torch = _FakeTorch()
        timer = TimedDiffusionStage(Stage(), fake_torch, official_loop)
        self.assertEqual(timer.marker, "official")
        self.assertEqual(timer(3, scale=4), 12)
        self.assertEqual(stage_calls, [(3, 4)])
        self.assertEqual(loop_calls, [(3, 4)])
        self.assertEqual(fake_torch.cuda.syncs, 4)
        self.assertGreaterEqual(timer.stage_seconds, timer.denoise_seconds)
        self.assertGreaterEqual(timer.denoise_seconds, 0.0)
        timer.reset()
        self.assertEqual(timer.stage_seconds, 0.0)
        self.assertEqual(timer.denoise_seconds, 0.0)

    def test_sparse_metrics_must_cover_every_block_without_fallback(self):
        valid = {
            "errors": 0,
            "sparse_calls": 9,
            "fallback_count": 0,
            "calls_by_block": {"0": 3, "1": 3, "2": 3},
        }
        validate_sparse_metrics(valid, 3)

        for key, value, pattern in (
            ("sparse_calls", 0, "no sparse calls"),
            ("fallback_count", 1, "dense fallback"),
            ("errors", 1, "reported errors"),
        ):
            metrics = dict(valid)
            metrics[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, pattern):
                validate_sparse_metrics(metrics, 3)

        missing = dict(valid)
        missing["calls_by_block"] = {"0": 3, "2": 3}
        with self.assertRaisesRegex(RuntimeError, "cover every"):
            validate_sparse_metrics(missing, 3)

    def test_append_jsonl_and_csv(self):
        record = {
            "model": "ltx-2.3-dev",
            "method": "dense",
            "status": "ok",
            "attention_metrics": {"backend": "official_dense"},
        }
        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory) / "results.jsonl"
            append_result(jsonl, record)
            self.assertEqual(json.loads(jsonl.read_text(encoding="utf-8"))["status"], "ok")

            csv_path = Path(directory) / "results.csv"
            append_result(csv_path, record)
            append_result(csv_path, record)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(json.loads(rows[0]["attention_metrics"])["backend"], "official_dense")

    def test_parser_does_not_import_ltx_runtime(self):
        args = build_parser().parse_args(
            [
                "--checkpoint",
                "/tmp/model.safetensors",
                "--gemma-root",
                "/tmp/gemma",
                "--prompt",
                "test",
                "--output-dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(args.pipeline, "distilled")
        self.assertEqual(args.method, "dense")
        self.assertEqual(args.height, 512)
        self.assertEqual(args.width, 768)
        self.assertEqual(args.num_frames, 121)

    def test_results_suffix_must_be_jsonl_or_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "jsonl or .csv"):
                append_result(Path(directory) / "results.txt", {})

    def test_failed_first_prompt_releases_decoder_before_second_prompt(self):
        events = []

        class ClosableVideo:
            def __init__(self, name):
                self.name = name
                self.closed = False

            def close(self):
                self.closed = True
                events.append(f"{self.name}:closed")

        class RuntimeTorch(_FakeTorch):
            def inference_mode(self):
                return nullcontext()

        class Pipeline:
            stage = lambda *args, **kwargs: None

        first_video = ClosableVideo("first")
        second_video = ClosableVideo("second")
        calls = 0

        def invoke(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return first_video, object(), 0.1, 1
            self.assertTrue(first_video.closed)
            self.assertEqual(events[:2], ["first:closed", "cleanup"])
            events.append("second:started")
            return second_video, object(), 0.2, 1

        def decode(_args, _video, _audio, _torch, output_path, _chunks):
            output_path.write_bytes(b"mp4")
            return 0.05, {
                "decoded_video_frames": 121,
                "media_video_streams": 1,
                "media_audio_streams": 1,
                "media_video_duration_seconds": 5.0,
                "media_audio_duration_seconds": 5.0,
            }

        sampler_module = types.ModuleType("ltx_pipelines.utils.samplers")
        sampler_module.euler_denoising_loop = lambda *args, **kwargs: None
        helper_module = types.ModuleType("ltx_pipelines.utils.helpers")
        helper_module.cleanup_memory = lambda: events.append("cleanup")
        package_modules = {
            "ltx_pipelines": types.ModuleType("ltx_pipelines"),
            "ltx_pipelines.utils": types.ModuleType("ltx_pipelines.utils"),
            "ltx_pipelines.utils.samplers": sampler_module,
            "ltx_pipelines.utils.helpers": helper_module,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.csv"
            prompts.write_text("prompt_id,prompt\nfirst,one\nsecond,two\n", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "--checkpoint",
                    str(root / "model.safetensors"),
                    "--spatial-upsampler",
                    str(root / "upsampler.safetensors"),
                    "--gemma-root",
                    str(root / "gemma"),
                    "--prompts-csv",
                    str(prompts),
                    "--output-dir",
                    str(root / "outputs"),
                ]
            )
            audio_gate = [RuntimeError("first prompt gate failed"), {"audio_rms": 0.1}]
            with (
                mock.patch.dict(sys.modules, package_modules),
                mock.patch.object(ltx_inference, "_validate_runtime_paths"),
                mock.patch.object(
                    ltx_inference,
                    "_load_official_runtime",
                    return_value=(Pipeline(), RuntimeTorch(), DISTILLED_STEPS, "ltx-test"),
                ),
                mock.patch.object(
                    ltx_inference, "_install_attention_backend", return_value=(None, None)
                ),
                mock.patch.object(ltx_inference, "_invoke_official_pipeline", side_effect=invoke),
                mock.patch.object(
                    ltx_inference, "validate_audio_output", side_effect=audio_gate
                ),
                mock.patch.object(
                    ltx_inference, "_decode_encode_and_validate", side_effect=decode
                ),
                mock.patch.object(ltx_inference, "_reset_cuda_metrics"),
                mock.patch.object(
                    ltx_inference, "_peak_cuda_metrics", return_value=(123, 456)
                ),
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(ltx_inference.run(args), 1)

            records = [
                json.loads(line)
                for line in (root / "outputs" / "results.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual([record["status"] for record in records], ["failed", "ok"])
            self.assertEqual(records[0]["stage_seconds"], 0.0)
            self.assertEqual(records[0]["denoise_seconds"], 0.0)
            self.assertTrue(second_video.closed)
            self.assertEqual(
                events, ["first:closed", "cleanup", "second:started", "second:closed"]
            )


if __name__ == "__main__":
    unittest.main()
