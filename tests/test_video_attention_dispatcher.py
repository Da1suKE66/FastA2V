import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import unittest


DISPATCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "ovi"
    / "modules"
    / "video_attention_dispatcher.py"
)
SPEC = importlib.util.spec_from_file_location(
    "video_attention_dispatcher_under_test", DISPATCHER_PATH
)
DISPATCHER_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DISPATCHER_MODULE
SPEC.loader.exec_module(DISPATCHER_MODULE)

SUPPORTED_ATTENTION_METHODS = DISPATCHER_MODULE.SUPPORTED_ATTENTION_METHODS
VideoAttentionBackendUnavailableError = (
    DISPATCHER_MODULE.VideoAttentionBackendUnavailableError
)
VideoSelfAttentionDispatcher = DISPATCHER_MODULE.VideoSelfAttentionDispatcher
expected_video_self_attention_calls = (
    DISPATCHER_MODULE.expected_video_self_attention_calls
)


@dataclass(frozen=True)
class CpuMockTensor:
    values: tuple


class RecordingAttention:
    def __init__(self, label):
        self.label = label
        self.calls = []

    def __call__(self, x, seq_lens, grid_sizes, freqs):
        self.calls.append((x, seq_lens, grid_sizes, freqs))
        offset = len(seq_lens.values) + len(grid_sizes.values) + len(freqs.values)
        return CpuMockTensor(tuple(value * 2 + offset for value in x.values))


class VideoAttentionDispatcherTests(unittest.TestCase):
    def test_dense_output_exactly_matches_direct_original_call(self):
        x = CpuMockTensor((1.25, -2.0, 7.5))
        seq_lens = CpuMockTensor((3,))
        grid_sizes = CpuMockTensor((1, 1, 3))
        freqs = CpuMockTensor((0.5, 1.0))
        expected_attention = RecordingAttention("expected-video")
        actual_attention = RecordingAttention("actual-video")

        expected = expected_attention(x, seq_lens, grid_sizes, freqs)
        dispatcher = VideoSelfAttentionDispatcher("dense")
        actual = dispatcher(
            actual_attention,
            x,
            seq_lens,
            grid_sizes,
            freqs,
            block_index=4,
            debug_context={"branch": "conditional"},
        )

        self.assertEqual(actual, expected)
        self.assertEqual(actual_attention.calls, [(x, seq_lens, grid_sizes, freqs)])
        metrics = dispatcher.metrics()
        self.assertEqual(metrics["calls_total"], 1)
        self.assertEqual(metrics["calls_by_method"]["dense"], 1)
        self.assertFalse(metrics["fallback_allowed"])
        self.assertFalse(metrics["fallback_used"])
        self.assertEqual(metrics["fallback_count"], 0)

    def test_mock_fusion_routes_only_video_self_attention_through_dispatcher(self):
        dispatcher = VideoSelfAttentionDispatcher("dense")
        video_self_attention = RecordingAttention("video-self")
        audio_self_attention = RecordingAttention("audio-self")
        audio_cross_attention = RecordingAttention("audio-cross")
        video_cross_attention = RecordingAttention("video-cross")
        tensor = CpuMockTensor((1.0, 2.0))
        lengths = CpuMockTensor((2,))
        grid = CpuMockTensor((1, 1, 2))
        freqs = CpuMockTensor((0.5,))

        # This mirrors the call ownership in FusionModel: audio and cross
        # attention remain direct calls; only video self-attention is routed.
        audio_self_attention(tensor, lengths, grid, freqs)
        dispatcher(video_self_attention, tensor, lengths, grid, freqs)
        audio_cross_attention(tensor, lengths, grid, freqs)
        video_cross_attention(tensor, lengths, grid, freqs)

        self.assertEqual(dispatcher.metrics()["calls_total"], 1)
        self.assertEqual(len(video_self_attention.calls), 1)
        self.assertEqual(len(audio_self_attention.calls), 1)
        self.assertEqual(len(audio_cross_attention.calls), 1)
        self.assertEqual(len(video_cross_attention.calls), 1)

    def test_fusion_source_has_exactly_one_video_only_dispatch_site(self):
        source_path = (
            Path(__file__).resolve().parents[1] / "ovi" / "modules" / "fusion.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        attribute_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                owner_name = owner.id if isinstance(owner, ast.Name) else None
                attribute_calls.append((owner_name, node.func.attr))

        self.assertEqual(
            attribute_calls.count(("self", "video_self_attention_dispatcher")),
            1,
        )
        self.assertEqual(attribute_calls.count(("audio_block", "self_attn")), 1)
        self.assertEqual(attribute_calls.count(("vid_block", "self_attn")), 0)

    def test_all_unimplemented_sparse_methods_fail_fast(self):
        for method in ("sparge", "radial", "svg"):
            with self.subTest(method=method):
                with self.assertRaises(VideoAttentionBackendUnavailableError) as error:
                    VideoSelfAttentionDispatcher(method)
                self.assertIn("refusing to fall back", str(error.exception))

    def test_unknown_method_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "Unsupported attention_method"):
            VideoSelfAttentionDispatcher("typo")

    def test_sparse_backend_is_explicit_and_never_falls_back(self):
        original = RecordingAttention("dense-original")

        def failing_backend(*args, **kwargs):
            raise RuntimeError("backend failed")

        dispatcher = VideoSelfAttentionDispatcher(
            "sparge", backends={"sparge": failing_backend}
        )
        tensor = CpuMockTensor((1.0,))
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            dispatcher(original, tensor, tensor, tensor, tensor)

        self.assertEqual(original.calls, [])
        metrics = dispatcher.metrics()
        self.assertEqual(metrics["calls_by_method"]["sparge"], 1)
        self.assertEqual(metrics["errors_by_method"]["sparge"], 1)
        self.assertFalse(metrics["fallback_used"])
        self.assertEqual(metrics["fallback_count"], 0)

    def test_slg_call_count_for_fifty_steps(self):
        dispatcher = VideoSelfAttentionDispatcher("dense")
        attention = RecordingAttention("video-self")
        tensor = CpuMockTensor((0.0,))
        sample_steps = 50
        num_blocks = 30
        slg_layer = 11

        for _step in range(sample_steps):
            for _block in range(num_blocks):
                dispatcher(attention, tensor, tensor, tensor, tensor)
            for block in range(num_blocks):
                if slg_layer > 0 and block == slg_layer:
                    continue
                dispatcher(attention, tensor, tensor, tensor, tensor)

        expected = expected_video_self_attention_calls(
            sample_steps=sample_steps,
            num_blocks=num_blocks,
            slg_layer=slg_layer,
        )
        self.assertEqual(expected, 2_950)
        self.assertEqual(dispatcher.metrics()["calls_total"], expected)
        self.assertEqual(
            dispatcher.metrics()["calls_by_method"],
            {method: (expected if method == "dense" else 0)
             for method in SUPPORTED_ATTENTION_METHODS},
        )

    def test_cfg_cache_reduces_expected_negative_dispatches(self):
        self.assertEqual(
            expected_video_self_attention_calls(
                sample_steps=50,
                num_blocks=30,
                slg_layer=11,
                negative_forward_count=26,
            ),
            2_254,
        )

    def test_reset_metrics_starts_a_new_generation(self):
        dispatcher = VideoSelfAttentionDispatcher("dense")
        attention = RecordingAttention("video-self")
        tensor = CpuMockTensor((0.0,))
        dispatcher(attention, tensor, tensor, tensor, tensor)
        dispatcher.reset_metrics()
        self.assertEqual(dispatcher.metrics()["calls_total"], 0)


if __name__ == "__main__":
    unittest.main()
