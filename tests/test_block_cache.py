import ast
from dataclasses import dataclass
import math
from pathlib import Path
import unittest

from ovi.block_cache import (
    FusionBlockCache,
    _pool_hidden_for_cosine,
    expected_fixed_block_cache_metrics,
    fixed_block_cache_metric_errors,
    validate_block_cache_config,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CpuMockTensor:
    values: tuple
    shape: tuple
    dtype: str = "bfloat16"
    device: str = "cpu"

    def detach(self):
        return self


class RecordingPooledTensor:
    def __init__(self, events):
        self.events = events

    def detach(self):
        self.events.append(("detach",))
        return self


class RecordingHiddenTensor:
    shape = (2, 3, 4)

    def __init__(self):
        self.events = []
        self.pooled = RecordingPooledTensor(self.events)

    def mean(self, *, dim):
        self.events.append(("mean", dim))
        return self.pooled


def tensor(values, *, shape=None, dtype="bfloat16", device="cpu"):
    values = tuple(float(value) for value in values)
    return CpuMockTensor(
        values=values,
        shape=tuple(shape if shape is not None else (len(values),)),
        dtype=dtype,
        device=device,
    )


def pair(label, *, video=None, audio=None):
    return (
        video or tensor((1.0, 2.0), shape=(1, 2)),
        audio or tensor((3.0,), shape=(1, 1)),
    )


def cosine(left, right):
    numerator = sum(a * b for a, b in zip(left.values, right.values))
    left_norm = math.sqrt(sum(value * value for value in left.values))
    right_norm = math.sqrt(sum(value * value for value in right.values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class FusionBlockCacheStateMachineTests(unittest.TestCase):
    def resolve(
        self,
        cache,
        step,
        branch,
        input_pair,
        compute_window,
        *,
        slg_layer=0,
    ):
        skipped = (slg_layer,) if slg_layer > 0 else ()
        return cache.resolve(
            step=step,
            branch=branch,
            input_pair=input_pair,
            slg_signature=("slg_layer", slg_layer, "num_blocks", 30),
            skipped_blocks=skipped,
            compute_window=compute_window,
        )

    def test_fixed_policy_is_compute_reuse_compute(self):
        cache = FusionBlockCache(10, 19, "fixed", 0.95, 1, num_blocks=30)
        calls = []
        actions = []
        outputs = []

        for step in range(3):
            def compute_window(step=step):
                calls.append(step)
                return pair(f"output-{step}")

            output, action = self.resolve(
                cache,
                step,
                "conditional",
                pair(f"input-{step}"),
                compute_window,
            )
            actions.append(action)
            outputs.append(output)

        self.assertEqual(actions, ["refresh", "hit", "refresh"])
        self.assertEqual(calls, [0, 2])
        self.assertIs(outputs[1][0], outputs[0][0])
        metrics = cache.metrics()
        self.assertEqual(metrics["block_cache_hits"], 1)
        self.assertEqual(metrics["block_cache_refreshes"], 2)
        self.assertEqual(
            metrics["block_cache_saved_video_self_attention_calls"], 10
        )
        self.assertEqual(
            metrics["block_cache_branch_metrics"]["conditional"][
                "refresh_reasons"
            ],
            {"empty": 1, "max_consecutive_reuses": 1},
        )

    def test_conditional_and_unconditional_payloads_are_physically_isolated(self):
        cache = FusionBlockCache(num_blocks=30)
        conditional_output = pair("conditional")
        unconditional_output = pair("unconditional")

        self.resolve(
            cache,
            0,
            "conditional",
            pair("conditional-input"),
            lambda: conditional_output,
        )
        self.resolve(
            cache,
            0,
            "unconditional",
            pair("unconditional-input"),
            lambda: unconditional_output,
            slg_layer=11,
        )
        conditional_hit, conditional_action = self.resolve(
            cache,
            1,
            "conditional",
            pair("conditional-next"),
            lambda: self.fail("conditional hit unexpectedly recomputed"),
        )
        unconditional_hit, unconditional_action = self.resolve(
            cache,
            1,
            "unconditional",
            pair("unconditional-next"),
            lambda: self.fail("unconditional hit unexpectedly recomputed"),
            slg_layer=11,
        )

        self.assertEqual(conditional_action, "hit")
        self.assertEqual(unconditional_action, "hit")
        self.assertIs(conditional_hit[0], conditional_output[0])
        self.assertIs(unconditional_hit[0], unconditional_output[0])
        self.assertIsNot(conditional_hit[0], unconditional_hit[0])
        self.assertIsNot(
            cache._branches["conditional"], cache._branches["unconditional"]
        )
        branch_metrics = cache.metrics()["block_cache_branch_metrics"]
        self.assertEqual(
            branch_metrics["conditional"][
                "saved_video_self_attention_calls"
            ],
            10,
        )
        # SLG already skips block 11, so a negative cache hit saves only the
        # other nine video-self-attention calls in blocks 10..19.
        self.assertEqual(
            branch_metrics["unconditional"][
                "saved_video_self_attention_calls"
            ],
            9,
        )

    def test_step_gap_forces_negative_refresh_after_cfg_skip(self):
        cache = FusionBlockCache(num_blocks=30)
        calls = []
        self.resolve(
            cache,
            10,
            "unconditional",
            pair("negative-10"),
            lambda: (calls.append(10) or pair("output-10")),
            slg_layer=11,
        )
        _, action = self.resolve(
            cache,
            15,
            "unconditional",
            pair("negative-15"),
            lambda: (calls.append(15) or pair("output-15")),
            slg_layer=11,
        )
        self.assertEqual(action, "refresh")
        self.assertEqual(calls, [10, 15])
        reasons = cache.metrics()["block_cache_branch_metrics"][
            "unconditional"
        ]["refresh_reasons"]
        self.assertEqual(reasons, {"empty": 1, "step_gap": 1})

    def test_full_cfg_schedule_never_reuses_across_skipped_negative_steps(self):
        cache = FusionBlockCache(num_blocks=30)
        cfg_negative_steps = [
            step
            for step in range(50)
            if not (
                10 <= step <= 39
                and (step - 10) % 5 != 0
            )
        ]
        self.assertEqual(len(cfg_negative_steps), 26)

        for step in range(50):
            self.resolve(
                cache,
                step,
                "conditional",
                pair(f"conditional-{step}"),
                lambda step=step: pair(f"conditional-output-{step}"),
            )
        for step in cfg_negative_steps:
            self.resolve(
                cache,
                step,
                "unconditional",
                pair(f"negative-{step}"),
                lambda step=step: pair(f"negative-output-{step}"),
                slg_layer=11,
            )

        branches = cache.metrics()["block_cache_branch_metrics"]
        self.assertEqual(
            (branches["conditional"]["hits"], branches["conditional"]["refreshes"]),
            (25, 25),
        )
        self.assertEqual(
            (
                branches["unconditional"]["hits"],
                branches["unconditional"]["refreshes"],
            ),
            (10, 16),
        )
        self.assertEqual(
            branches["unconditional"]["refresh_reasons"]["step_gap"], 6
        )
        self.assertEqual(
            cache.metrics()["block_cache_saved_video_self_attention_calls"],
            340,
        )

    def test_shape_dtype_device_and_slg_mismatch_force_refresh(self):
        variants = {
            "video_shape_mismatch": pair(
                "shape", video=tensor((1.0, 2.0), shape=(2, 1))
            ),
            "video_dtype_mismatch": pair(
                "dtype",
                video=tensor((1.0, 2.0), shape=(1, 2), dtype="float32"),
            ),
            "video_device_mismatch": pair(
                "device",
                video=tensor((1.0, 2.0), shape=(1, 2), device="cpu:1"),
            ),
            "audio_shape_mismatch": pair(
                "audio-shape", audio=tensor((3.0,), shape=(1,))
            ),
            "audio_dtype_mismatch": pair(
                "audio-dtype",
                audio=tensor((3.0,), shape=(1, 1), dtype="float32"),
            ),
            "audio_device_mismatch": pair(
                "audio-device",
                audio=tensor((3.0,), shape=(1, 1), device="cpu:1"),
            ),
        }
        for expected_reason, next_input in variants.items():
            with self.subTest(expected_reason=expected_reason):
                cache = FusionBlockCache(num_blocks=30)
                self.resolve(
                    cache,
                    0,
                    "conditional",
                    pair("base"),
                    lambda: pair("base-output"),
                )
                _, action = self.resolve(
                    cache,
                    1,
                    "conditional",
                    next_input,
                    lambda next_input=next_input: next_input,
                )
                self.assertEqual(action, "refresh")
                reasons = cache.metrics()["block_cache_branch_metrics"][
                    "conditional"
                ]["refresh_reasons"]
                self.assertEqual(reasons[expected_reason], 1)

        cache = FusionBlockCache(num_blocks=30)
        self.resolve(
            cache,
            0,
            "unconditional",
            pair("negative"),
            lambda: pair("output"),
            slg_layer=11,
        )
        _, action = self.resolve(
            cache,
            1,
            "unconditional",
            pair("negative-next"),
            lambda: pair("new-output"),
            slg_layer=12,
        )
        self.assertEqual(action, "refresh")
        self.assertEqual(
            cache.metrics()["block_cache_branch_metrics"]["unconditional"][
                "refresh_reasons"
            ]["slg_signature_mismatch"],
            1,
        )

    def test_cosine_policy_uses_minimum_of_video_and_audio(self):
        low_audio_cache = FusionBlockCache(
            policy="cosine",
            cosine_threshold=0.9,
            num_blocks=30,
            cosine_fn=cosine,
            pool_fn=lambda value: value,
        )
        base = (
            tensor((1.0, 0.0), shape=(1, 2)),
            tensor((1.0, 0.0), shape=(1, 2)),
        )
        self.resolve(
            low_audio_cache, 0, "conditional", base, lambda: base
        )
        low_audio_input = (
            tensor((1.0, 0.0), shape=(1, 2)),
            tensor((0.0, 1.0), shape=(1, 2)),
        )
        _, action = self.resolve(
            low_audio_cache,
            1,
            "conditional",
            low_audio_input,
            lambda: low_audio_input,
        )
        self.assertEqual(action, "refresh")
        branch = low_audio_cache.metrics()["block_cache_branch_metrics"][
            "conditional"
        ]
        self.assertAlmostEqual(branch["last_video_cosine"], 1.0)
        self.assertAlmostEqual(branch["last_audio_cosine"], 0.0)
        self.assertAlmostEqual(branch["last_min_cosine"], 0.0)
        self.assertEqual(branch["refresh_reasons"]["cosine_below_threshold"], 1)

        high_cache = FusionBlockCache(
            policy="cosine",
            cosine_threshold=0.9,
            num_blocks=30,
            cosine_fn=cosine,
            pool_fn=lambda value: value,
        )
        self.resolve(
            high_cache, 0, "conditional", base, lambda: base
        )
        _, action = self.resolve(
            high_cache,
            1,
            "conditional",
            (
                tensor((0.99, 0.01), shape=(1, 2)),
                tensor((0.98, 0.02), shape=(1, 2)),
            ),
            lambda: self.fail("high-similarity pair should hit"),
        )
        self.assertEqual(action, "hit")

    def test_production_cosine_pooling_is_mean_dim_one_then_detach(self):
        hidden = RecordingHiddenTensor()

        pooled = _pool_hidden_for_cosine(hidden)

        self.assertIs(pooled, hidden.pooled)
        self.assertEqual(hidden.events, [("mean", 1), ("detach",)])
        with self.assertRaisesRegex(ValueError, "at least two dimensions"):
            _pool_hidden_for_cosine(tensor((1.0,), shape=(1,)))

    def test_cosine_cache_retains_only_pooled_input_pair(self):
        pooled_values = []

        def pool_fn(_value):
            pooled = object()
            pooled_values.append(pooled)
            return pooled

        cache = FusionBlockCache(
            policy="cosine",
            num_blocks=30,
            pool_fn=pool_fn,
            cosine_fn=lambda _left, _right: 1.0,
        )
        input_pair = pair("full-input")
        output_pair = pair("window-output")

        self.resolve(
            cache,
            0,
            "conditional",
            input_pair,
            lambda: output_pair,
        )

        record = cache._branches["conditional"]
        self.assertEqual(
            record.cached_pooled_input_pair,
            tuple(pooled_values),
        )
        self.assertFalse(hasattr(record, "cached_input_pair"))
        for full_input, pooled in zip(
            input_pair, record.cached_pooled_input_pair
        ):
            self.assertIsNot(pooled, full_input)
        self.assertIs(record.cached_output_pair[0], output_pair[0])

    def test_nonfinite_video_or_audio_cosine_forces_refresh(self):
        for stream_index, stream_name in enumerate(("video", "audio")):
            for nonfinite in (float("nan"), float("inf")):
                with self.subTest(stream=stream_name, value=nonfinite):
                    similarities = [1.0, 1.0]
                    similarities[stream_index] = nonfinite
                    results = iter(similarities)
                    cache = FusionBlockCache(
                        policy="cosine",
                        num_blocks=30,
                        pool_fn=lambda value: value,
                        cosine_fn=lambda _left, _right: next(results),
                    )
                    base_input = pair("base")
                    next_input = pair("next")
                    calls = []
                    self.resolve(
                        cache,
                        0,
                        "conditional",
                        base_input,
                        lambda: pair("base-output"),
                    )
                    _, action = self.resolve(
                        cache,
                        1,
                        "conditional",
                        next_input,
                        lambda: (
                            calls.append("refresh") or pair("next-output")
                        ),
                    )

                    branch = cache.metrics()["block_cache_branch_metrics"][
                        "conditional"
                    ]
                    self.assertEqual(action, "refresh")
                    self.assertEqual(calls, ["refresh"])
                    self.assertEqual(branch["hits"], 0)
                    self.assertEqual(
                        branch["refresh_reasons"]["cosine_nonfinite"], 1
                    )
                    self.assertIsNone(branch["last_min_cosine"])

    def test_window_output_metadata_must_match_each_input_stream(self):
        variants = {
            "video_shape_mismatch": pair(
                "video-shape",
                video=tensor((1.0, 2.0), shape=(2, 1)),
            ),
            "video_dtype_mismatch": pair(
                "video-dtype",
                video=tensor(
                    (1.0, 2.0), shape=(1, 2), dtype="float32"
                ),
            ),
            "video_device_mismatch": pair(
                "video-device",
                video=tensor(
                    (1.0, 2.0), shape=(1, 2), device="cpu:1"
                ),
            ),
            "audio_shape_mismatch": pair(
                "audio-shape",
                audio=tensor((3.0,), shape=(1,)),
            ),
            "audio_dtype_mismatch": pair(
                "audio-dtype",
                audio=tensor((3.0,), shape=(1, 1), dtype="float32"),
            ),
            "audio_device_mismatch": pair(
                "audio-device",
                audio=tensor((3.0,), shape=(1, 1), device="cpu:1"),
            ),
        }
        for mismatch, bad_output in variants.items():
            with self.subTest(mismatch=mismatch):
                cache = FusionBlockCache(num_blocks=30)
                with self.assertRaisesRegex(
                    ValueError, f"mismatch={mismatch}"
                ):
                    self.resolve(
                        cache,
                        0,
                        "conditional",
                        pair("input"),
                        lambda bad_output=bad_output: bad_output,
                    )
                self.assertFalse(cache.has_cached_pair())
                self.assertEqual(cache.metrics()["block_cache_refreshes"], 0)

    def test_mutated_cached_output_metadata_forces_refresh(self):
        cache = FusionBlockCache(num_blocks=30)
        self.resolve(
            cache,
            0,
            "conditional",
            pair("base"),
            lambda: pair("base-output"),
        )
        record = cache._branches["conditional"]
        record.cached_output_pair = (
            tensor((1.0, 2.0), shape=(1, 2), dtype="float32"),
            record.cached_output_pair[1],
        )
        calls = []

        _, action = self.resolve(
            cache,
            1,
            "conditional",
            pair("next"),
            lambda: (calls.append("refresh") or pair("new-output")),
        )

        self.assertEqual(action, "refresh")
        self.assertEqual(calls, ["refresh"])
        self.assertEqual(
            cache.metrics()["block_cache_branch_metrics"]["conditional"][
                "refresh_reasons"
            ]["cached_output_video_dtype_mismatch"],
            1,
        )

    def test_failed_partial_refresh_does_not_publish_half_pair(self):
        cache = FusionBlockCache(num_blocks=30)
        original, _ = self.resolve(
            cache,
            0,
            "unconditional",
            pair("negative"),
            lambda: pair("original-output"),
            slg_layer=11,
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.resolve(
                cache,
                1,
                "unconditional",
                pair("negative-next"),
                lambda: (pair("new-output")[0], None),
                slg_layer=12,
            )
        recovered, action = self.resolve(
            cache,
            1,
            "unconditional",
            pair("negative-retry"),
            lambda: self.fail("failed refresh replaced the intact cache"),
            slg_layer=11,
        )
        self.assertEqual(action, "hit")
        self.assertIs(recovered[0], original[0])
        self.assertEqual(cache.metrics()["block_cache_refreshes"], 1)

    def test_generation_finally_clears_both_branch_payloads(self):
        cache = FusionBlockCache(num_blocks=30)
        try:
            self.resolve(
                cache,
                0,
                "conditional",
                pair("conditional"),
                lambda: pair("conditional-output"),
            )
            self.resolve(
                cache,
                0,
                "unconditional",
                pair("unconditional"),
                lambda: pair("unconditional-output"),
                slg_layer=11,
            )
            self.assertTrue(cache.has_cached_pair("conditional"))
            self.assertTrue(cache.has_cached_pair("unconditional"))
            raise RuntimeError("synthetic generation failure")
        except RuntimeError:
            pass
        finally:
            cache.clear()
        self.assertFalse(cache.has_cached_pair())
        # Clearing payloads does not erase evidence counters needed by metrics.
        self.assertEqual(cache.metrics()["block_cache_refreshes"], 2)

    def test_configuration_validation(self):
        self.assertEqual(
            validate_block_cache_config(10, 19, "fixed", 0.95, 1),
            (10, 19, "fixed", 0.95, 1),
        )
        with self.assertRaisesRegex(ValueError, "start_block"):
            validate_block_cache_config(-1, 19)
        with self.assertRaisesRegex(ValueError, "end_block"):
            validate_block_cache_config(10, 9)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_block_cache_config(10, 30, num_blocks=30)
        with self.assertRaisesRegex(ValueError, "policy"):
            validate_block_cache_config(10, 19, policy="unknown")
        with self.assertRaisesRegex(ValueError, "cosine_threshold"):
            validate_block_cache_config(10, 19, cosine_threshold=1.1)
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            validate_block_cache_config(
                10, 19, max_consecutive_reuses=2
            )


class FixedBlockCacheScheduleTests(unittest.TestCase):
    def expected(self, sample_steps, *, use_cfg_cache):
        return expected_fixed_block_cache_metrics(
            sample_steps=sample_steps,
            use_cfg_cache=use_cfg_cache,
            cfg_cache_start_step=10,
            cfg_cache_end_step=39,
            cfg_cache_refresh_interval=5,
            block_cache_start_block=10,
            block_cache_end_block=19,
            block_cache_max_consecutive_reuses=1,
            slg_layer=11,
        )

    def test_no_cfg_smoke_20_step_fixed_schedule_is_exact(self):
        expected = self.expected(20, use_cfg_cache=False)
        branches = expected["block_cache_branch_metrics"]

        self.assertEqual(
            (
                expected["block_cache_hits"],
                expected["block_cache_refreshes"],
                expected["block_cache_saved_video_self_attention_calls"],
            ),
            (20, 20, 190),
        )
        self.assertEqual(
            (branches["conditional"]["hits"], branches["conditional"]["refreshes"]),
            (10, 10),
        )
        self.assertEqual(
            (
                branches["unconditional"]["hits"],
                branches["unconditional"]["refreshes"],
            ),
            (10, 10),
        )
        self.assertEqual(
            branches["conditional"]["saved_video_self_attention_calls"],
            100,
        )
        self.assertEqual(
            branches["unconditional"]["saved_video_self_attention_calls"],
            90,
        )

    def test_no_cfg_formal_50_step_fixed_schedule_is_exact(self):
        expected = self.expected(50, use_cfg_cache=False)
        branches = expected["block_cache_branch_metrics"]

        self.assertEqual(
            (
                expected["block_cache_hits"],
                expected["block_cache_refreshes"],
                expected["block_cache_saved_video_self_attention_calls"],
            ),
            (50, 50, 475),
        )
        for branch in ("conditional", "unconditional"):
            self.assertEqual(
                (branches[branch]["hits"], branches[branch]["refreshes"]),
                (25, 25),
            )
        self.assertEqual(
            branches["conditional"]["saved_video_self_attention_calls"],
            250,
        )
        self.assertEqual(
            branches["unconditional"]["saved_video_self_attention_calls"],
            225,
        )

    def test_cfg_50_step_fixed_schedule_tracks_negative_forwards(self):
        expected = self.expected(50, use_cfg_cache=True)
        branches = expected["block_cache_branch_metrics"]

        self.assertEqual(
            (
                expected["block_cache_hits"],
                expected["block_cache_refreshes"],
                expected["block_cache_saved_video_self_attention_calls"],
            ),
            (35, 41, 340),
        )
        self.assertEqual(
            (
                branches["conditional"]["hits"],
                branches["conditional"]["refreshes"],
            ),
            (25, 25),
        )
        self.assertEqual(
            (
                branches["unconditional"]["forward_count"],
                branches["unconditional"]["hits"],
                branches["unconditional"]["refreshes"],
            ),
            (26, 10, 16),
        )
        self.assertEqual(
            branches["unconditional"]["refresh_reasons"],
            {"empty": 1, "max_consecutive_reuses": 9, "step_gap": 6},
        )

    def test_metric_validator_rejects_self_consistent_wrong_counts(self):
        config = {
            "sample_steps": 50,
            "use_cfg_cache": True,
            "cfg_cache_start_step": 10,
            "cfg_cache_end_step": 39,
            "cfg_cache_refresh_interval": 5,
            "block_cache_start_block": 10,
            "block_cache_end_block": 19,
            "block_cache_max_consecutive_reuses": 1,
            "slg_layer": 11,
            "cfg_negative_forwards": 26,
        }
        expected = self.expected(50, use_cfg_cache=True)
        actual_branches = {
            branch: {
                field: value
                for field, value in branch_metrics.items()
                if field != "forward_count"
            }
            for branch, branch_metrics in expected[
                "block_cache_branch_metrics"
            ].items()
        }
        metrics = {
            **config,
            **expected,
            "block_cache_branch_metrics": actual_branches,
        }
        self.assertEqual(fixed_block_cache_metric_errors(metrics), [])

        # Keep aggregate and branch sums internally consistent while making
        # the conditional schedule one hit short.  The exact verifier must
        # still reject a record that the old sum-only checks accepted.
        actual_branches["conditional"]["hits"] -= 1
        actual_branches["conditional"][
            "saved_video_self_attention_calls"
        ] -= 10
        metrics["block_cache_hits"] -= 1
        metrics["block_cache_saved_video_self_attention_calls"] -= 10
        errors = fixed_block_cache_metric_errors(metrics)
        self.assertTrue(
            any("conditional.hits" in error for error in errors), errors
        )
        self.assertTrue(
            any("conditional forward count" in error for error in errors),
            errors,
        )

        metrics["cfg_negative_forwards"] = 25
        errors = fixed_block_cache_metric_errors(metrics)
        self.assertTrue(
            any("cfg_negative_forwards" in error for error in errors), errors
        )


class FusionBlockCacheMockAndAstTests(unittest.TestCase):
    def test_mock_block_execution_preserves_slg_and_skips_whole_window_on_hit(self):
        cache = FusionBlockCache(num_blocks=30)

        def mock_forward(step, branch, slg_layer):
            calls = []
            current = pair(f"{branch}-input")

            def run_block(block_index, state):
                if slg_layer > 0 and block_index == slg_layer:
                    return state
                calls.append(block_index)
                return state

            for block_index in range(30):
                if 10 <= block_index <= 19:
                    if block_index != 10:
                        continue

                    def compute_window():
                        state = current
                        for cached_block in range(10, 20):
                            state = run_block(cached_block, state)
                        return state

                    current, _ = cache.resolve(
                        step=step,
                        branch=branch,
                        input_pair=current,
                        slg_signature=("slg_layer", slg_layer),
                        skipped_blocks=((slg_layer,) if slg_layer > 0 else ()),
                        compute_window=compute_window,
                    )
                    continue
                current = run_block(block_index, current)
            return calls

        conditional_compute = mock_forward(0, "conditional", 0)
        conditional_hit = mock_forward(1, "conditional", 0)
        negative_compute = mock_forward(0, "unconditional", 11)
        negative_hit = mock_forward(1, "unconditional", 11)

        self.assertEqual(conditional_compute, list(range(30)))
        self.assertEqual(
            conditional_hit, list(range(10)) + list(range(20, 30))
        )
        self.assertEqual(
            negative_compute, [block for block in range(30) if block != 11]
        )
        self.assertEqual(
            negative_hit, list(range(10)) + list(range(20, 30))
        )
        self.assertNotIn(11, negative_compute)

    def test_engine_keeps_state_local_to_generate_and_clears_it_in_finally(self):
        source = (ROOT / "ovi" / "ovi_fusion_engine.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        engine_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "OviFusionEngine"
        )
        generate = next(
            node
            for node in engine_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "generate"
        )

        self_attributes = {
            node.attr
            for node in ast.walk(engine_class)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        }
        self.assertNotIn("block_cache_state", self_attributes)
        local_assignments = {
            target.id
            for node in ast.walk(generate)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertIn("block_cache_state", local_assignments)
        clear_calls = [
            node
            for node in ast.walk(generate)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "clear"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "block_cache_state"
        ]
        self.assertEqual(len(clear_calls), 1)

    def test_fusion_dense_branch_has_no_cache_lookup(self):
        source = (ROOT / "ovi" / "modules" / "fusion.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        forward = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        cache_if = next(
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "block_cache_state"
        )
        dense_names = {
            node.id for statement in cache_if.body for node in ast.walk(statement)
            if isinstance(node, ast.Name)
        }
        self.assertNotIn("FusionBlockCache", dense_names)
        self.assertNotIn("block_cache_context", dense_names)
        resolve_calls = [
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "resolve"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "block_cache_state"
        ]
        self.assertEqual(len(resolve_calls), 1)

    def test_verifier_binds_fixed_metrics_to_run_schedule_and_environment(self):
        verifier_source = (
            ROOT / "scripts" / "verify_ovi_output.py"
        ).read_text(encoding="utf-8")
        verifier_tree = ast.parse(verifier_source)
        fixed_schedule_calls = [
            node
            for node in ast.walk(verifier_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "fixed_block_cache_metric_errors"
        ]
        # One call validates each artifact sidecar; the other covers every
        # warm-up/timing record in the run protocol.
        self.assertEqual(len(fixed_schedule_calls), 2)

        protocol = next(
            node
            for node in verifier_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "verify_run_protocol"
        )
        schedule_assignment = next(
            node
            for node in ast.walk(protocol)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "schedule_fields"
                for target in node.targets
            )
        )
        schedule_fields = set(ast.literal_eval(schedule_assignment.value))
        self.assertTrue(
            {
                "sample_steps",
                "slg_layer",
                "use_block_cache",
                "use_cfg_cache",
                "cfg_cache_start_step",
                "cfg_cache_end_step",
                "cfg_cache_refresh_interval",
                "block_cache_start_block",
                "block_cache_end_block",
                "block_cache_policy",
                "block_cache_cosine_threshold",
                "block_cache_max_consecutive_reuses",
            }.issubset(schedule_fields)
        )

        inference_tree = ast.parse(
            (ROOT / "inference.py").read_text(encoding="utf-8")
        )
        collect_environment = next(
            node
            for node in inference_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_collect_environment"
        )
        environment_return = next(
            node
            for node in ast.walk(collect_environment)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Dict)
        )
        environment_keys = {
            key.value
            for key in environment_return.value.keys
            if isinstance(key, ast.Constant)
        }
        self.assertIn("sample_steps", environment_keys)
        self.assertIn("slg_layer", environment_keys)


if __name__ == "__main__":
    unittest.main()
