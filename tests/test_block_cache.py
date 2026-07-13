import ast
from dataclasses import dataclass
import math
from pathlib import Path
import unittest

from ovi.block_cache import FusionBlockCache, validate_block_cache_config


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CpuMockTensor:
    values: tuple
    shape: tuple
    dtype: str = "bfloat16"
    device: str = "cpu"

    def detach(self):
        return self


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
                    lambda: pair("refreshed-output"),
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
        )
        base = (
            tensor((1.0, 0.0)),
            tensor((1.0, 0.0)),
        )
        self.resolve(
            low_audio_cache, 0, "conditional", base, lambda: pair("output")
        )
        _, action = self.resolve(
            low_audio_cache,
            1,
            "conditional",
            (tensor((1.0, 0.0)), tensor((0.0, 1.0))),
            lambda: pair("low-audio-refresh"),
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
        )
        self.resolve(
            high_cache, 0, "conditional", base, lambda: pair("output")
        )
        _, action = self.resolve(
            high_cache,
            1,
            "conditional",
            (tensor((0.99, 0.01)), tensor((0.98, 0.02))),
            lambda: self.fail("high-similarity pair should hit"),
        )
        self.assertEqual(action, "hit")

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


if __name__ == "__main__":
    unittest.main()
