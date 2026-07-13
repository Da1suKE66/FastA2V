import unittest

from ovi.cfg_cache import (
    CfgNegativeCache,
    expected_cfg_cache_metrics,
    validate_cfg_cache_config,
)


class CfgNegativeCacheTests(unittest.TestCase):
    def test_expected_schedule_metrics(self):
        self.assertEqual(
            expected_cfg_cache_metrics(50, 10, 39, 5),
            {
                "cfg_cache_hits": 24,
                "cfg_cache_refreshes": 6,
                "cfg_negative_forwards": 26,
            },
        )
        self.assertEqual(
            expected_cfg_cache_metrics(50, 10, 39, 1)[
                "cfg_negative_forwards"
            ],
            50,
        )
        self.assertEqual(
            expected_cfg_cache_metrics(8, 10, 39, 5)[
                "cfg_negative_forwards"
            ],
            8,
        )

    def test_steps_10_through_39_interval_5(self):
        cache = CfgNegativeCache(10, 39, 5)
        model_calls = []
        actions = []

        for step in range(50):
            def negative_forward(step=step):
                model_calls.append(step)
                return f"video-{step}", f"audio-{step}"

            pair, action = cache.resolve(step, negative_forward)
            actions.append(action)
            # A returned pair must always originate from the same model call.
            self.assertEqual(pair[0].split("-")[1], pair[1].split("-")[1])

        self.assertEqual(cache.hits, 24)
        self.assertEqual(cache.refreshes, 6)
        self.assertEqual(cache.negative_forwards, 26)
        self.assertEqual(len(model_calls), 26)
        self.assertFalse(cache.has_cached_pair)
        self.assertEqual(
            [step for step, action in enumerate(actions) if action == "refresh"],
            [10, 15, 20, 25, 30, 35],
        )
        self.assertEqual(
            [step for step, action in enumerate(actions) if action == "hit"],
            [
                step
                for step in range(10, 40)
                if step not in {10, 15, 20, 25, 30, 35}
            ],
        )

    def test_interval_one_matches_dense_forward_schedule(self):
        cache = CfgNegativeCache(10, 39, 1)
        cached_outputs = []
        model_calls = []

        for step in range(50):
            def negative_forward(step=step):
                model_calls.append(step)
                return ("video", step), ("audio", step)

            pair, _ = cache.resolve(step, negative_forward)
            cached_outputs.append(pair)

        dense_outputs = [(("video", step), ("audio", step)) for step in range(50)]
        self.assertEqual(cached_outputs, dense_outputs)
        self.assertEqual(model_calls, list(range(50)))
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.refreshes, 30)
        self.assertEqual(cache.negative_forwards, 50)

    def test_separate_generations_do_not_share_cached_predictions(self):
        def run_generation(label):
            # Mirrors generate(): state is constructed locally and always cleared.
            cache = CfgNegativeCache(10, 39, 5)
            calls = []
            try:
                outputs = []
                for step in (10, 11):
                    def negative_forward(step=step):
                        calls.append(step)
                        return f"{label}-video", f"{label}-audio"

                    outputs.append(cache.resolve(step, negative_forward)[0])
                return outputs, calls, cache
            finally:
                cache.clear()

        first_outputs, first_calls, first_cache = run_generation("first")
        second_outputs, second_calls, second_cache = run_generation("second")

        self.assertEqual(first_outputs[-1], ("first-video", "first-audio"))
        self.assertEqual(second_outputs[0], ("second-video", "second-audio"))
        self.assertEqual(first_calls, [10])
        self.assertEqual(second_calls, [10])
        self.assertFalse(first_cache.has_cached_pair)
        self.assertFalse(second_cache.has_cached_pair)

    def test_generation_finally_clears_cache_after_exception(self):
        cache = CfgNegativeCache(10, 39, 5)
        try:
            cache.resolve(10, lambda: ("video", "audio"))
            self.assertTrue(cache.has_cached_pair)
            raise RuntimeError("synthetic generation failure")
        except RuntimeError:
            pass
        finally:
            cache.clear()

        self.assertFalse(cache.has_cached_pair)

    def test_video_audio_refresh_is_atomic(self):
        cache = CfgNegativeCache(10, 39, 5)
        original_pair, action = cache.resolve(
            10, lambda: ("old-video", "old-audio")
        )
        self.assertEqual(action, "refresh")

        with self.assertRaisesRegex(ValueError, "incomplete"):
            cache.resolve(15, lambda: ("new-video", None))

        # A failed refresh must not publish only the new video prediction.
        pair_after_failure, action = cache.resolve(
            16,
            lambda: self.fail("step 16 should reuse the intact prior pair"),
        )
        self.assertEqual(action, "hit")
        self.assertEqual(pair_after_failure, original_pair)
        self.assertEqual(pair_after_failure, ("old-video", "old-audio"))
        self.assertEqual(cache.negative_forwards, 2)
        self.assertEqual(cache.refreshes, 1)
        self.assertEqual(cache.hits, 1)

    def test_configuration_validation(self):
        self.assertEqual(validate_cfg_cache_config(10, 39, 5), (10, 39, 5))
        with self.assertRaisesRegex(ValueError, "start_step"):
            validate_cfg_cache_config(-1, 39, 5)
        with self.assertRaisesRegex(ValueError, "end_step"):
            validate_cfg_cache_config(10, 9, 5)
        with self.assertRaisesRegex(ValueError, "refresh_interval"):
            validate_cfg_cache_config(10, 39, 0)


if __name__ == "__main__":
    unittest.main()
