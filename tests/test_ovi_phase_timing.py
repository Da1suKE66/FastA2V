import unittest

from ovi.phase_timing import GenerationPhaseTimer


class SequenceClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class GenerationPhaseTimerTests(unittest.TestCase):
    def test_records_contiguous_phase_breakdown_and_total(self):
        timer = GenerationPhaseTimer(
            clock=SequenceClock([100.0, 102.0, 112.0, 115.0, 120.0])
        )

        timer.transition("denoise")
        timer.transition("audio_decode")
        timer.transition("video_decode")
        timer.finish()

        self.assertEqual(
            timer.metrics(),
            {
                "pre_denoise_seconds": 2.0,
                "denoise_seconds": 10.0,
                "audio_decode_seconds": 3.0,
                "video_decode_seconds": 5.0,
                "total_generation_seconds": 20.0,
            },
        )

    def test_failure_finishes_active_phase_without_fabricating_later_phases(self):
        timer = GenerationPhaseTimer(
            clock=SequenceClock([10.0, 11.0, 14.5])
        )
        timer.transition("denoise")

        timer.finish()

        self.assertEqual(
            timer.metrics(),
            {
                "pre_denoise_seconds": 1.0,
                "denoise_seconds": 3.5,
                "audio_decode_seconds": None,
                "video_decode_seconds": None,
                "total_generation_seconds": 4.5,
            },
        )
        timer.finish()

    def test_rejects_skipped_or_reordered_transitions(self):
        timer = GenerationPhaseTimer(clock=SequenceClock([0.0]))

        with self.assertRaisesRegex(ValueError, "expected next phase 'denoise'"):
            timer.transition("audio_decode")


if __name__ == "__main__":
    unittest.main()
