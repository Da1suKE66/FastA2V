"""Low-overhead wall-clock timing for the coarse Ovi generation phases."""

import time


PHASES = (
    "pre_denoise",
    "denoise",
    "audio_decode",
    "video_decode",
)


class GenerationPhaseTimer:
    """Time contiguous phases without instrumenting denoising steps.

    The caller owns accelerator synchronization at the coarse boundaries. This
    helper only uses a monotonic CPU clock and never imports or synchronizes
    CUDA, so it adds no per-step device overhead.
    """

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._started_at = self._clock()
        self._phase_started_at = self._started_at
        self._phase_index = 0
        self._durations = {phase: None for phase in PHASES}
        self._total_seconds = None

    def transition(self, next_phase):
        """Complete the active phase and start its immediate successor."""
        if self._total_seconds is not None:
            raise RuntimeError("generation phase timer is already finished")
        if self._phase_index >= len(PHASES):
            raise RuntimeError("all generation phases are already complete")

        expected_next_index = self._phase_index + 1
        expected_next_phase = (
            PHASES[expected_next_index]
            if expected_next_index < len(PHASES)
            else None
        )
        if expected_next_phase != next_phase:
            raise ValueError(
                f"expected next phase {expected_next_phase!r}, "
                f"got {next_phase!r}"
            )

        now = self._clock()
        active_phase = PHASES[self._phase_index]
        self._durations[active_phase] = now - self._phase_started_at
        self._phase_index = expected_next_index
        self._phase_started_at = now

    def finish(self):
        """Finish the active phase and total timing; safe to call twice."""
        if self._total_seconds is not None:
            return

        now = self._clock()
        if self._phase_index < len(PHASES):
            active_phase = PHASES[self._phase_index]
            self._durations[active_phase] = now - self._phase_started_at
        self._total_seconds = now - self._started_at

    def metrics(self):
        """Return JSON-safe metrics, with ``None`` for unstarted phases."""
        return {
            **{
                f"{phase}_seconds": self._durations[phase]
                for phase in PHASES
            },
            "total_generation_seconds": self._total_seconds,
        }
