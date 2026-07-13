"""Small, dependency-free helpers for per-generation CFG caching."""


def validate_cfg_cache_config(start_step, end_step, refresh_interval):
    """Validate and normalize the inclusive CFG-cache window."""
    start_step = int(start_step)
    end_step = int(end_step)
    refresh_interval = int(refresh_interval)
    if start_step < 0:
        raise ValueError("cfg_cache_start_step must be >= 0")
    if end_step < start_step:
        raise ValueError(
            "cfg_cache_end_step must be >= cfg_cache_start_step "
            "(the cache window is inclusive)"
        )
    if refresh_interval < 1:
        raise ValueError("cfg_cache_refresh_interval must be >= 1")
    return start_step, end_step, refresh_interval


class CfgNegativeCache:
    """Cache one atomic ``(video, audio)`` negative-prediction pair.

    Instances are intentionally short lived: ``OviFusionEngine.generate`` creates
    one per generation and clears it in ``finally``.  No cache state is attached
    to the engine or model.
    """

    def __init__(self, start_step, end_step, refresh_interval):
        (
            self.start_step,
            self.end_step,
            self.refresh_interval,
        ) = validate_cfg_cache_config(start_step, end_step, refresh_interval)
        self._cached_pair = None
        self.hits = 0
        self.refreshes = 0
        self.negative_forwards = 0

    @property
    def has_cached_pair(self):
        return self._cached_pair is not None

    def _inside_window(self, step):
        return self.start_step <= step <= self.end_step

    def action(self, step):
        """Return ``outside_window``, ``refresh``, or ``hit`` for ``step``."""
        step = int(step)
        if not self._inside_window(step):
            return "outside_window"
        if self._cached_pair is None:
            return "refresh"
        if (step - self.start_step) % self.refresh_interval == 0:
            return "refresh"
        return "hit"

    @staticmethod
    def _atomic_pair(candidate):
        if not isinstance(candidate, (tuple, list)) or len(candidate) != 2:
            raise ValueError(
                "negative forward must return one atomic (video, audio) pair"
            )
        video_prediction, audio_prediction = candidate
        if video_prediction is None or audio_prediction is None:
            raise ValueError(
                "negative forward returned an incomplete (video, audio) pair"
            )
        return video_prediction, audio_prediction

    def resolve(self, step, negative_forward):
        """Resolve the negative pair and return ``(pair, action)``.

        ``negative_forward`` is evaluated on every step outside the inclusive
        cache window and on refresh steps anchored at ``start_step``.  The cache
        is only replaced after both video and audio predictions validate, so a
        failed or partial refresh cannot expose a half-updated pair.
        """
        step = int(step)
        action = self.action(step)
        if action == "hit":
            self.hits += 1
            return self._cached_pair, action

        if action == "outside_window":
            # Predictions from inside the window must not survive outside it.
            self.clear()

        try:
            candidate = negative_forward()
        finally:
            # Count the actual attempted negative model forward even if it raises.
            self.negative_forwards += 1
        pair = self._atomic_pair(candidate)

        if action == "refresh":
            # One assignment makes video/audio cache replacement atomic.
            self._cached_pair = pair
            self.refreshes += 1
        return pair, action

    def metrics(self):
        return {
            "cfg_cache_hits": self.hits,
            "cfg_cache_refreshes": self.refreshes,
            "cfg_negative_forwards": self.negative_forwards,
        }

    def clear(self):
        self._cached_pair = None
