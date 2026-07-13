"""Strict dispatcher for Ovi video self-attention implementations.

This module intentionally has no torch dependency.  The dispatcher receives the
original video self-attention object at the single call site in ``fusion.py``.
Dense mode calls that object directly; sparse backends must be supplied
explicitly and are never allowed to fall back to dense attention.
"""

from collections.abc import Callable, Mapping


SUPPORTED_ATTENTION_METHODS = ("dense", "sparge", "radial", "svg")


class VideoAttentionBackendUnavailableError(NotImplementedError):
    """Raised before inference when a requested sparse backend is unavailable."""


def expected_video_self_attention_calls(
    *,
    sample_steps,
    num_blocks,
    slg_layer,
    conditional_forwards_per_step=1,
    unconditional_forwards_per_step=1,
    negative_forward_count=None,
):
    """Return the expected dispatcher calls for the standard Ovi CFG loop.

    Ovi always evaluates every block for the conditional branch.  Its
    unconditional branch skips exactly one block when ``0 < slg_layer <
    num_blocks``.
    """

    values = {
        "sample_steps": sample_steps,
        "num_blocks": num_blocks,
        "conditional_forwards_per_step": conditional_forwards_per_step,
        "unconditional_forwards_per_step": unconditional_forwards_per_step,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if negative_forward_count is not None and (
        not isinstance(negative_forward_count, int)
        or isinstance(negative_forward_count, bool)
        or negative_forward_count < 0
    ):
        raise ValueError(
            "negative_forward_count must be a non-negative integer or None, "
            f"got {negative_forward_count!r}"
        )

    unconditional_blocks = num_blocks
    if isinstance(slg_layer, int) and not isinstance(slg_layer, bool):
        if 0 < slg_layer < num_blocks:
            unconditional_blocks -= 1

    conditional_calls = (
        sample_steps * conditional_forwards_per_step * num_blocks
    )
    if negative_forward_count is None:
        negative_forward_count = (
            sample_steps * unconditional_forwards_per_step
        )
    return conditional_calls + negative_forward_count * unconditional_blocks


class VideoSelfAttentionDispatcher:
    """Dispatch only the video self-attention operation selected by config.

    Args:
        method: One of ``dense``, ``sparge``, ``radial``, or ``svg``.
        backends: Explicit sparse backend callables keyed by method name.  A
            requested sparse method without a callable fails immediately.

    A sparse backend receives the original attention object plus the same four
    positional arguments as dense attention.  Optional block/debug context is
    passed by keyword.  Backends must either produce their result or raise;
    this class never catches an error and reroutes the call to dense attention.
    """

    def __init__(self, method="dense", *, backends=None):
        normalized_method = str(method).strip().lower()
        if normalized_method not in SUPPORTED_ATTENTION_METHODS:
            expected = ", ".join(SUPPORTED_ATTENTION_METHODS)
            raise ValueError(
                f"Unsupported attention_method={normalized_method!r}; "
                f"expected one of {expected}."
            )

        if backends is None:
            backends = {}
        if not isinstance(backends, Mapping):
            raise TypeError("backends must be a mapping of method names to callables")

        unknown_backends = sorted(set(backends) - set(SUPPORTED_ATTENTION_METHODS))
        if unknown_backends:
            raise ValueError(f"Unsupported video attention backends: {unknown_backends}")

        backend = backends.get(normalized_method)
        if normalized_method != "dense" and not callable(backend):
            raise VideoAttentionBackendUnavailableError(
                f"attention_method={normalized_method!r} has no installed and "
                "implemented backend; refusing to fall back to dense attention."
            )
        if normalized_method == "dense" and backend is not None:
            raise ValueError(
                "dense attention must call the original video self-attention "
                "object directly; do not register a dense backend override"
            )

        self.method = normalized_method
        self._backend: Callable | None = backend
        self.reset_metrics()

    def reset_metrics(self):
        self._calls_by_method = {
            method: 0 for method in SUPPORTED_ATTENTION_METHODS
        }
        self._errors_by_method = {
            method: 0 for method in SUPPORTED_ATTENTION_METHODS
        }

    def metrics(self):
        """Return JSON-serializable dispatcher state for run metrics."""

        return {
            "configured_method": self.method,
            "active_method": self.method,
            "backend_ready": self.method == "dense" or self._backend is not None,
            "calls_total": sum(self._calls_by_method.values()),
            "calls_by_method": dict(self._calls_by_method),
            "errors_by_method": dict(self._errors_by_method),
            "fallback_allowed": False,
            "fallback_used": False,
            "fallback_count": 0,
            "fallback_reason": None,
        }

    def __call__(
        self,
        original_attention,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        *,
        block_index=None,
        debug_context=None,
    ):
        """Run the configured method without changing any non-video call site."""

        self._calls_by_method[self.method] += 1
        try:
            if self.method == "dense":
                return original_attention(x, seq_lens, grid_sizes, freqs)

            return self._backend(
                original_attention,
                x,
                seq_lens,
                grid_sizes,
                freqs,
                block_index=block_index,
                debug_context=debug_context,
            )
        except Exception:
            self._errors_by_method[self.method] += 1
            raise
