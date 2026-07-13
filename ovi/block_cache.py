"""Per-generation cache for one contiguous Ovi fusion-block window.

The cache intentionally owns no model or engine state.  A fresh instance is
created by :meth:`OviFusionEngine.generate`, passed into the two CFG branches,
and cleared in ``finally``.  Conditional and unconditional payloads live in
different branch records so a cache hit can never cross the CFG boundary.
"""

from dataclasses import dataclass, field
import math


SUPPORTED_BLOCK_CACHE_POLICIES = ("fixed", "cosine")
SUPPORTED_BLOCK_CACHE_BRANCHES = ("conditional", "unconditional")


def validate_block_cache_config(
    start_block,
    end_block,
    policy="fixed",
    cosine_threshold=0.95,
    max_consecutive_reuses=1,
    *,
    num_blocks=None,
):
    """Validate and normalize the fusion-block cache configuration."""
    start_block = int(start_block)
    end_block = int(end_block)
    policy = str(policy).lower()
    cosine_threshold = float(cosine_threshold)
    max_consecutive_reuses = int(max_consecutive_reuses)

    if start_block < 0:
        raise ValueError("block_cache_start_block must be >= 0")
    if end_block < start_block:
        raise ValueError(
            "block_cache_end_block must be >= block_cache_start_block "
            "(the block window is inclusive)"
        )
    if num_blocks is not None and end_block >= int(num_blocks):
        raise ValueError(
            f"block_cache_end_block={end_block} is outside a model with "
            f"{int(num_blocks)} fusion blocks"
        )
    if policy not in SUPPORTED_BLOCK_CACHE_POLICIES:
        raise ValueError(
            f"Unsupported block_cache_policy={policy!r}; expected one of "
            f"{', '.join(SUPPORTED_BLOCK_CACHE_POLICIES)}"
        )
    if not math.isfinite(cosine_threshold) or not 0.0 <= cosine_threshold <= 1.0:
        raise ValueError("block_cache_cosine_threshold must be in [0, 1]")
    # The first implementation deliberately supports only the audited
    # compute -> reuse -> compute schedule.  Accepting larger values would make
    # the configured safety bound look effective while silently changing it.
    if max_consecutive_reuses != 1:
        raise ValueError("block_cache_max_consecutive_reuses must be exactly 1")

    return (
        start_block,
        end_block,
        policy,
        cosine_threshold,
        max_consecutive_reuses,
    )


def _nonnegative_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _fixed_branch_schedule(forward_steps, max_consecutive_reuses):
    has_cached_pair = False
    last_seen_step = None
    consecutive_reuses = 0
    hits = 0
    refreshes = 0
    refresh_reasons = {}

    for step in forward_steps:
        if not has_cached_pair:
            action = "refresh"
            reason = "empty"
        elif step != last_seen_step + 1:
            action = "refresh"
            reason = "step_gap"
        elif consecutive_reuses >= max_consecutive_reuses:
            action = "refresh"
            reason = "max_consecutive_reuses"
        else:
            action = "hit"
            reason = None

        if action == "hit":
            hits += 1
            consecutive_reuses += 1
        else:
            has_cached_pair = True
            refreshes += 1
            consecutive_reuses = 0
            refresh_reasons[reason] = refresh_reasons.get(reason, 0) + 1
        last_seen_step = step

    return {
        "forward_count": len(forward_steps),
        "hits": hits,
        "refreshes": refreshes,
        "refresh_reasons": refresh_reasons,
    }


def expected_fixed_block_cache_metrics(
    *,
    sample_steps,
    use_cfg_cache,
    cfg_cache_start_step,
    cfg_cache_end_step,
    cfg_cache_refresh_interval,
    block_cache_start_block,
    block_cache_end_block,
    block_cache_max_consecutive_reuses,
    slg_layer,
):
    """Return the exact fixed-policy schedule for one Ovi generation."""
    sample_steps = _nonnegative_int("sample_steps", sample_steps)
    cfg_cache_start_step = _nonnegative_int(
        "cfg_cache_start_step", cfg_cache_start_step
    )
    cfg_cache_end_step = _nonnegative_int(
        "cfg_cache_end_step", cfg_cache_end_step
    )
    cfg_cache_refresh_interval = _nonnegative_int(
        "cfg_cache_refresh_interval", cfg_cache_refresh_interval
    )
    if cfg_cache_end_step < cfg_cache_start_step:
        raise ValueError(
            "cfg_cache_end_step must be >= cfg_cache_start_step"
        )
    if cfg_cache_refresh_interval < 1:
        raise ValueError("cfg_cache_refresh_interval must be >= 1")
    block_cache_start_block = _nonnegative_int(
        "block_cache_start_block", block_cache_start_block
    )
    block_cache_end_block = _nonnegative_int(
        "block_cache_end_block", block_cache_end_block
    )
    if block_cache_end_block < block_cache_start_block:
        raise ValueError(
            "block_cache_end_block must be >= block_cache_start_block"
        )
    block_cache_max_consecutive_reuses = _nonnegative_int(
        "block_cache_max_consecutive_reuses",
        block_cache_max_consecutive_reuses,
    )
    if block_cache_max_consecutive_reuses != 1:
        raise ValueError(
            "fixed block-cache verification requires exactly one consecutive reuse"
        )
    if not isinstance(use_cfg_cache, bool):
        raise ValueError("use_cfg_cache must be a bool")
    if not isinstance(slg_layer, int) or isinstance(slg_layer, bool):
        raise ValueError("slg_layer must be an integer")

    conditional_steps = list(range(sample_steps))
    if use_cfg_cache:
        unconditional_steps = [
            step
            for step in range(sample_steps)
            if (
                step < cfg_cache_start_step
                or step > cfg_cache_end_step
                or (
                    (step - cfg_cache_start_step)
                    % cfg_cache_refresh_interval
                    == 0
                )
            )
        ]
    else:
        unconditional_steps = list(range(sample_steps))

    branch_metrics = {
        "conditional": _fixed_branch_schedule(
            conditional_steps, block_cache_max_consecutive_reuses
        ),
        "unconditional": _fixed_branch_schedule(
            unconditional_steps, block_cache_max_consecutive_reuses
        ),
    }
    window_size = block_cache_end_block - block_cache_start_block + 1
    branch_metrics["conditional"]["saved_video_self_attention_calls"] = (
        branch_metrics["conditional"]["hits"] * window_size
    )
    unconditional_window_calls = window_size - int(
        slg_layer > 0
        and block_cache_start_block <= slg_layer <= block_cache_end_block
    )
    branch_metrics["unconditional"][
        "saved_video_self_attention_calls"
    ] = branch_metrics["unconditional"]["hits"] * unconditional_window_calls

    return {
        "block_cache_hits": sum(
            branch["hits"] for branch in branch_metrics.values()
        ),
        "block_cache_refreshes": sum(
            branch["refreshes"] for branch in branch_metrics.values()
        ),
        "block_cache_saved_video_self_attention_calls": sum(
            branch["saved_video_self_attention_calls"]
            for branch in branch_metrics.values()
        ),
        "block_cache_branch_metrics": branch_metrics,
    }


def fixed_block_cache_metric_errors(metrics):
    """Validate recorded fixed-policy metrics against the exact CFG schedule."""
    if not isinstance(metrics, dict):
        return ["fixed block-cache metrics must be a JSON object"]
    try:
        expected = expected_fixed_block_cache_metrics(
            sample_steps=metrics["sample_steps"],
            use_cfg_cache=metrics["use_cfg_cache"],
            cfg_cache_start_step=metrics["cfg_cache_start_step"],
            cfg_cache_end_step=metrics["cfg_cache_end_step"],
            cfg_cache_refresh_interval=metrics["cfg_cache_refresh_interval"],
            block_cache_start_block=metrics["block_cache_start_block"],
            block_cache_end_block=metrics["block_cache_end_block"],
            block_cache_max_consecutive_reuses=metrics[
                "block_cache_max_consecutive_reuses"
            ],
            slg_layer=metrics["slg_layer"],
        )
    except (KeyError, TypeError, ValueError) as error:
        return [f"cannot derive fixed block-cache schedule: {error}"]

    errors = []
    for field_name in (
        "block_cache_hits",
        "block_cache_refreshes",
        "block_cache_saved_video_self_attention_calls",
    ):
        if metrics.get(field_name) != expected[field_name]:
            errors.append(
                f"{field_name}={metrics.get(field_name)} != fixed schedule "
                f"{expected[field_name]}"
            )

    expected_negative_forwards = expected["block_cache_branch_metrics"][
        "unconditional"
    ]["forward_count"]
    if metrics.get("cfg_negative_forwards") != expected_negative_forwards:
        errors.append(
            "cfg_negative_forwards="
            f"{metrics.get('cfg_negative_forwards')} != configured CFG "
            f"schedule {expected_negative_forwards}"
        )

    actual_branches = metrics.get("block_cache_branch_metrics")
    if not isinstance(actual_branches, dict):
        errors.append("fixed block-cache branch metrics must be a JSON object")
        return errors
    for branch_name in SUPPORTED_BLOCK_CACHE_BRANCHES:
        actual = actual_branches.get(branch_name)
        branch_expected = expected["block_cache_branch_metrics"][branch_name]
        if not isinstance(actual, dict):
            errors.append(
                f"fixed block-cache branch {branch_name} must be a JSON object"
            )
            continue
        for field_name in (
            "hits",
            "refreshes",
            "saved_video_self_attention_calls",
            "refresh_reasons",
        ):
            if actual.get(field_name) != branch_expected[field_name]:
                errors.append(
                    f"fixed block-cache {branch_name}.{field_name}="
                    f"{actual.get(field_name)} != schedule "
                    f"{branch_expected[field_name]}"
                )
        actual_forward_count = (
            actual.get("hits", 0) + actual.get("refreshes", 0)
            if isinstance(actual.get("hits"), int)
            and not isinstance(actual.get("hits"), bool)
            and isinstance(actual.get("refreshes"), int)
            and not isinstance(actual.get("refreshes"), bool)
            else None
        )
        if actual_forward_count != branch_expected["forward_count"]:
            errors.append(
                f"fixed block-cache {branch_name} forward count="
                f"{actual_forward_count} != schedule "
                f"{branch_expected['forward_count']}"
            )
    return errors


def _atomic_pair(candidate, label):
    if not isinstance(candidate, (tuple, list)) or len(candidate) != 2:
        raise ValueError(f"{label} must be one atomic (video, audio) pair")
    video, audio = candidate
    if video is None or audio is None:
        raise ValueError(f"{label} contains an incomplete (video, audio) pair")
    return video, audio


def _tensor_metadata(value):
    """Return only the compatibility fields that must match before reuse."""
    try:
        shape = tuple(int(dimension) for dimension in value.shape)
        dtype = str(value.dtype)
        device = str(value.device)
    except AttributeError as error:
        raise TypeError(
            "block-cache inputs must expose shape, dtype, and device"
        ) from error
    return {"shape": shape, "dtype": dtype, "device": device}


def _pair_metadata(pair):
    video, audio = pair
    return {
        "video": _tensor_metadata(video),
        "audio": _tensor_metadata(audio),
    }


def _metadata_mismatch_reason(cached, current):
    for stream in ("video", "audio"):
        for field_name in ("shape", "dtype", "device"):
            if cached[stream][field_name] != current[stream][field_name]:
                return f"{stream}_{field_name}_mismatch"
    return None


def _detached_pair(pair):
    # ``detach`` preserves the exact complete tensors without allocating a
    # second copy.  In inference mode the fusion blocks do not mutate these
    # tensors in-place; later blocks bind new tensors instead.
    return tuple(
        value.detach() if callable(getattr(value, "detach", None)) else value
        for value in pair
    )


def _pool_hidden_for_cosine(hidden):
    """Pool ``[batch, sequence, ...]`` hidden state over sequence dim 1."""
    shape = getattr(hidden, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError(
            "cosine block-cache inputs must have at least two dimensions "
            "so sequence dim=1 can be pooled"
        )
    # Cache only this detached pooled representation.  For Ovi [B, S, D]
    # tensors this avoids retaining another full video hidden state per branch.
    return hidden.mean(dim=1).detach()


def _torch_cosine_similarity(left_pooled, right_pooled):
    import torch
    import torch.nn.functional as F

    if not isinstance(left_pooled, torch.Tensor) or not isinstance(
        right_pooled, torch.Tensor
    ):
        raise TypeError(
            "cosine block-cache policy requires pooled torch.Tensor inputs"
        )
    # The optional cosine policy necessarily materializes one scalar decision
    # per stream.  The fixed policy has no such synchronization.
    similarity = F.cosine_similarity(
        left_pooled.float().reshape(-1),
        right_pooled.float().reshape(-1),
        dim=0,
        eps=1e-8,
    )
    return float(similarity.item())


@dataclass
class _BranchRecord:
    cached_pooled_input_pair: object = None
    cached_output_pair: object = None
    input_metadata: object = None
    output_metadata: object = None
    signature: object = None
    source_step: object = None
    last_seen_step: object = None
    consecutive_reuses: int = 0
    hits: int = 0
    refreshes: int = 0
    saved_video_self_attention_calls: int = 0
    refresh_reasons: dict = field(default_factory=dict)
    last_action: object = None
    last_refresh_reason: object = None
    last_video_cosine: object = None
    last_audio_cosine: object = None
    last_min_cosine: object = None

    def clear_payload(self):
        self.cached_pooled_input_pair = None
        self.cached_output_pair = None
        self.input_metadata = None
        self.output_metadata = None
        self.signature = None
        self.source_step = None
        self.last_seen_step = None
        self.consecutive_reuses = 0


class FusionBlockCache:
    """Cache complete video/audio outputs for one fusion-block window.

    ``resolve`` is called exactly once when a model forward reaches
    ``start_block``.  A refresh computes every non-SLG block through
    ``end_block`` and atomically publishes the resulting video/audio pair.  A
    hit skips that whole window and returns the branch-local pair.
    """

    def __init__(
        self,
        start_block=10,
        end_block=19,
        policy="fixed",
        cosine_threshold=0.95,
        max_consecutive_reuses=1,
        *,
        num_blocks=None,
        cosine_fn=None,
        pool_fn=None,
    ):
        (
            self.start_block,
            self.end_block,
            self.policy,
            self.cosine_threshold,
            self.max_consecutive_reuses,
        ) = validate_block_cache_config(
            start_block,
            end_block,
            policy,
            cosine_threshold,
            max_consecutive_reuses,
            num_blocks=num_blocks,
        )
        # Pre-create distinct records.  No payload object or reuse counter is
        # shared between positive and negative CFG branches.
        self._branches = {
            branch: _BranchRecord()
            for branch in SUPPORTED_BLOCK_CACHE_BRANCHES
        }
        # Dependency injection keeps the state-machine tests CPU-only and free
        # of the heavyweight Ovi runtime.  Production uses the Torch function.
        self._cosine_fn = cosine_fn or _torch_cosine_similarity
        self._pool_fn = pool_fn or _pool_hidden_for_cosine

    @property
    def window_size(self):
        return self.end_block - self.start_block + 1

    def _branch_record(self, branch):
        branch = str(branch)
        if branch not in self._branches:
            raise ValueError(
                f"Unsupported block-cache branch={branch!r}; expected one of "
                f"{', '.join(SUPPORTED_BLOCK_CACHE_BRANCHES)}"
            )
        return branch, self._branches[branch]

    def has_cached_pair(self, branch=None):
        if branch is None:
            return any(
                record.cached_output_pair is not None
                for record in self._branches.values()
            )
        _, record = self._branch_record(branch)
        return record.cached_output_pair is not None

    def _decision(self, record, step, input_pair, signature):
        if record.cached_output_pair is None:
            return "refresh", "empty"
        if step != record.last_seen_step + 1:
            return "refresh", "step_gap"

        current_metadata = _pair_metadata(input_pair)
        mismatch = _metadata_mismatch_reason(
            record.input_metadata, current_metadata
        )
        if mismatch is not None:
            return "refresh", mismatch
        if signature != record.signature:
            return "refresh", "slg_signature_mismatch"
        cached_output_metadata = _pair_metadata(record.cached_output_pair)
        mismatch = _metadata_mismatch_reason(
            record.output_metadata, cached_output_metadata
        )
        if mismatch is not None:
            return "refresh", f"cached_output_{mismatch}"
        mismatch = _metadata_mismatch_reason(
            current_metadata, record.output_metadata
        )
        if mismatch is not None:
            return "refresh", f"output_{mismatch}"
        if record.consecutive_reuses >= self.max_consecutive_reuses:
            return "refresh", "max_consecutive_reuses"

        if self.policy == "cosine":
            current_pooled_pair = tuple(
                self._pool_fn(value) for value in input_pair
            )
            video_cosine = self._cosine_fn(
                current_pooled_pair[0], record.cached_pooled_input_pair[0]
            )
            audio_cosine = self._cosine_fn(
                current_pooled_pair[1], record.cached_pooled_input_pair[1]
            )
            record.last_video_cosine = video_cosine
            record.last_audio_cosine = audio_cosine
            if not (
                math.isfinite(video_cosine) and math.isfinite(audio_cosine)
            ):
                record.last_min_cosine = None
                return "refresh", "cosine_nonfinite"
            min_cosine = min(video_cosine, audio_cosine)
            record.last_min_cosine = min_cosine
            if min_cosine < self.cosine_threshold:
                return "refresh", "cosine_below_threshold"

        return "hit", None

    def resolve(
        self,
        *,
        step,
        branch,
        input_pair,
        slg_signature,
        skipped_blocks,
        compute_window,
    ):
        """Return ``(complete_pair, action)`` for one branch/model forward.

        Compatibility checks happen before a hit: denoising step continuity,
        video/audio shape, dtype, device, and the full SLG skip signature.  The
        callback is evaluated only for refreshes, and its output is published
        only after both streams validate.
        """
        if isinstance(step, bool):
            raise TypeError("block-cache step must be an integer, not bool")
        step = int(step)
        branch, record = self._branch_record(branch)
        input_pair = _atomic_pair(input_pair, "block-cache input")
        skipped_blocks = tuple(sorted({int(item) for item in skipped_blocks}))
        signature = (slg_signature, skipped_blocks)

        action, refresh_reason = self._decision(
            record, step, input_pair, signature
        )
        if action == "hit":
            record.hits += 1
            record.consecutive_reuses += 1
            record.last_seen_step = step
            record.last_action = "hit"
            record.last_refresh_reason = None
            skipped_inside_window = sum(
                self.start_block <= block <= self.end_block
                for block in skipped_blocks
            )
            record.saved_video_self_attention_calls += (
                self.window_size - skipped_inside_window
            )
            return record.cached_output_pair, action

        candidate = compute_window()
        output_pair = _atomic_pair(candidate, "block-cache window output")
        # All fields are replaced together only after the complete pair has
        # validated; a partial/failed refresh leaves the old payload intact.
        new_cached_pooled_input_pair = (
            tuple(self._pool_fn(value) for value in input_pair)
            if self.policy == "cosine"
            else None
        )
        new_input_metadata = _pair_metadata(input_pair)
        new_cached_output_pair = _detached_pair(output_pair)
        # Validate the exact detached pair that will be published, rather than
        # assuming a tensor-like ``detach`` implementation preserves metadata.
        new_output_metadata = _pair_metadata(new_cached_output_pair)
        mismatch = _metadata_mismatch_reason(
            new_input_metadata, new_output_metadata
        )
        if mismatch is not None:
            raise ValueError(
                "block-cache window output metadata must match its input; "
                f"mismatch={mismatch} input={new_input_metadata} "
                f"output={new_output_metadata}"
            )
        record.cached_pooled_input_pair = new_cached_pooled_input_pair
        record.cached_output_pair = new_cached_output_pair
        record.input_metadata = new_input_metadata
        record.output_metadata = new_output_metadata
        record.signature = signature
        record.source_step = step
        record.last_seen_step = step
        record.consecutive_reuses = 0
        record.refreshes += 1
        record.last_action = "refresh"
        record.last_refresh_reason = refresh_reason
        record.refresh_reasons[refresh_reason] = (
            record.refresh_reasons.get(refresh_reason, 0) + 1
        )
        return output_pair, action

    def metrics(self):
        branch_metrics = {}
        for branch, record in self._branches.items():
            branch_metrics[branch] = {
                "hits": record.hits,
                "refreshes": record.refreshes,
                "saved_video_self_attention_calls": (
                    record.saved_video_self_attention_calls
                ),
                "refresh_reasons": dict(record.refresh_reasons),
                "last_action": record.last_action,
                "last_refresh_reason": record.last_refresh_reason,
                "last_video_cosine": record.last_video_cosine,
                "last_audio_cosine": record.last_audio_cosine,
                "last_min_cosine": record.last_min_cosine,
            }
        return {
            "block_cache_hits": sum(
                item["hits"] for item in branch_metrics.values()
            ),
            "block_cache_refreshes": sum(
                item["refreshes"] for item in branch_metrics.values()
            ),
            "block_cache_saved_video_self_attention_calls": sum(
                item["saved_video_self_attention_calls"]
                for item in branch_metrics.values()
            ),
            "block_cache_branch_metrics": branch_metrics,
        }

    def clear(self):
        for record in self._branches.values():
            record.clear_payload()
