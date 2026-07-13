"""CPU-only immutable protocols for audited Ovi evaluation runs.

The protocol registry is deliberately independent of Torch, OmegaConf, and the
CUDA runtime.  A completed run is only comparable when every field recorded in
``environment.json`` matches the fixed values for its ``run_kind``.

``official_reference`` is the one media-only identity: the pinned, unmodified
upstream does not emit FastA2V environment/timing evidence.  Its entry freezes
the checked-in YAML for decoded-media equivalence checks and is never a
benchmark candidate.
"""

from collections.abc import Mapping, Sequence
import hashlib
import json
from types import MappingProxyType


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _base_protocol():
    return {
        "model_name": "720x720_5s",
        "mode": "t2v",
        "video_frame_height_width": [720, 720],
        "solver_name": "unipc",
        "shift": 5.0,
        "seed": 103,
        "video_guidance_scale": 4.0,
        "audio_guidance_scale": 3.0,
        "slg_layer": 11,
        "prompts_sha256": (
            "1e7f242591c86e24334da252cb9ca5ee1e448cfc6e5990bbcd099b0767e1e42e"
        ),
        "prompt_count": 1,
        "each_example_n_times": 1,
        "video_negative_prompt": "jitter, bad hands, blur, distortion",
        "audio_negative_prompt": "robotic, muffled, echo, distorted",
        "fp8": False,
        "qint8": False,
        "cpu_offload": False,
        "sp_size": 1,
        "gpu_process_monitor_interval_seconds": 5.0,
        "cfg_cache_start_step": 10,
        "cfg_cache_end_step": 39,
        "cfg_cache_window_inclusive": True,
        "cfg_cache_refresh_interval": 5,
        "block_cache_start_block": 10,
        "block_cache_end_block": 19,
        "block_cache_window_inclusive": True,
        "block_cache_policy": "fixed",
        "block_cache_cosine_threshold": 0.95,
        "block_cache_max_consecutive_reuses": 1,
    }


def _run_protocol(
    run_kind,
    *,
    sample_steps,
    warmup_runs,
    measurement_runs,
    attention_method="dense",
    use_cfg_cache=False,
    use_block_cache=False,
    benchmark_eligible=False,
    debug_forward=False,
    debug_forward_step=0,
    sparge_topk=None,
):
    protocol = {
        **_base_protocol(),
        "run_kind": run_kind,
        "sample_steps": sample_steps,
        "warmup_runs": warmup_runs,
        "measurement_runs": measurement_runs,
        "expected_warmup_records": warmup_runs,
        "expected_measurement_records": measurement_runs,
        "attention_method": attention_method,
        "use_cfg_cache": use_cfg_cache,
        "use_block_cache": use_block_cache,
        "benchmark_eligible": benchmark_eligible,
        "debug_forward": debug_forward,
        "debug_forward_step": debug_forward_step,
    }
    if attention_method == "sparge":
        protocol.update(
            {
                "sparge_topk": sparge_topk,
                "sparge_pvthreshd": 50.0,
                "sparge_smooth_k": True,
            }
        )
    return protocol


def _build_protocols():
    protocols = (
        _run_protocol(
            "dense_baseline",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            debug_forward=True,
        ),
        _run_protocol(
            "official_reference",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
        ),
        _run_protocol(
            "cfg_cache_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            use_cfg_cache=True,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "cfg_cache_diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            use_cfg_cache=True,
            debug_forward=True,
            debug_forward_step=11,
        ),
        _run_protocol(
            "block_cache_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            use_block_cache=True,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "block_cache_diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            use_block_cache=True,
            debug_forward=True,
            debug_forward_step=1,
        ),
        _run_protocol(
            "sparge_baseline",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            attention_method="sparge",
            sparge_topk=0.50,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "sparge_diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            attention_method="sparge",
            sparge_topk=0.50,
            debug_forward=True,
        ),
        _run_protocol(
            "sparge_topk75_baseline",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            attention_method="sparge",
            sparge_topk=0.75,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "sparge_topk75_diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            attention_method="sparge",
            sparge_topk=0.75,
            debug_forward=True,
        ),
    )
    return MappingProxyType(
        {protocol["run_kind"]: _freeze(protocol) for protocol in protocols}
    )


RUN_KIND_PROTOCOLS = _build_protocols()
AUDITED_RUN_KINDS = frozenset(RUN_KIND_PROTOCOLS)


def prompt_sequence_sha256(prompts):
    """Hash the exact ordered prompt sequence with canonical UTF-8 JSON."""

    payload = json.dumps(
        list(prompts),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _same_fixed_value(actual, expected):
    if isinstance(expected, tuple):
        if not isinstance(actual, Sequence) or isinstance(
            actual, (str, bytes, bytearray)
        ):
            return False
        return len(actual) == len(expected) and all(
            _same_fixed_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def run_protocol_errors(environment):
    """Return all deviations from the immutable protocol for ``environment``."""

    if not isinstance(environment, Mapping):
        return ["Ovi run environment must be a mapping"]

    run_kind = environment.get("run_kind")
    if not isinstance(run_kind, str):
        return [
            f"run_kind {run_kind!r} is not an audited immutable Ovi "
            "evaluation protocol"
        ]
    expected_protocol = RUN_KIND_PROTOCOLS.get(run_kind)
    if expected_protocol is None:
        return [
            f"run_kind {run_kind!r} is not an audited immutable Ovi "
            "evaluation protocol"
        ]

    errors = []
    for field, expected in expected_protocol.items():
        if field not in environment:
            errors.append(f"{run_kind} protocol is missing required field {field}")
        elif not _same_fixed_value(environment[field], expected):
            errors.append(
                f"{run_kind} protocol {field}={environment[field]!r} "
                f"!= fixed value {expected!r}"
            )
    return errors


def validate_run_protocol(environment, errors):
    """Append immutable run-protocol deviations to an existing error list."""

    errors.extend(run_protocol_errors(environment))


def materialize_run_protocol(run_kind):
    """Return a mutable JSON-shaped copy, primarily for CPU tests and tooling."""

    protocol = RUN_KIND_PROTOCOLS[run_kind]
    return {
        field: list(value) if isinstance(value, tuple) else value
        for field, value in protocol.items()
    }
