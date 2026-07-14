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


_PROMPT_SET_CONTRACTS = MappingProxyType(
    {
        "smoke": MappingProxyType(
            {
                "prompts_sha256": (
                    "1e7f242591c86e24334da252cb9ca5ee1e448cfc6e5990bbcd099b0767e1e42e"
                ),
                "prompt_count": 1,
                "each_example_n_times": 1,
            }
        ),
        "formal8": MappingProxyType(
            {
                "prompts_sha256": (
                    "d98397111b1ab060a61d588f4ca388c5c929430a59ac6ab49b7c2e247bb6be91"
                ),
                "prompt_count": 8,
                "each_example_n_times": 3,
            }
        ),
    }
)


def _base_protocol(prompt_set):
    try:
        prompt_contract = _PROMPT_SET_CONTRACTS[prompt_set]
    except KeyError as exc:
        raise ValueError(f"unknown audited prompt set {prompt_set!r}") from exc
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
        **prompt_contract,
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
    prompt_set,
    attention_method="dense",
    use_cfg_cache=False,
    use_block_cache=False,
    benchmark_eligible=False,
    debug_forward=False,
    debug_forward_step=0,
    sparge_topk=None,
    radial_profile=None,
    radial_decay_factor=None,
):
    base_protocol = _base_protocol(prompt_set)
    protocol = {
        **base_protocol,
        "run_kind": run_kind,
        "sample_steps": sample_steps,
        "warmup_runs": warmup_runs,
        "measurement_runs": measurement_runs,
        "expected_warmup_records": warmup_runs,
        "expected_measurement_records": (
            measurement_runs
            * base_protocol["prompt_count"]
            * base_protocol["each_example_n_times"]
        ),
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
    elif attention_method == "radial":
        protocol.update(
            {
                "radial_profile": radial_profile,
                "radial_decay_factor": radial_decay_factor,
                "radial_block_size": 128,
                "radial_model_type": "wan",
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
            prompt_set="formal8",
            benchmark_eligible=True,
        ),
        _run_protocol(
            "diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            prompt_set="smoke",
            debug_forward=True,
        ),
        _run_protocol(
            "official_reference",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            prompt_set="smoke",
        ),
        _run_protocol(
            "cfg_cache_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            use_cfg_cache=True,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "cfg_cache_diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            prompt_set="smoke",
            use_cfg_cache=True,
            debug_forward=True,
            debug_forward_step=11,
        ),
        _run_protocol(
            "block_cache_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            use_block_cache=True,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "block_cache_diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            prompt_set="smoke",
            use_block_cache=True,
            debug_forward=True,
            debug_forward_step=1,
        ),
        _run_protocol(
            "sparge_baseline",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="sparge",
            sparge_topk=0.50,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "sparge_diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            prompt_set="smoke",
            attention_method="sparge",
            sparge_topk=0.50,
            debug_forward=True,
        ),
        _run_protocol(
            "sparge_topk75_baseline",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="sparge",
            sparge_topk=0.75,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "sparge_topk75_diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            prompt_set="smoke",
            attention_method="sparge",
            sparge_topk=0.75,
            debug_forward=True,
        ),
        _run_protocol(
            "radial_conservative_baseline",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="radial",
            radial_profile="conservative",
            radial_decay_factor=4.0,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "radial_conservative_diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            prompt_set="smoke",
            attention_method="radial",
            radial_profile="conservative",
            radial_decay_factor=4.0,
            debug_forward=True,
        ),
        _run_protocol(
            "radial_aggressive_baseline",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="radial",
            radial_profile="aggressive",
            radial_decay_factor=1.0,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "radial_aggressive_diagnostic_smoke",
            sample_steps=20,
            warmup_runs=0,
            measurement_runs=1,
            prompt_set="smoke",
            attention_method="radial",
            radial_profile="aggressive",
            radial_decay_factor=1.0,
            debug_forward=True,
        ),
        _run_protocol(
            "sparge_topk50_cfg_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="sparge",
            sparge_topk=0.50,
            use_cfg_cache=True,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "sparge_topk75_cfg_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="sparge",
            sparge_topk=0.75,
            use_cfg_cache=True,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "radial_conservative_cfg_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="radial",
            radial_profile="conservative",
            radial_decay_factor=4.0,
            use_cfg_cache=True,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "radial_aggressive_cfg_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="radial",
            radial_profile="aggressive",
            radial_decay_factor=1.0,
            use_cfg_cache=True,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "sparge_topk50_block_cache_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="sparge",
            sparge_topk=0.50,
            use_block_cache=True,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "sparge_topk75_block_cache_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="sparge",
            sparge_topk=0.75,
            use_block_cache=True,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "radial_conservative_block_cache_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="radial",
            radial_profile="conservative",
            radial_decay_factor=4.0,
            use_block_cache=True,
            benchmark_eligible=True,
        ),
        _run_protocol(
            "radial_aggressive_block_cache_benchmark",
            sample_steps=50,
            warmup_runs=1,
            measurement_runs=3,
            prompt_set="formal8",
            attention_method="radial",
            radial_profile="aggressive",
            radial_decay_factor=1.0,
            use_block_cache=True,
            benchmark_eligible=True,
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
