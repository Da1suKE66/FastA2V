"""Pure-stdlib contract for the Ovi CFG-cache ablation v2 protocol.

The protocol data is intentionally independent of Torch, OmegaConf, and CUDA so
that matrix/config validation can run before a GPU job is launched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


PROTOCOL_ID = "ovi_cfg_cache_ablation_v2"
SAMPLE_STEPS = 30
VIDEO_BLOCKS = 30
UNCONDITIONAL_VIDEO_BLOCKS = 29
DENSE_VIDEO_SELF_ATTENTION_CALLS = 1770
ALLOWED_STAGES = ("0", "1", "2")

MATRIX_FIELDS = (
    "stage",
    "config_id",
    "use_cfg_cache",
    "start_step",
    "end_step",
    "refresh_interval",
    "eligible_steps",
    "refreshes",
    "cache_hits",
    "negative_forwards",
    "expected_video_self_attention_calls",
    "max_cache_age",
    "role",
    "advance_rule",
)

EXPECTED_CONFIG_IDS = (
    "dense",
    "late_12_29_r1_null",
    "current_9_26_r5_anchor",
    "new_12_29_r5_repeat",
    "bin_00_04_r5",
    "bin_05_09_r5",
    "bin_10_14_r5",
    "bin_15_19_r5",
    "bin_20_24_r5",
    "bin_25_29_r5",
    "current_6_23_r3",
    "late_12_29_r2",
    "late_12_29_r3",
    "late_12_29_r4",
    "late_12_29_r5",
    "late_15_29_r5",
    "late_14_29_r8",
    "late_15_29_r15",
)

FROZEN_CONFIG = {
    "model_name": "720x720_5s",
    "mode": "t2v",
    "video_frame_height_width": [720, 720],
    "sample_steps": SAMPLE_STEPS,
    "solver_name": "euler",
    "shift": 5.0,
    "sp_size": 1,
    "batch_size": 1,
    "audio_guidance_scale": 3.0,
    "video_guidance_scale": 4.0,
    "slg_layer": 11,
    "video_negative_prompt": "jitter, bad hands, blur, distortion",
    "audio_negative_prompt": "robotic, muffled, echo, distorted",
    "attention_method": "dense",
    "fp8": False,
    "qint8": False,
    "cpu_offload": False,
    "use_block_cache": False,
    "debug_forward": False,
    "debug_forward_step": 0,
    "profiling_enabled": False,
}

FROZEN_MEDIA_CONTRACT = {
    "decoded_video_frames": 121,
    "decoded_width": 704,
    "decoded_height": 704,
    "video_codec": "h264",
    "audio_codec": "aac",
    "audio_must_be_non_silent": True,
}

STAGE_SEEDS = {
    "0": (103,),
    "1": (103, 211),
    "2": (103, 211),
    "3": (503, 887, 1291),
}

STAGE0_ORDER = (
    {"label": "D0", "config_id": "dense", "repetition": 1},
    {"label": "null", "config_id": "late_12_29_r1_null", "repetition": 1},
    {"label": "old_14", "config_id": "current_9_26_r5_anchor", "repetition": 1},
    {"label": "new_14_r1", "config_id": "new_12_29_r5_repeat", "repetition": 1},
    {"label": "new_14_r2", "config_id": "new_12_29_r5_repeat", "repetition": 2},
    {"label": "D1", "config_id": "dense", "repetition": 2},
)

STAGE1_CONFIG_IDS = tuple(
    f"bin_{start:02d}_{start + 4:02d}_r5" for start in range(0, 30, 5)
)

STAGE2_CONFIG_IDS = (
    "current_6_23_r3",
    "late_12_29_r2",
    "late_12_29_r3",
    "late_12_29_r4",
    "late_12_29_r5",
    "late_15_29_r5",
    "late_14_29_r8",
    "late_15_29_r15",
)

CANDIDATE_FREEZE_RULE = {
    "freeze_before_stage": 3,
    "required_candidate_count": 2,
    "conservative_12_hit_allowed": ["late_12_29_r3", "late_15_29_r5"],
    "aggressive_14_hit_default": "late_12_29_r5",
    "aggressive_14_hit_allowed": [
        "late_12_29_r5",
        "late_14_29_r8",
        "late_15_29_r15",
    ],
    "optional_13_hit": "late_12_29_r4",
    "max_stage2_cells_advanced_to_dev5": 3,
}

STAGE3_BALANCED_ORDER = {
    503: ("dense", "old_12", "new_12", "old_14", "new_14"),
    887: ("new_12", "old_14", "new_14", "dense", "old_12"),
    1291: ("new_14", "dense", "old_12", "new_12", "old_14"),
}

STAGE3_FIXED = {
    "prompt_count": 8,
    "heldout_prompt_file": "prompts/ovi_cfg_cache_heldout_prompts.csv",
    "heldout_prompt_manifest": "prompts/ovi_cfg_cache_heldout_prompt_manifest.csv",
    "forbidden_substitute": "prompts/ovi_formal8.csv",
    "old_12_config_id": "current_6_23_r3",
    "old_14_config_id": "current_9_26_r5_anchor",
    "balanced_order": STAGE3_BALANCED_ORDER,
}

STAGE3_ALLOWED_CONFIG_IDS = frozenset(
    {
        "dense",
        "current_6_23_r3",
        "current_9_26_r5_anchor",
        *CANDIDATE_FREEZE_RULE["conservative_12_hit_allowed"],
        *CANDIDATE_FREEZE_RULE["aggressive_14_hit_allowed"],
    }
)

STAGE4_ALLOWED_CONFIG_IDS = frozenset(
    {
        "dense",
        *CANDIDATE_FREEZE_RULE["conservative_12_hit_allowed"],
        *CANDIDATE_FREEZE_RULE["aggressive_14_hit_allowed"],
    }
)

STAGE4_FIXED = {
    "workloads": ("dense", "frozen_new_12", "frozen_new_14"),
    "minimum_warmup_runs": 3,
    "minimum_measurement_runs": 5,
    "balanced_configuration_order_required": True,
    "profiling_enabled": False,
    "debug_forward": False,
    "extra_tensor_export": False,
}

PROMPT_SET_CONTRACTS = {
    "stage0": {
        "path": "prompts/ovi_cfg_ablation_v2_stage0.csv",
        "prompt_count": 1,
        "source_dev5_rows_1_based": (1,),
    },
    "dev3": {
        "path": "prompts/ovi_cfg_ablation_v2_dev3.csv",
        "prompt_count": 3,
        "source_dev5_rows_1_based": (1, 4, 5),
    },
    "dev5": {
        "path": "prompts/ovi_cfg_ablation_v2_dev5.csv",
        "prompt_count": 5,
        "source": "runtime/ovi_user5.csv",
    },
    "heldout8": {
        "path": "prompts/ovi_cfg_cache_heldout_prompts.csv",
        "prompt_count": 8,
        "manifest": "prompts/ovi_cfg_cache_heldout_prompt_manifest.csv",
        "forbidden_substitute": "prompts/ovi_formal8.csv",
    },
}

_CONFIG_ID_RE = re.compile(r"^[a-z0-9_]+$")


class ProtocolError(ValueError):
    """A fail-closed ablation protocol validation error."""


@dataclass(frozen=True)
class Workload:
    eligible_steps: int
    refreshes: int
    cache_hits: int
    negative_forwards: int
    expected_video_self_attention_calls: int
    max_cache_age: int


@dataclass(frozen=True)
class Cell:
    stage: str
    config_id: str
    use_cfg_cache: bool
    start_step: int | None
    end_step: int | None
    refresh_interval: int | None
    eligible_steps: int
    refreshes: int
    cache_hits: int
    negative_forwards: int
    expected_video_self_attention_calls: int
    max_cache_age: int
    role: str
    advance_rule: str

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _strict_int(value: str, *, field: str, config_id: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(
            f"{config_id}: {field} must be an integer, found {value!r}"
        ) from exc
    if str(parsed) != value.strip():
        raise ProtocolError(
            f"{config_id}: {field} must use canonical integer syntax, found {value!r}"
        )
    return parsed


def _strict_bool(value: str, *, field: str, config_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ProtocolError(
        f"{config_id}: {field} must be 0/1 or true/false, found {value!r}"
    )


def expected_workload(
    use_cfg_cache: bool,
    start_step: int | None = None,
    end_step: int | None = None,
    refresh_interval: int | None = None,
) -> Workload:
    """Compute the exact analytical workload for one 30-step Ovi run."""

    if not use_cfg_cache:
        if any(value is not None for value in (start_step, end_step, refresh_interval)):
            raise ProtocolError("dense must not define a CFG-cache window")
        return Workload(0, 0, 0, SAMPLE_STEPS, DENSE_VIDEO_SELF_ATTENTION_CALLS, 0)
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (start_step, end_step, refresh_interval)
    ):
        raise ProtocolError("cache cells require integer start/end/refresh values")
    assert start_step is not None and end_step is not None
    assert refresh_interval is not None
    if not 0 <= start_step <= end_step < SAMPLE_STEPS:
        raise ProtocolError(
            f"cache window must be zero-based inclusive inside 0..{SAMPLE_STEPS - 1}"
        )
    if refresh_interval < 1:
        raise ProtocolError("refresh_interval must be >= 1")
    eligible = end_step - start_step + 1
    refreshes = math.ceil(eligible / refresh_interval)
    hits = eligible - refreshes
    negative = SAMPLE_STEPS - hits
    calls = DENSE_VIDEO_SELF_ATTENTION_CALLS - UNCONDITIONAL_VIDEO_BLOCKS * hits
    max_age = min(refresh_interval - 1, eligible - 1)
    return Workload(eligible, refreshes, hits, negative, calls, max_age)


def _parse_cell(row: Mapping[str, str], line_number: int) -> Cell:
    config_id = (row.get("config_id") or "").strip()
    if not _CONFIG_ID_RE.fullmatch(config_id):
        raise ProtocolError(
            f"matrix line {line_number}: invalid config_id {config_id!r}"
        )
    stage = (row.get("stage") or "").strip()
    if stage not in ALLOWED_STAGES:
        raise ProtocolError(
            f"{config_id}: stage must be one of {ALLOWED_STAGES}, found {stage!r}"
        )
    use_cache = _strict_bool(
        row.get("use_cfg_cache", ""), field="use_cfg_cache", config_id=config_id
    )
    if use_cache:
        start = _strict_int(row["start_step"], field="start_step", config_id=config_id)
        end = _strict_int(row["end_step"], field="end_step", config_id=config_id)
        interval = _strict_int(
            row["refresh_interval"], field="refresh_interval", config_id=config_id
        )
    else:
        for field in ("start_step", "end_step", "refresh_interval"):
            if (row.get(field) or "").strip():
                raise ProtocolError(f"{config_id}: dense {field} must be blank")
        start = end = interval = None
    expected = expected_workload(use_cache, start, end, interval)
    observed = Workload(
        *(
            _strict_int(row[field], field=field, config_id=config_id)
            for field in (
                "eligible_steps",
                "refreshes",
                "cache_hits",
                "negative_forwards",
                "expected_video_self_attention_calls",
                "max_cache_age",
            )
        )
    )
    if observed != expected:
        raise ProtocolError(
            f"{config_id}: workload formula mismatch: observed={asdict(observed)} "
            f"expected={asdict(expected)}"
        )
    role = (row.get("role") or "").strip()
    advance_rule = (row.get("advance_rule") or "").strip()
    if not role or not advance_rule:
        raise ProtocolError(f"{config_id}: role and advance_rule must be non-empty")
    return Cell(
        stage,
        config_id,
        use_cache,
        start,
        end,
        interval,
        *asdict(expected).values(),
        role,
        advance_rule,
    )


def load_and_validate_matrix(path: Path) -> list[Cell]:
    """Load a complete matrix, reject schema/formula drift and duplicate IDs."""

    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MATRIX_FIELDS:
                raise ProtocolError(
                    f"matrix header {tuple(reader.fieldnames or ())!r} != {MATRIX_FIELDS!r}"
                )
            rows = list(reader)
    except OSError as exc:
        raise ProtocolError(f"cannot read matrix {path}: {exc}") from exc
    cells: list[Cell] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        cell = _parse_cell(row, line_number)
        if cell.config_id in seen:
            raise ProtocolError(f"duplicate config_id {cell.config_id!r}")
        seen.add(cell.config_id)
        cells.append(cell)
    actual_ids = tuple(cell.config_id for cell in cells)
    if actual_ids != EXPECTED_CONFIG_IDS:
        raise ProtocolError(
            "matrix config IDs/order differ from the authoritative v2 matrix: "
            f"{actual_ids!r}"
        )
    return cells


def validate_frozen_base_config(base: Mapping[str, Any]) -> None:
    """Require a base YAML compatible with every frozen generation constant."""

    if not isinstance(base, Mapping):
        raise ProtocolError("base YAML must contain a top-level mapping")
    errors = []
    for field, expected in FROZEN_CONFIG.items():
        if field in {"batch_size", "profiling_enabled"} and field not in base:
            continue
        actual = base.get(field)
        if isinstance(expected, float):
            equal = isinstance(actual, (int, float)) and not isinstance(actual, bool)
            equal = equal and float(actual) == expected
        else:
            equal = type(actual) is type(expected) and actual == expected
        if not equal:
            errors.append(f"{field}={actual!r} != frozen {expected!r}")
    if errors:
        raise ProtocolError("base config violates frozen constants: " + "; ".join(errors))


def filter_cells(
    cells: Iterable[Cell],
    stages: set[str] | None,
    config_ids: set[str] | None,
) -> list[Cell]:
    """Apply fail-closed stage/config filters after validating the full matrix."""

    cells = list(cells)
    if stages is not None:
        unknown = stages - set(ALLOWED_STAGES)
        if unknown:
            raise ProtocolError(f"unknown stages: {sorted(unknown)}")
        cells = [cell for cell in cells if cell.stage in stages]
    if config_ids is not None:
        known = {cell.config_id for cell in cells}
        missing = config_ids - known
        if missing:
            raise ProtocolError(
                f"unknown or stage-filtered config IDs: {sorted(missing)}"
            )
        cells = [cell for cell in cells if cell.config_id in config_ids]
    if not cells:
        raise ProtocolError("no matrix rows matched the requested filters")
    return cells


def validate_seed_filter(
    seeds: Iterable[int],
    cells: Iterable[Cell],
    execution_stage: str | None = None,
) -> list[int]:
    seeds = list(seeds)
    if not seeds:
        raise ProtocolError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ProtocolError("seed filter contains duplicates")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ProtocolError("seeds must be nonnegative integers")
    selected_stages = {cell.stage for cell in cells}
    if execution_stage == "4":
        # Stage 4 freezes the chosen policies and may use any explicit fixed seed.
        return seeds
    allowed_stages = {execution_stage} if execution_stage is not None else selected_stages
    allowed = set().union(*(STAGE_SEEDS[stage] for stage in allowed_stages))
    unknown = set(seeds) - allowed
    if unknown:
        raise ProtocolError(
            f"seeds {sorted(unknown)} are not allowed for stages {sorted(selected_stages)}"
        )
    return seeds


def cell_filename(
    cell: Cell, seed: int, execution_stage: str | None = None
) -> str:
    """Return a zero-based, inclusive schedule filename."""

    stage = execution_stage if execution_stage is not None else cell.stage
    prefix = f"ovi_cfg_v2_s{stage}_{cell.config_id}"
    if cell.use_cfg_cache:
        schedule = (
            f"steps{cell.start_step:02d}-{cell.end_step:02d}_inclusive_"
            f"r{cell.refresh_interval}"
        )
    else:
        schedule = "dense_no-cache"
    return f"{prefix}_{schedule}_seed{seed}.yaml"


def protocol_summary() -> dict[str, Any]:
    """Return JSON-shaped frozen protocol metadata for manifests and tests."""

    return {
        "protocol_id": PROTOCOL_ID,
        "indexing": "zero_based_inclusive",
        "frozen_config": dict(FROZEN_CONFIG),
        "frozen_media_contract": dict(FROZEN_MEDIA_CONTRACT),
        "stage0_order": [dict(item) for item in STAGE0_ORDER],
        "stage1": {
            "config_ids": list(STAGE1_CONFIG_IDS),
            "prompt_set": "dev3",
            "seeds": list(STAGE_SEEDS["1"]),
            "independent_prompt_seed_units": 6,
        },
        "stage2": {
            "config_ids": list(STAGE2_CONFIG_IDS),
            "prompt_sets": ["stage0", "dev5"],
            "seeds": list(STAGE_SEEDS["2"]),
            "candidate_freeze_rule": dict(CANDIDATE_FREEZE_RULE),
        },
        "stage3": {
            **{key: value for key, value in STAGE3_FIXED.items() if key != "balanced_order"},
            "seeds": list(STAGE_SEEDS["3"]),
            "balanced_order": {
                str(seed): list(order)
                for seed, order in STAGE3_BALANCED_ORDER.items()
            },
        },
        "stage4": {
            **STAGE4_FIXED,
            "workloads": list(STAGE4_FIXED["workloads"]),
        },
        "prompt_sets": {
            name: {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in contract.items()
            }
            for name, contract in PROMPT_SET_CONTRACTS.items()
        },
    }
