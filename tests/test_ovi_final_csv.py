import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_ovi_final_csv.py"
SPEC = importlib.util.spec_from_file_location("build_ovi_final_csv", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
FINAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FINAL
SPEC.loader.exec_module(FINAL)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def tagged_hash(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def make_pinned_protocol_fixture(root: Path) -> tuple[Path, dict]:
    protocol = json.loads(
        (REPO_ROOT / "configs" / "quality_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    cache_root = root / "cache"
    environment_root = cache_root / "envs" / "eval"
    site_packages = environment_root / "lib" / "python3.11" / "site-packages"
    module_path = site_packages / "fake_module.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("# fixed synthetic module\n", encoding="utf-8")
    record_path = site_packages / "fake_dist-1.0.dist-info" / "RECORD"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("fake_module.py,,\n", encoding="utf-8")
    wheelhouse = cache_root / "checkpoints" / "eval" / "wheels"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    wheel_name = "fake_dist-1.0-py3-none-any.whl"
    archive_path = wheelhouse / wheel_name
    archive_path.write_bytes(b"fixed synthetic wheel")
    archive_url = f"https://files.pythonhosted.org/packages/{wheel_name}"
    lock = [
        {
            "distribution": "fake-dist",
            "version": "1.0",
            "source_index": "https://pypi.org/simple",
            "archive_url": archive_url,
            "archive_sha256": digest(archive_path),
        }
    ]
    lock_sha = FINAL._dependency_lock_sha256(lock)
    weight_path = cache_root / "checkpoints" / "eval" / "weight.pth"
    weight_path.write_bytes(b"fixed synthetic weight")
    direct_package = {
        "distribution": "fake-dist",
        "version": "1.0",
        "module": "fake_module",
        "module_path": str(module_path),
        "source_index": "https://pypi.org/simple",
        "trusted_archive_sha256": digest(archive_path),
    }
    weight_contract = {
        "weight_id": "fake-weight",
        "path": str(weight_path),
        "source_type": "url",
        "source": "https://example.invalid/fake-weight.pth",
        "trusted_sha256": digest(weight_path),
    }
    report_paths = []
    for index in range(3):
        report_path = cache_root / "reports" / f"pip-{index}.json"
        write_json(report_path, {"version": "1", "install": []})
        report_paths.append(report_path)
    raw_package = {
        **lock[0],
        "archive_path": str(archive_path),
        "record_path": str(record_path),
        "record_sha256": digest(record_path),
        "module": "fake_module",
        "module_path": str(module_path),
        "module_sha256": digest(module_path),
    }
    raw_weight = {
        "weight_id": "fake-weight",
        "path": str(weight_path),
        "bytes": weight_path.stat().st_size,
        "sha256": digest(weight_path),
        "source_type": "url",
        "source": "https://example.invalid/fake-weight.pth",
    }
    receipt_path = cache_root / "checkpoints" / "eval" / "lpips_receipt.json"
    raw_receipt = {
        "schema_version": 2,
        "created_by": "scripts/install_ovi_quality_env.sh",
        "environment_root": str(environment_root),
        "python_executable": str(environment_root / "bin" / "python"),
        "sys_prefix": str(environment_root),
        "python_version": "3.11.15",
        "runtime_contract": {
            "python_arguments": ["-I", "-S", "-B"],
            "python_minor": "3.11",
            "site_packages": str(site_packages),
        },
        "environment_lock_sha256": lock_sha,
        "installer_reports": [
            {"path": str(path), "sha256": digest(path)} for path in report_paths
        ],
        "packages": [raw_package],
        "weights": [raw_weight],
    }
    write_json(receipt_path, raw_receipt)
    lpips = protocol["lpips"]
    lpips.update(
        {
            "environment_root": str(environment_root),
            "python_executable": str(environment_root / "bin" / "python"),
            "receipt_path": str(receipt_path),
            "trusted_lock_status": "pinned",
            "trusted_environment_lock_sha256": lock_sha,
            "trusted_environment_packages": lock,
            "packages": [direct_package],
            "weights": [weight_contract],
        }
    )
    protocol_path = root / "quality_protocol.json"
    write_json(protocol_path, protocol)
    inline = {
        "receipt_path": str(receipt_path),
        "receipt_sha256": digest(receipt_path),
        "environment_root": str(environment_root),
        "python_executable": str(environment_root / "bin" / "python"),
        "sys_prefix": str(environment_root),
        "python_version": "3.11.15",
        "runtime_contract": raw_receipt["runtime_contract"],
        "environment_lock_sha256": lock_sha,
        "packages": [raw_package],
        "weights": [raw_weight],
    }
    return protocol_path, inline


class FinalCsvFixture:
    """Create the exact schemas emitted by the timing and quality builders."""

    def __init__(self, root: Path, lpips_inline: dict):
        self.root = root
        self.protocol_path = FINAL.DEFAULT_PROTOCOL
        self.matrix_path = REPO_ROOT / "configs" / "ovi_eval_matrix.json"
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.matrix = json.loads(self.matrix_path.read_text(encoding="utf-8"))
        with (REPO_ROOT / "prompts" / "ovi_formal8.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            self.prompts = [
                row["text_prompt"] for row in csv.DictReader(handle)
            ]
        self.asserted_prompt_sha256 = digest_bytes(
            json.dumps(
                self.prompts,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        if self.asserted_prompt_sha256 != self.matrix["fixed_protocol"][
            "prompts_sha256"
        ]:
            raise AssertionError("formal prompt fixture hash drifted")
        self.artifact_count = (
            self.matrix["fixed_protocol"]["measurement_runs"]
            * self.matrix["fixed_protocol"]["prompt_count"]
            * self.matrix["fixed_protocol"]["each_example_n_times"]
        )
        self.identities = [
            (measurement, prompt, sample)
            for measurement in self.protocol["measurement_indices"]
            for prompt in range(self.matrix["fixed_protocol"]["prompt_count"])
            for sample in range(
                self.matrix["fixed_protocol"]["each_example_n_times"]
            )
        ]
        self.commit = "a" * 40
        self.timing_path = self.root / "timing.csv"
        self.output_path = self.root / "final.csv"
        self.quality_paths: dict[str, Path] = {}
        self.manual_paths: dict[str, Path] = {}
        self.rows: dict[str, dict[str, str]] = {}
        self.artifacts: dict[str, dict[tuple[int, int, int], str]] = {}
        self.run_bindings: dict[str, dict] = {}
        self.metrics_hashes: dict[
            str, dict[tuple[int, int, int], str]
        ] = {}
        self.lpips_inline = lpips_inline

    @property
    def method_records(self) -> dict[str, dict]:
        return {item["method_id"]: item for item in self.matrix["methods"]}

    def make_timing_row(self, method_id: str, index: int) -> dict[str, str]:
        row = {field: "" for field in FINAL.TIMING_FIELDS}
        run_dir = self.root / "runs" / method_id
        run_dir.mkdir(parents=True, exist_ok=False)
        artifacts = {
            identity: tagged_hash(f"{method_id}-artifact-{identity}")
            for identity in self.identities
        }
        self.artifacts[method_id] = artifacts
        row.update(
            {
                "method_id": method_id,
                "label": self.method_records[method_id]["label"],
                "required": "True",
                "implementation_status": "ready",
                "status": "pending",
                "timing_status": "valid",
                "pending_reason": "Quality metric and manual review are not yet provided.",
                "run_dir": str(run_dir),
                "run_id": f"formal-{method_id}",
                "verification_sha256": tagged_hash(f"verify-{method_id}"),
                "timings_path": str(run_dir / "timings.jsonl"),
                "timings_bytes": str(1000 + index),
                "timings_sha256": tagged_hash(f"timings-{method_id}"),
                "timings_record_count": str(self.artifact_count),
                "warmup_timings_path": str(run_dir / "warmup_timings.jsonl"),
                "warmup_timings_bytes": str(100 + index),
                "warmup_timings_sha256": tagged_hash(f"warmup-{method_id}"),
                "warmup_record_count": "1",
                "git_commit": self.commit,
                "checkpoint_manifest_sha256": tagged_hash("checkpoint-manifest"),
                "checkpoint_fingerprint_sha256": tagged_hash("checkpoint-fingerprint"),
                "gpu_uuid": "GPU-fixture",
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "prompt_set_sha256": self.matrix["fixed_protocol"]["prompts_sha256"],
                "prompt_count": str(self.matrix["fixed_protocol"]["prompt_count"]),
                "seed": (
                    "103"
                    if self.matrix["fixed_protocol"]["each_example_n_times"] == 1
                    else ""
                ),
                "seed_count": str(
                    self.matrix["fixed_protocol"]["each_example_n_times"]
                ),
                "seeds": ";".join(
                    str(103 + index)
                    for index in range(
                        self.matrix["fixed_protocol"]["each_example_n_times"]
                    )
                ),
                "requested_height": "720",
                "requested_width": "720",
                "actual_height": "704",
                "actual_width": "704",
                "sample_steps": str(self.matrix["fixed_protocol"]["sample_steps"]),
                "measurement_count": "3",
                "measurement_indices": "0;1;2",
                "artifact_count": str(self.artifact_count),
                "denoise_seconds_median": str(20.0 - index),
                "total_generation_seconds_median": str(30.0 - index),
                "artifact_ready_seconds_median": str(31.0 - index),
                "peak_memory_allocated_gib_median": "50.0",
                "peak_memory_reserved_gib_median": "60.0",
                "denoise_speedup_vs_dense": "1.0" if method_id == "dense" else str(20.0 / (20.0 - index)),
                "total_speedup_vs_dense": "1.0" if method_id == "dense" else str(30.0 / (30.0 - index)),
                "artifact_sha256": ";".join(
                    f"{identity[0]}:{identity[1]}:{identity[2]}:{artifacts[identity]}"
                    for identity in self.identities
                ),
                "metrics_sidecar_sha256": ";".join(
                    f"{identity[0]}:{identity[1]}:{identity[2]}:"
                    f"{tagged_hash(f'{method_id}-metrics-{identity}')}"
                    for identity in self.identities
                ),
            }
        )
        return row

    def make_pending_timing_row(self, method_id: str) -> dict[str, str]:
        method = self.method_records[method_id]
        row = {field: "" for field in FINAL.TIMING_FIELDS}
        row.update(
            {
                "method_id": method_id,
                "label": method["label"],
                "required": "True" if method["required"] else "False",
                "implementation_status": method["implementation_status"],
                "status": "pending",
                "timing_status": "pending",
                "pending_reason": method.get("pending_reason")
                or "No explicit run mapping was provided.",
            }
        )
        return row

    def make_run_binding(self, method_id: str) -> dict:
        row = self.rows[method_id]
        method = self.method_records[method_id]
        run_dir = Path(row["run_dir"])
        acceleration = dict(method["expected_environment"])
        generated_video_shape = [3, 121, 704, 704]
        generated_audio_shape = [80640]
        metrics_hashes = {}
        timing_records = []
        for identity in self.identities:
            stem = (
                f"measurement_{identity[0]:02d}_prompt_{identity[1]:03d}_"
                f"sample_{identity[2]:03d}"
            )
            artifact_path = run_dir / f"{stem}.mp4"
            artifact_path.write_bytes(
                f"{method_id}-artifact-{identity}".encode("utf-8")
            )
            metrics_path = run_dir / f"{stem}.metrics.json"
            metrics_payload = {
                "status": "ok",
                "record_type": "measurement",
                "benchmark_candidate": True,
                "benchmark_valid": False,
                "run_id": row["run_id"],
                "measurement_index": identity[0],
                "prompt_index": identity[1],
                "sample_index": identity[2],
                "prompt": self.prompts[identity[1]],
                "seed": 103 + identity[2],
                "sample_steps": int(row["sample_steps"]),
                "requested_video_frame_height_width": [720, 720],
                "actual_video_frame_height_width": [704, 704],
                "generated_video_shape": generated_video_shape,
                "generated_audio_shape": generated_audio_shape,
                "output_path": str(artifact_path),
                "output_sha256": self.artifacts[method_id][identity],
            }
            for field in (
                "attention_method",
                "use_cfg_cache",
                "use_block_cache",
            ):
                if field in acceleration:
                    metrics_payload[field] = acceleration[field]
            write_json(metrics_path, metrics_payload)
            metrics_hashes[identity] = digest(metrics_path)
            timing_records.append(metrics_payload)
        self.metrics_hashes[method_id] = metrics_hashes
        row["metrics_sidecar_sha256"] = ";".join(
            f"{identity[0]}:{identity[1]}:{identity[2]}:"
            f"{metrics_hashes[identity]}"
            for identity in self.identities
        )

        evidence_payloads = {
            "environment.json": {"method_id": method_id, **acceleration},
            "verification.json": {"status": "ok", "method_id": method_id},
            "checkpoint_manifest.json": {
                "checkpoint_fingerprint_sha256": row[
                    "checkpoint_fingerprint_sha256"
                ]
            },
        }
        for name, payload in evidence_payloads.items():
            write_json(run_dir / name, payload)
        timings_path = run_dir / "timings.jsonl"
        timings_path.write_text(
            "".join(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                for record in timing_records
            ),
            encoding="utf-8",
        )
        warmup_path = run_dir / "warmup_timings.jsonl"
        warmup_path.write_text(
            json.dumps({"status": "ok", "record_type": "warmup"}) + "\n",
            encoding="utf-8",
        )
        row.update(
            {
                "verification_sha256": digest(run_dir / "verification.json"),
                "timings_bytes": str(timings_path.stat().st_size),
                "timings_sha256": digest(timings_path),
                "warmup_timings_bytes": str(warmup_path.stat().st_size),
                "warmup_timings_sha256": digest(warmup_path),
                "checkpoint_manifest_sha256": digest(
                    run_dir / "checkpoint_manifest.json"
                ),
            }
        )
        evidence_bindings = {}
        for name in (
            "environment.json",
            "verification.json",
            "timings.jsonl",
            "warmup_timings.jsonl",
            "checkpoint_manifest.json",
        ):
            path = run_dir / name
            evidence_bindings[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        return {
            "method_id": method_id,
            "run_dir": row["run_dir"],
            "run_id": row["run_id"],
            "verification_sha256": row["verification_sha256"],
            "timings_path": row["timings_path"],
            "timings_bytes": int(row["timings_bytes"]),
            "timings_sha256": row["timings_sha256"],
            "timings_record_count": int(row["timings_record_count"]),
            "warmup_timings_path": row["warmup_timings_path"],
            "warmup_timings_bytes": int(row["warmup_timings_bytes"]),
            "warmup_timings_sha256": row["warmup_timings_sha256"],
            "warmup_record_count": int(row["warmup_record_count"]),
            "environment_sha256": digest(run_dir / "environment.json"),
            "git_commit": row["git_commit"],
            "checkpoint_manifest_sha256": row["checkpoint_manifest_sha256"],
            "checkpoint_fingerprint_sha256": row[
                "checkpoint_fingerprint_sha256"
            ],
            "gpu_physical_index": 0,
            "gpu_uuid": row["gpu_uuid"],
            "gpu_name": row["gpu_name"],
            "prompt_set_sha256": row["prompt_set_sha256"],
            "prompt_count": int(row["prompt_count"]),
            "prompts": self.prompts,
            "base_seed": 103,
            "sample_count": self.matrix["fixed_protocol"][
                "each_example_n_times"
            ],
            "sample_seeds": [
                103 + index
                for index in range(
                    self.matrix["fixed_protocol"]["each_example_n_times"]
                )
            ],
            "selected_sparse_profile": "",
            "requested_shape": [720, 720],
            "actual_shape": [704, 704],
            "generated_video_shape": generated_video_shape,
            "generated_audio_shape": generated_audio_shape,
            "sample_steps": int(row["sample_steps"]),
            "acceleration_environment": acceleration,
            "evidence_bindings": evidence_bindings,
        }

    def evaluator_receipt(self) -> dict:
        files = {}
        for role, path in FINAL.SOURCE_ROLES.items():
            path = Path(path).resolve()
            files[role] = {"path": str(path), "sha256": digest(path)}
        return {"git_commit": self.commit, "files": files}

    def artifact_binding(
        self, method_id: str, identity: tuple[int, int, int]
    ) -> dict:
        binding = json.loads(json.dumps(self.run_bindings[method_id]))
        run_dir = Path(binding["run_dir"])
        stem = (
            f"measurement_{identity[0]:02d}_prompt_{identity[1]:03d}_"
            f"sample_{identity[2]:03d}"
        )
        binding.update(
            {
                "measurement_index": identity[0],
                "prompt_index": identity[1],
                "sample_index": identity[2],
                "artifact_path": str(run_dir / f"{stem}.mp4"),
                "artifact_sha256": self.artifacts[method_id][identity],
                "metrics_sidecar_path": str(run_dir / f"{stem}.metrics.json"),
                "metrics_sidecar_sha256": self.metrics_hashes[method_id][
                    identity
                ],
            }
        )
        return binding

    def metrics(self, candidate_index: int, identity: tuple[int, int, int]) -> dict:
        # Values are floats, as emitted by the real metric normalizer.  Varying
        # one dimension makes the fixture exercise median recomputation.
        variation = identity[1] * 0.001
        return {
            "compared_video_frames": 121,
            "reference_audio_samples": 80640,
            "candidate_audio_samples": 80640,
            "audio_sample_count_compared": 80640,
            "lpips_frame_count": 121,
            "video_psnr_db": 30.0 + candidate_index + variation,
            "video_ssim": 0.90 + candidate_index * 0.001,
            "lpips_alex": 0.10 + candidate_index * 0.001 + variation,
            "audio_rmse": 0.01 + candidate_index * 0.001,
            "audio_max_abs_difference": 0.1 + candidate_index * 0.001,
            "audio_snr_db": 20.0 + candidate_index,
            "audio_correlation": 0.95,
        }

    def make_quality_and_manual(self, method_id: str, candidate_index: int) -> None:
        quality_dir = self.root / "quality" / method_id
        quality_dir.mkdir(parents=True, exist_ok=False)
        evaluator = self.evaluator_receipt()
        lpips_receipt = self.lpips_inline
        media_receipt = {
            "tools": [
                {
                    "name": "ffmpeg",
                    "path": "/usr/bin/false",
                    "sha256": tagged_hash("ffmpeg"),
                    "version_line": "fixture",
                }
            ]
        }
        pair_bindings = []
        metric_values = {field: [] for field in FINAL.METRIC_FIELDS}
        for identity in self.identities:
            metrics = self.metrics(candidate_index, identity)
            for field in FINAL.METRIC_FIELDS:
                metric_values[field].append(float(metrics[field]))
            pair_payload = {
                "schema_version": 2,
                "record_type": "ovi_quality_pair",
                "quality_protocol_id": self.protocol["protocol_id"],
                "quality_protocol_sha256": digest(self.protocol_path),
                "measurement_index": identity[0],
                "prompt_index": identity[1],
                "sample_index": identity[2],
                "dense": self.artifact_binding("dense", identity),
                "candidate": self.artifact_binding(method_id, identity),
                "metrics": metrics,
                "automatic_acceptance": None,
                "comparison_script_sha256": evaluator["files"][
                    "comparison_script"
                ]["sha256"],
                "compare_media_script_sha256": evaluator["files"][
                    "compare_media_script"
                ]["sha256"],
                "run_validator_script_sha256": evaluator["files"][
                    "run_validator_script"
                ]["sha256"],
                "evaluation_matrix_sha256": evaluator["files"][
                    "evaluation_matrix"
                ]["sha256"],
                "evaluator_source_receipt": evaluator,
                "lpips_dependency_receipt": lpips_receipt,
                "media_tool_receipt": media_receipt,
            }
            pair_path = quality_dir / (
                f"measurement_{identity[0]:02d}_prompt_{identity[1]:03d}_"
                f"sample_{identity[2]:03d}.quality.json"
            )
            write_json(pair_path, pair_payload)
            pair_bindings.append(
                {
                    "measurement_index": identity[0],
                    "prompt_index": identity[1],
                    "sample_index": identity[2],
                    "pair_sidecar_path": str(pair_path),
                    "pair_sidecar_sha256": digest(pair_path),
                    "dense_artifact_sha256": self.artifacts["dense"][identity],
                    "candidate_artifact_sha256": self.artifacts[method_id][
                        identity
                    ],
                }
            )
        medians = {
            field: float(statistics.median(values))
            for field, values in metric_values.items()
        }
        median_path = quality_dir / "median.quality.json"
        median_payload = {
            "schema_version": 2,
            "record_type": "ovi_quality_median",
            "quality_protocol_id": self.protocol["protocol_id"],
            "quality_protocol_sha256": digest(self.protocol_path),
            "comparison_script_sha256": evaluator["files"]["comparison_script"][
                "sha256"
            ],
            "compare_media_script_sha256": evaluator["files"][
                "compare_media_script"
            ]["sha256"],
            "run_validator_script_sha256": evaluator["files"][
                "run_validator_script"
            ]["sha256"],
            "evaluation_matrix_sha256": evaluator["files"]["evaluation_matrix"][
                "sha256"
            ],
            "evaluator_source_receipt": evaluator,
            "lpips_dependency_receipt": lpips_receipt,
            "media_tool_receipt": media_receipt,
            "dense_run": self.run_bindings["dense"],
            "candidate_run": self.run_bindings[method_id],
            "pairs": pair_bindings,
            "pair_count": self.artifact_count,
            "metric_medians": medians,
            "automatic_acceptance": None,
            "manual_review": {
                "status": "not_provided",
                "row_count": 0,
                "csv_path": None,
                "csv_sha256": None,
            },
        }
        write_json(median_path, median_payload)
        self.quality_paths[method_id] = median_path

        manual_csv = self.root / "reviews" / f"{method_id}.csv"
        manual_csv.parent.mkdir(parents=True, exist_ok=True)
        with manual_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FINAL.MANUAL_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for offset, identity in enumerate(self.identities):
                rating = (
                    "fail"
                    if offset == self.artifact_count - 2
                    else "uncertain"
                    if offset == self.artifact_count - 1
                    else "pass"
                )
                writer.writerow(
                    {
                        "measurement_index": identity[0],
                        "prompt_index": identity[1],
                        "sample_index": identity[2],
                        "dense_artifact_sha256": self.artifacts["dense"][
                            identity
                        ],
                        "candidate_artifact_sha256": self.artifacts[method_id][
                            identity
                        ],
                        "reviewer": "fixture-human",
                        "reviewed_at_utc": "2026-07-14T12:00:00Z",
                        "sync_rating": rating,
                        "notes": "human-authored fixture row",
                    }
                )
        receipt_path = quality_dir / "manual-review.validation.json"
        receipt_payload = {
            "schema_version": 2,
            "record_type": "ovi_manual_sync_review_validation",
            "quality_protocol_id": self.protocol["protocol_id"],
            "quality_protocol_sha256": digest(self.protocol_path),
            "quality_median_path": str(median_path),
            "quality_median_sha256": digest(median_path),
            "manual_reviews_csv_path": str(manual_csv),
            "manual_reviews_csv_sha256": digest(manual_csv),
            "manual_review_status": "complete",
            "manual_review_row_count": self.artifact_count,
            "pairs": [
                {
                    "measurement_index": identity[0],
                    "prompt_index": identity[1],
                    "sample_index": identity[2],
                    "dense_artifact_sha256": self.artifacts["dense"][identity],
                    "candidate_artifact_sha256": self.artifacts[method_id][
                        identity
                    ],
                }
                for identity in self.identities
            ],
        }
        write_json(receipt_path, receipt_payload)
        self.manual_paths[method_id] = receipt_path

    def build(self) -> None:
        for index, method_id in enumerate(FINAL.METHOD_IDS):
            self.rows[method_id] = self.make_timing_row(method_id, index)
        self.run_bindings = {
            method_id: self.make_run_binding(method_id)
            for method_id in FINAL.METHOD_IDS
        }
        with self.timing_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FINAL.TIMING_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(self.rows[method_id] for method_id in FINAL.METHOD_IDS)
            writer.writerows(
                self.make_pending_timing_row(item["method_id"])
                for item in self.matrix["methods"][len(FINAL.METHOD_IDS) :]
            )
        for index, method_id in enumerate(FINAL.CANDIDATE_METHOD_IDS, start=1):
            self.make_quality_and_manual(method_id, index)


class OviFinalCsvTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name).resolve()
        self.protocol_path, self.lpips_inline = make_pinned_protocol_fixture(
            self.root
        )
        self.full_timing_audit_impl = FINAL._full_validate_timing_runs
        self.recollect_media_impl = FINAL._recollect_media_receipt
        self.recollect_source_impl = FINAL._recollect_evaluator_source
        self.recompute_metrics_impl = FINAL._recompute_quality_metrics
        self.output_path_impl = FINAL._output_path
        self.output_writer_impl = FINAL._write_atomic_exclusive
        self.patches = [
            mock.patch.object(FINAL, "DEFAULT_PROTOCOL", self.protocol_path),
            mock.patch.dict(
                FINAL.SOURCE_ROLES,
                {"quality_protocol": self.protocol_path},
            ),
            mock.patch.object(FINAL, "_require_fixed_eval_runtime"),
            mock.patch.object(
                FINAL, "_audit_repository_source", return_value="a" * 40
            ),
            mock.patch.object(FINAL, "_full_validate_quality_protocol"),
            mock.patch.object(FINAL, "_full_validate_timing_runs"),
            mock.patch.object(FINAL, "_full_validate_lpips_environment"),
            mock.patch.object(
                FINAL,
                "_validate_media_receipt",
                side_effect=lambda receipt, registry, context: receipt,
            ),
            mock.patch.object(FINAL, "_recollect_media_receipt"),
            mock.patch.object(FINAL, "_recollect_evaluator_source"),
            mock.patch.object(FINAL, "_recompute_quality_metrics"),
        ]
        self.started_patches = [patch.start() for patch in self.patches]
        self.fixture = FinalCsvFixture(self.root, self.lpips_inline)
        self.fixture.build()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.temporary.cleanup()

    def build(self, output: Path | None = None) -> Path:
        return FINAL.build_final_csv(
            timing_csv=self.fixture.timing_path,
            quality_paths=self.fixture.quality_paths,
            manual_paths=self.fixture.manual_paths,
            output=output or self.fixture.output_path,
        )

    def test_complete_real_schema_fixture_produces_exact_a_to_f(self):
        output = self.build()
        for patched in self.started_patches[2:]:
            patched.assert_called_once()
        with output.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames or ()), FINAL.FINAL_FIELDS)
            rows = list(reader)
        self.assertEqual([row["method_id"] for row in rows], list(FINAL.METHOD_IDS))
        self.assertTrue(all(row["status"] == "complete" for row in rows))
        self.assertEqual(rows[0]["quality_status"], "reference")
        self.assertEqual(rows[0]["manual_review_status"], "reference")
        self.assertEqual(rows[0]["video_psnr_db_median"], "")
        candidate = rows[1]
        self.assertEqual(candidate["quality_status"], "complete")
        self.assertEqual(candidate["manual_review_status"], "complete")
        self.assertEqual(
            candidate["manual_review_row_count"], str(self.fixture.artifact_count)
        )
        self.assertEqual(
            candidate["manual_pass_count"], str(self.fixture.artifact_count - 2)
        )
        self.assertEqual(candidate["manual_fail_count"], "1")
        self.assertEqual(candidate["manual_uncertain_count"], "1")
        self.assertEqual(
            candidate["quality_median_sha256"],
            digest(self.fixture.quality_paths["dense_cfg_cache"]),
        )
        self.assertEqual(
            candidate["manual_validation_sha256"],
            digest(self.fixture.manual_paths["dense_cfg_cache"]),
        )

    def test_refuses_to_overwrite_complete_output(self):
        self.build()
        original = self.fixture.output_path.read_bytes()
        with self.assertRaisesRegex(FINAL.FinalCsvError, "refusing to overwrite"):
            self.build()
        self.assertEqual(self.fixture.output_path.read_bytes(), original)

    def test_exact_b_to_f_quality_and_manual_sets_are_required_before_write(self):
        qualities = dict(self.fixture.quality_paths)
        qualities.pop("radial_aggressive")
        with self.assertRaisesRegex(FINAL.FinalCsvError, "exactly one B--F"):
            FINAL.build_final_csv(
                timing_csv=self.fixture.timing_path,
                quality_paths=qualities,
                manual_paths=self.fixture.manual_paths,
                output=self.fixture.output_path,
            )
        self.assertFalse(self.fixture.output_path.exists())

    def test_candidate_run_method_binding_must_match_timing_slot(self):
        path = self.fixture.quality_paths["dense_cfg_cache"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["candidate_run"]["method_id"] = "sparge_topk50"
        write_json(path, payload)
        with self.assertRaisesRegex(FINAL.FinalCsvError, "method_id differs"):
            self.build()
        self.assertFalse(self.fixture.output_path.exists())

    def test_pair_count_must_equal_current_fixed_protocol_artifact_count(self):
        path = self.fixture.quality_paths["sparge_topk50"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["pair_count"] = self.fixture.artifact_count - 1
        write_json(path, payload)
        with self.assertRaisesRegex(FINAL.FinalCsvError, "pair_count differs"):
            self.build()
        self.assertFalse(self.fixture.output_path.exists())

    def test_manual_csv_hash_and_human_rows_remain_receipt_bound(self):
        receipt_path = self.fixture.manual_paths["sparge_topk75"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        csv_path = Path(receipt["manual_reviews_csv_path"])
        csv_path.write_bytes(csv_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(FINAL.FinalCsvError, "manual CSV SHA256 drifted"):
            self.build()
        self.assertFalse(self.fixture.output_path.exists())

    def test_pair_metrics_sidecar_hash_must_match_timing_csv(self):
        median_path = self.fixture.quality_paths["radial_conservative"]
        median = json.loads(median_path.read_text(encoding="utf-8"))
        pair_path = Path(median["pairs"][0]["pair_sidecar_path"])
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        pair["candidate"]["metrics_sidecar_sha256"] = tagged_hash("forged")
        write_json(pair_path, pair)
        median["pairs"][0]["pair_sidecar_sha256"] = digest(pair_path)
        write_json(median_path, median)
        with self.assertRaisesRegex(
            FINAL.FinalCsvError,
            "metrics sidecar SHA256 differs from timing CSV",
        ):
            self.build()
        self.assertFalse(self.fixture.output_path.exists())

    def test_json_bool_cannot_impersonate_integer_pair_identity(self):
        median_path = self.fixture.quality_paths["radial_aggressive"]
        median = json.loads(median_path.read_text(encoding="utf-8"))
        # Pair offset three is identity (0, 1, 0) under the current protocol.
        median["pairs"][3]["prompt_index"] = True
        write_json(median_path, median)
        with self.assertRaisesRegex(FINAL.FinalCsvError, "nonnegative integers"):
            self.build()
        self.assertFalse(self.fixture.output_path.exists())

    def test_nonformal_g_h_rows_must_remain_empty_and_pending(self):
        with self.fixture.timing_path.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        rows[len(FINAL.METHOD_IDS)]["artifact_count"] = "1"
        with self.fixture.timing_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FINAL.TIMING_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(
            FINAL.FinalCsvError,
            "excluded slot unexpectedly contains artifact_count",
        ):
            self.build()
        self.assertFalse(self.fixture.output_path.exists())

    def test_symlinked_primary_input_is_rejected(self):
        link = self.root / "timing-link.csv"
        link.symlink_to(self.fixture.timing_path)
        with self.assertRaisesRegex(FINAL.FinalCsvError, "symlink or alias"):
            FINAL.build_final_csv(
                timing_csv=link,
                quality_paths=self.fixture.quality_paths,
                manual_paths=self.fixture.manual_paths,
                output=self.fixture.output_path,
            )
        self.assertFalse(self.fixture.output_path.exists())

    def test_duplicate_assignment_is_rejected(self):
        values = [
            f"{method_id}={self.fixture.quality_paths[method_id]}"
            for method_id in FINAL.CANDIDATE_METHOD_IDS
        ]
        values.append(values[0])
        with self.assertRaisesRegex(FINAL.FinalCsvError, "duplicate method"):
            FINAL._parse_assignments(values, FINAL.CANDIDATE_METHOD_IDS, "--quality")

    def test_unpinned_quality_protocol_is_rejected_before_output(self):
        protocol = json.loads(
            self.protocol_path.read_text(encoding="utf-8")
        )
        protocol["lpips"]["trusted_lock_status"] = "bootstrap_unpinned"
        write_json(self.protocol_path, protocol)
        with self.assertRaisesRegex(
            FINAL.FinalCsvError,
            "trusted_lock_status must be pinned",
        ):
            self.build()
        self.assertFalse(self.fixture.output_path.exists())

    def test_full_timing_audit_calls_fixed_validator_for_every_a_to_f_run(self):
        excluded = {
            "method_id",
            "label",
            "required",
            "implementation_status",
            "status",
            "timing_status",
            "pending_reason",
            "denoise_speedup_vs_dense",
            "total_speedup_vs_dense",
            "quality_metric_name",
            "quality_score",
            "manual_review",
        }
        summaries = {}
        for method_id in FINAL.METHOD_IDS:
            row = self.fixture.rows[method_id]
            summary = {
                field: value
                for field, value in row.items()
                if field not in excluded and value != ""
            }
            summary["denoise_seconds_median"] = float(
                row["denoise_seconds_median"]
            )
            summary["total_generation_seconds_median"] = float(
                row["total_generation_seconds_median"]
            )
            summaries[method_id] = summary
        validator = mock.Mock()
        validator.CSV_FIELDS = FINAL.TIMING_FIELDS
        validator.validate_run.side_effect = (
            lambda method, run_dir, fixed: summaries[method["method_id"]]
        )
        rows = {
            method_id: dict(self.fixture.rows[method_id])
            for method_id in FINAL.METHOD_IDS
        }
        with mock.patch.object(
            FINAL, "_load_fixed_module", return_value=validator
        ):
            self.full_timing_audit_impl(
                self.fixture.matrix,
                rows,
                FINAL.SnapshotRegistry(),
                "test timing audit",
            )
        self.assertEqual(validator.validate_run.call_count, len(FINAL.METHOD_IDS))

        rows["radial_aggressive"]["gpu_name"] = "forged GPU"
        validator.reset_mock()
        with mock.patch.object(
            FINAL, "_load_fixed_module", return_value=validator
        ):
            with self.assertRaisesRegex(
                FINAL.FinalCsvError,
                "timing CSV field gpu_name differs",
            ):
                self.full_timing_audit_impl(
                    self.fixture.matrix,
                    rows,
                    FINAL.SnapshotRegistry(),
                    "test timing audit",
                )

    def test_media_recollection_must_equal_submitted_receipt(self):
        validator = mock.Mock()
        validator.collect_media_tool_receipt.return_value = {
            "tools": [{"name": "different"}]
        }
        with mock.patch.object(
            FINAL, "_load_fixed_module", return_value=validator
        ):
            with self.assertRaisesRegex(
                FINAL.FinalCsvError,
                "differs from freshly collected",
            ):
                self.recollect_media_impl(
                    {"tools": []},
                    FINAL.SnapshotRegistry(),
                    "test media receipt",
                )

    def test_fresh_evaluator_source_receipt_must_equal_clean_head(self):
        receipt = {"git_commit": "a" * 40, "files": {}}
        timing_rows = {
            method_id: {"git_commit": "a" * 40}
            for method_id in FINAL.METHOD_IDS
        }
        validator = mock.Mock()
        validator.capture_evaluator_source_receipt.return_value = receipt
        with mock.patch.object(
            FINAL, "_load_fixed_module", return_value=validator
        ):
            self.recollect_source_impl(
                receipt,
                timing_rows,
                "a" * 40,
                FINAL.SnapshotRegistry(),
                "test evaluator source",
            )
        validator.capture_evaluator_source_receipt.return_value = {
            "git_commit": "b" * 40,
            "files": {},
        }
        with mock.patch.object(
            FINAL, "_load_fixed_module", return_value=validator
        ):
            with self.assertRaisesRegex(
                FINAL.FinalCsvError,
                "differs from clean tracked HEAD sources",
            ):
                self.recollect_source_impl(
                    receipt,
                    timing_rows,
                    "a" * 40,
                    FINAL.SnapshotRegistry(),
                    "test evaluator source",
                )

    def test_every_persisted_metric_is_independently_recomputed(self):
        dense_path = self.root / "recompute-dense.mp4"
        candidate_path = self.root / "recompute-candidate.mp4"
        dense_path.write_bytes(b"dense")
        candidate_path.write_bytes(b"candidate")
        computed = {
            "compared_video_frames": 121,
            "reference_audio_samples": 80640,
            "candidate_audio_samples": 80640,
            "audio_sample_count_compared": 80640,
            "lpips_frame_count": 121,
            "video_psnr_db": 31.0,
            "video_ssim": 0.91,
            "lpips_alex": 0.12,
            "audio_rmse": 0.01,
            "audio_max_abs_difference": 0.1,
            "audio_snr_db": 20.0,
            "audio_correlation": 0.95,
        }
        metric_pairs = {
            identity: (dense_path, candidate_path, dict(computed))
            for identity in self.fixture.identities
        }
        qualities = {
            method_id: mock.Mock(metric_pairs=metric_pairs)
            for method_id in FINAL.CANDIDATE_METHOD_IDS
        }
        metric_runner = mock.Mock(return_value=dict(computed))
        validator = mock.Mock()
        validator.validate_media_tool_receipt.return_value = {}
        validator.LpipsAlexCpu.return_value = mock.Mock()
        validator.make_metric_runner.return_value = metric_runner
        validator._normalize_metrics.side_effect = (
            lambda payload, context: (payload, {})
        )
        with mock.patch.object(
            FINAL, "_load_fixed_module", return_value=validator
        ):
            self.recompute_metrics_impl(
                qualities,
                self.fixture.protocol,
                {"tools": []},
                FINAL.SnapshotRegistry(),
                "test metric recomputation",
            )
        self.assertEqual(
            metric_runner.call_count,
            FINAL.FORMAL_ARTIFACT_COUNT
            * len(FINAL.CANDIDATE_METHOD_IDS),
        )

        forged_pairs = dict(metric_pairs)
        identity = self.fixture.identities[0]
        dense, candidate, persisted = forged_pairs[identity]
        persisted = dict(persisted)
        persisted["lpips_alex"] = 0.999
        forged_pairs[identity] = (dense, candidate, persisted)
        qualities["dense_cfg_cache"] = mock.Mock(metric_pairs=forged_pairs)
        with mock.patch.object(
            FINAL, "_load_fixed_module", return_value=validator
        ):
            with self.assertRaisesRegex(
                FINAL.FinalCsvError,
                "differ from independent MP4 recomputation",
            ):
                self.recompute_metrics_impl(
                    qualities,
                    self.fixture.protocol,
                    {"tools": []},
                    FINAL.SnapshotRegistry(),
                    "test metric recomputation",
                )

    def test_formal_matrix_cannot_shrink_below_72_artifacts(self):
        matrix = json.loads(
            (REPO_ROOT / "configs" / "ovi_eval_matrix.json").read_text(
                encoding="utf-8"
            )
        )
        matrix["fixed_protocol"]["prompt_count"] = 1
        matrix["fixed_protocol"]["each_example_n_times"] = 1
        matrix_path = self.root / "shrunk-matrix.json"
        write_json(matrix_path, matrix)
        with mock.patch.object(FINAL, "DEFAULT_MATRIX", matrix_path):
            with self.assertRaisesRegex(
                FINAL.FinalCsvError,
                "prompt count must be the fixed formal8 count",
            ):
                FINAL._load_protocol_and_matrix(FINAL.SnapshotRegistry())

    def test_output_parent_replacement_after_preflight_is_rejected(self):
        parent = self.root / "publish"
        parent.mkdir()
        output = parent / "final.csv"
        target = self.output_path_impl(output)
        moved = self.root / "publish-moved"
        parent.rename(moved)
        parent.mkdir()
        with self.assertRaisesRegex(
            FINAL.FinalCsvError,
            "output parent changed before publication",
        ):
            self.output_writer_impl(target, b"x\n", ())
        self.assertFalse(output.exists())
        self.assertFalse((moved / "final.csv").exists())


if __name__ == "__main__":
    unittest.main()
