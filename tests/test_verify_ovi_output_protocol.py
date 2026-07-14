import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from ovi.eval_protocol import prompt_sequence_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_ovi_output.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_ovi_output_protocol_test", SCRIPT_PATH
)
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
SPEC.loader.exec_module(VERIFIER)


def environment(prompts, *, samples=1, warmups=1):
    return {
        "measurement_runs": 3,
        "prompt_count": len(prompts),
        "each_example_n_times": samples,
        "expected_measurement_records": 3 * len(prompts) * samples,
        "warmup_runs": warmups,
        "expected_warmup_records": warmups,
        "seed": 103,
        "prompts_sha256": prompt_sequence_sha256(prompts),
    }


def measurements(prompts, *, samples=1):
    return [
        {
            "status": "ok",
            "record_type": "measurement",
            "measurement_index": measurement_index,
            "prompt_index": prompt_index,
            "sample_index": sample_index,
            "prompt": prompt,
            "seed": 103 + sample_index,
        }
        for measurement_index in range(3)
        for prompt_index, prompt in enumerate(prompts)
        for sample_index in range(samples)
    ]


def warmup(prompt, index=0):
    return {
        "status": "ok",
        "record_type": "warmup",
        "benchmark_valid": False,
        "warmup_index": index,
        "prompt": prompt,
        "seed": 103,
    }


def artifact_metrics(path, digest, *, bool_aliases=False):
    return {
        "status": "ok",
        "record_type": "measurement",
        "denoise_seconds": 1.0,
        "total_generation_seconds": 2.0,
        "peak_memory_allocated_bytes": 1,
        "peak_memory_reserved_bytes": 1,
        "generated_video_shape": [3, 1, 64, 64],
        "generated_audio_shape": [1, 80000],
        "actual_video_frame_height_width": [64, 64],
        "output_sha256": digest,
        "output_path": str(path),
        "save_video_seconds": 0.1,
        "artifact_ready_seconds": 0.1,
        "output_hash_seconds": 0.1,
        "measurement_index": 0,
        "prompt_index": 0,
        "sample_index": 0,
        "prompt": "one smoke prompt",
        "seed": 103,
        "benchmark_candidate": 1 if bool_aliases else False,
        "benchmark_valid": False,
        "attention_method": "dense",
        "use_cfg_cache": 0 if bool_aliases else False,
        "cfg_cache_hits": 0,
        "cfg_cache_refreshes": 0,
        "cfg_negative_forwards": 0,
        "expected_cfg_cache_metrics": {
            "cfg_cache_hits": 0,
            "cfg_cache_refreshes": 0,
            "cfg_negative_forwards": 0,
        },
        "use_block_cache": 0 if bool_aliases else False,
        "video_self_attention_dispatcher": {
            "configured_method": "dense",
            "active_method": "dense",
            "fallback_allowed": False,
            "fallback_used": False,
            "fallback_count": 0,
            "calls_total": 1,
            "expected_calls": 1,
            "calls_match_expected": True,
            "errors_by_method": {"dense": 0},
        },
        "gpu_process_monitor": {},
    }


def media_mocks():
    return (
        mock.patch.object(
            VERIFIER,
            "probe",
            return_value={
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 64,
                        "height": 64,
                        "nb_read_frames": "1",
                    },
                    {"codec_type": "audio", "codec_name": "aac"},
                ],
                "format": {"duration": "5.0"},
            },
        ),
        mock.patch.object(
            VERIFIER,
            "decode_audio",
            return_value=VERIFIER.np.ones(80000, dtype=VERIFIER.np.float32),
        ),
        mock.patch.object(
            VERIFIER,
            "decode_video_gray",
            return_value=VERIFIER.np.arange(64 * 64, dtype=VERIFIER.np.uint8),
        ),
    )


def make_protocol_fixture(run_dir):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    def write_json(path, payload):
        path.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path

    run_config = run_dir / "run_config.yaml"
    run_config.write_text("fixture: true\n", encoding="utf-8")
    pre_run = write_json(run_dir / "pre_run_gpu.json", {})
    preflight = write_json(run_dir / "preflight.json", {"errors": []})
    checkpoint = write_json(run_dir / "checkpoint_manifest.json", {})
    freeze = run_dir / "environment.freeze.txt"
    freeze.write_text("fixture\n", encoding="utf-8")

    artifact = run_dir / "smoke.mp4"
    artifact.write_bytes(b"stable protocol artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    metrics = artifact_metrics(artifact, digest)
    metrics_path = write_json(artifact.with_suffix(".metrics.json"), metrics)
    (run_dir / "timings.jsonl").write_text(
        json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "path": str(artifact),
        "sha256": digest,
        "measurement_index": 0,
        "prompt_index": 0,
        "sample_index": 0,
        "prompt": "one smoke prompt",
        "seed": 103,
        "metrics_path": str(metrics_path),
        "artifact_binding": {
            "path": str(artifact),
            "bytes": artifact.stat().st_size,
            "sha256": digest,
        },
        "metrics_binding": {
            "path": str(metrics_path),
            "bytes": metrics_path.stat().st_size,
            "sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        },
        "status": "ok",
        "errors": [],
    }
    environment_payload = {
        "measurement_runs": 1,
        "prompt_count": 1,
        "each_example_n_times": 1,
        "expected_measurement_records": 1,
        "warmup_runs": 0,
        "expected_warmup_records": 0,
        "seed": 103,
        "prompts_sha256": prompt_sequence_sha256(["one smoke prompt"]),
        "attention_method": "dense",
        "benchmark_eligible": False,
        "debug_forward": False,
        "git_dirty": False,
        "use_cfg_cache": False,
        "use_block_cache": False,
        "evidence_file_sha256": {
            "pre_run_gpu.json": VERIFIER.sha256(pre_run),
            "preflight.json": VERIFIER.sha256(preflight),
            "environment.freeze.txt": VERIFIER.sha256(freeze),
            "checkpoint_manifest.json": VERIFIER.sha256(checkpoint),
        },
        "pre_run_gpu_sha256": VERIFIER.sha256(pre_run),
        "run_config_sha256": VERIFIER.sha256(run_config),
    }
    write_json(run_dir / "environment.json", environment_payload)
    return [report], {str(artifact): copy.deepcopy(metrics)}


class MeasurementProtocolTests(unittest.TestCase):
    def setUp(self):
        self.prompts = [f"fixed prompt {index}" for index in range(6)]
        self.environment = environment(self.prompts)
        self.records = measurements(self.prompts)

    def validate(self, records=None, environment_payload=None):
        return VERIFIER.measurement_record_protocol_errors(
            self.records if records is None else records,
            self.environment if environment_payload is None else environment_payload,
        )

    def test_accepts_six_prompts_times_three_measurements(self):
        errors, ordered_prompts = self.validate()
        self.assertEqual(errors, [])
        self.assertEqual(ordered_prompts, self.prompts)
        self.assertEqual(len(self.records), 18)

    def test_accepts_multiple_samples_with_seed_offset(self):
        payload = environment(self.prompts, samples=2)
        records = measurements(self.prompts, samples=2)
        errors, ordered_prompts = self.validate(records, payload)
        self.assertEqual(errors, [])
        self.assertEqual(ordered_prompts, self.prompts)

    def test_duplicate_key_and_missing_cartesian_cell_are_rejected(self):
        records = copy.deepcopy(self.records)
        records[1]["prompt_index"] = 0
        errors, _ = self.validate(records)
        rendered = "; ".join(errors)
        self.assertIn("duplicate measurement/prompt/sample key", rendered)
        self.assertIn("Cartesian product", rendered)

    def test_record_order_is_strict(self):
        records = copy.deepcopy(self.records)
        records[0], records[1] = records[1], records[0]
        errors, _ = self.validate(records)
        self.assertTrue(any("record order" in error for error in errors), errors)

    def test_prompt_text_hash_and_index_binding_are_strict(self):
        records = copy.deepcopy(self.records)
        records[7]["prompt"] = "tampered prompt"
        errors, _ = self.validate(records)
        self.assertTrue(any("changed prompt text" in error for error in errors), errors)

        records = copy.deepcopy(self.records)
        records[0]["prompt"] = "tampered first prompt"
        errors, _ = self.validate(records)
        self.assertTrue(any("prompt hash" in error for error in errors), errors)

    def test_seed_and_json_integer_types_are_strict(self):
        records = copy.deepcopy(self.records)
        records[0]["seed"] = 104
        records[1]["prompt_index"] = True
        errors, _ = self.validate(records)
        rendered = "; ".join(errors)
        self.assertIn("fixed seed", rendered)
        self.assertIn("prompt_index must be a JSON integer", rendered)

    def test_expected_record_count_must_match_product(self):
        payload = dict(self.environment)
        payload["expected_measurement_records"] = 3
        errors, _ = self.validate(environment_payload=payload)
        self.assertTrue(
            any("expected_measurement_records does not equal" in error for error in errors),
            errors,
        )

    def test_warmup_uses_only_first_prompt(self):
        errors, ordered_prompts = self.validate()
        self.assertEqual(errors, [])
        self.assertEqual(
            VERIFIER.warmup_record_protocol_errors(
                [warmup(self.prompts[0])], self.environment, ordered_prompts
            ),
            [],
        )

        bad = warmup(self.prompts[1])
        errors = VERIFIER.warmup_record_protocol_errors(
            [bad], self.environment, ordered_prompts
        )
        self.assertTrue(any("first fixed prompt" in error for error in errors), errors)

    def test_one_prompt_smoke_without_warmup_remains_valid(self):
        prompts = ["one smoke prompt"]
        payload = environment(prompts, warmups=0)
        records = measurements(prompts)
        errors, ordered_prompts = VERIFIER.measurement_record_protocol_errors(
            records, payload
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            VERIFIER.warmup_record_protocol_errors([], payload, ordered_prompts),
            [],
        )


class WarmupSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "warmup_timings.jsonl"
        self.record = warmup("first prompt")

    def write(self, records):
        self.path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_binding_records_canonical_path_bytes_hash_and_count(self):
        self.write([self.record])
        snapshot, records = VERIFIER._snapshot_jsonl(self.path)
        binding = VERIFIER._warmup_timings_binding(snapshot, records)
        self.assertEqual(
            set(binding), {"path", "bytes", "sha256", "record_count"}
        )
        self.assertEqual(binding["path"], str(self.path.resolve()))
        self.assertIs(type(binding["bytes"]), int)
        self.assertEqual(binding["bytes"], len(self.path.read_bytes()))
        self.assertEqual(binding["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertIs(type(binding["record_count"]), int)
        self.assertEqual(binding["record_count"], 1)
        self.assertEqual(
            VERIFIER.warmup_timings_binding_errors(self.path, binding, 1), []
        )

    def test_symlinked_warmup_evidence_is_rejected(self):
        target = self.root / "target.jsonl"
        target.write_text(json.dumps(self.record) + "\n", encoding="utf-8")
        self.path.symlink_to(target)
        with self.assertRaisesRegex(
            VERIFIER.EvidenceSnapshotError, "must not be a symlink"
        ):
            VERIFIER._snapshot_jsonl(self.path)

    def test_replacement_after_snapshot_is_rejected(self):
        self.write([self.record])
        snapshot, _records = VERIFIER._snapshot_jsonl(self.path)
        replacement = self.root / "replacement.jsonl"
        replacement.write_bytes(self.path.read_bytes())
        os.replace(replacement, self.path)
        with self.assertRaisesRegex(
            VERIFIER.EvidenceSnapshotError,
            "changed after its stable byte snapshot",
        ):
            VERIFIER._revalidate_snapshot(snapshot)

    def test_binding_detects_content_replacement(self):
        self.write([self.record])
        snapshot, records = VERIFIER._snapshot_jsonl(self.path)
        binding = VERIFIER._warmup_timings_binding(snapshot, records)
        changed = dict(self.record)
        changed["prompt"] = "changed"
        self.write([changed])
        errors = VERIFIER.warmup_timings_binding_errors(self.path, binding, 1)
        self.assertTrue(any("changed" in error for error in errors), errors)

    def test_binding_rejects_bool_forged_integer_fields(self):
        self.write([self.record])
        snapshot, records = VERIFIER._snapshot_jsonl(self.path)
        binding = VERIFIER._warmup_timings_binding(snapshot, records)
        for field in ("bytes", "record_count"):
            with self.subTest(field=field):
                forged = copy.deepcopy(binding)
                forged[field] = True
                errors = VERIFIER.warmup_timings_binding_errors(
                    self.path, forged, 1
                )
                self.assertTrue(any(field in error for error in errors), errors)

    def test_null_smoke_binding_requires_file_absence(self):
        self.assertEqual(
            VERIFIER.warmup_timings_binding_errors(self.path, None, 0), []
        )
        self.path.write_bytes(b"")
        errors = VERIFIER.warmup_timings_binding_errors(self.path, None, 0)
        self.assertTrue(any("must not create" in error for error in errors), errors)


class StableArtifactCounterexampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()

    def test_timings_binding_is_persistable_and_symlink_is_rejected(self):
        path = self.run_dir / "timings.jsonl"
        path.write_text(json.dumps({"record_type": "measurement"}) + "\n")
        snapshot, records = VERIFIER._snapshot_jsonl(path)
        binding = VERIFIER._jsonl_binding(snapshot, records)
        self.assertEqual(
            set(binding), {"path", "bytes", "sha256", "record_count"}
        )
        self.assertEqual(binding["record_count"], 1)

        target = self.root / "outside-timings.jsonl"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        with self.assertRaisesRegex(
            VERIFIER.EvidenceSnapshotError, "must not be a symlink"
        ):
            VERIFIER._snapshot_jsonl(path)

    def test_cross_binding_rejects_duplicate_and_hash_permutation(self):
        timings = []
        reports = []
        for index in range(2):
            path = self.run_dir / f"artifact-{index}.mp4"
            metrics_path = path.with_suffix(".metrics.json")
            digest = str(index + 1) * 64
            timing = {
                "measurement_index": index,
                "prompt_index": 0,
                "sample_index": 0,
                "prompt": f"prompt {index}",
                "seed": 103,
                "output_path": str(path),
                "output_sha256": digest,
            }
            timings.append(timing)
            reports.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "measurement_index": index,
                    "prompt_index": 0,
                    "sample_index": 0,
                    "prompt": f"prompt {index}",
                    "seed": 103,
                    "metrics_path": str(metrics_path),
                    "artifact_binding": {
                        "path": str(path), "bytes": 1, "sha256": digest,
                    },
                    "metrics_binding": {
                        "path": str(metrics_path),
                        "bytes": 1,
                        "sha256": str(index + 3) * 64,
                    },
                }
            )

        swapped = copy.deepcopy(reports)
        swapped[0]["sha256"], swapped[1]["sha256"] = (
            swapped[1]["sha256"], swapped[0]["sha256"]
        )
        swapped[0]["artifact_binding"]["sha256"] = swapped[0]["sha256"]
        swapped[1]["artifact_binding"]["sha256"] = swapped[1]["sha256"]
        errors = VERIFIER.artifact_report_protocol_errors(
            swapped, timings, self.run_dir
        )
        self.assertTrue(any("same-path timing" in error for error in errors), errors)

        duplicated = copy.deepcopy(reports)
        duplicated[1]["path"] = duplicated[0]["path"]
        duplicated[1]["metrics_path"] = duplicated[0]["metrics_path"]
        duplicated[1]["artifact_binding"]["path"] = duplicated[0]["path"]
        duplicated[1]["metrics_binding"]["path"] = duplicated[0]["metrics_path"]
        errors = VERIFIER.artifact_report_protocol_errors(
            duplicated, timings, self.run_dir
        )
        self.assertTrue(any("duplicate artifact report path" in error for error in errors), errors)

    def test_artifact_and_metrics_symlinks_or_out_of_run_are_rejected(self):
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"outside")
        with self.assertRaisesRegex(
            VERIFIER.EvidenceSnapshotError, "direct child"
        ):
            VERIFIER.verify(outside, run_dir=self.run_dir)

        linked_artifact = self.run_dir / "linked.mp4"
        linked_artifact.symlink_to(outside)
        with self.assertRaisesRegex(
            VERIFIER.EvidenceSnapshotError, "must not be a symlink"
        ):
            VERIFIER.verify(linked_artifact, run_dir=self.run_dir)

        artifact = self.run_dir / "artifact.mp4"
        artifact.write_bytes(b"snapshot media")
        metrics_target = self.root / "outside.metrics.json"
        metrics_target.write_text("{}\n")
        artifact.with_suffix(".metrics.json").symlink_to(metrics_target)
        probe_patch, audio_patch, video_patch = media_mocks()
        with probe_patch, audio_patch, video_patch, self.assertRaisesRegex(
            VERIFIER.EvidenceSnapshotError, "must not be a symlink"
        ):
            VERIFIER.verify(
                artifact, expected_video_frames=1, run_dir=self.run_dir
            )

    def test_final_revalidation_publishes_failed_after_replacement(self):
        evidence_path = self.run_dir / "timings.jsonl"
        evidence_path.write_text("{}\n")
        snapshot = VERIFIER._stable_file_snapshot(evidence_path)
        output_path = self.run_dir / "verification.json"
        output_path.write_text('{"status":"ok"}\n')
        summary = {
            "status": "ok",
            "artifact_count": 0,
            "artifacts": [],
            "benchmark_valid": True,
            "protocol": {"status": "ok", "errors": [], "benchmark_valid": True},
        }
        original_revalidate = VERIFIER._revalidate_snapshot
        replaced = False

        def replace_then_revalidate(candidate):
            nonlocal replaced
            if not replaced:
                replacement = self.run_dir / "replacement.jsonl"
                replacement.write_bytes(evidence_path.read_bytes())
                os.replace(replacement, evidence_path)
                replaced = True
            return original_revalidate(candidate)

        with mock.patch.object(
            VERIFIER, "_revalidate_snapshot", side_effect=replace_then_revalidate
        ):
            published = VERIFIER._publish_verified_summary(
                output_path, summary, [snapshot]
            )
        persisted = json.loads(output_path.read_text())
        self.assertEqual(published["status"], "failed")
        self.assertEqual(persisted["status"], "failed")
        self.assertIs(persisted["benchmark_valid"], False)
        self.assertTrue(persisted["publication_errors"])

    def test_boolean_integer_aliases_are_rejected(self):
        artifact = self.run_dir / "artifact.mp4"
        artifact.write_bytes(b"snapshot media")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        artifact.with_suffix(".metrics.json").write_text(
            json.dumps(artifact_metrics(artifact, digest, bool_aliases=True)) + "\n"
        )
        probe_patch, audio_patch, video_patch = media_mocks()
        with probe_patch, audio_patch, video_patch:
            report = VERIFIER.verify(
                artifact, expected_video_frames=1, run_dir=self.run_dir
            )
        rendered = "; ".join(report["errors"])
        self.assertIn("benchmark_candidate must be a JSON boolean", rendered)
        self.assertIn("use_cfg_cache must be a JSON boolean", rendered)
        self.assertIn("use_block_cache must be a JSON boolean", rendered)


class CompleteEvidencePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def verify_protocol(self, run_dir, reports, metrics_payloads, snapshots=None):
        with (
            mock.patch.object(VERIFIER, "validate_run_protocol"),
            mock.patch.object(
                VERIFIER,
                "validate_pre_run_gpu",
                return_value=(0, "GPU-fixture", "A100"),
            ),
            mock.patch.object(VERIFIER, "validate_gpu_monitor"),
        ):
            return VERIFIER.verify_run_protocol(
                run_dir,
                reports,
                evidence_snapshots=snapshots,
                metrics_payloads=metrics_payloads,
            )

    @staticmethod
    def ok_summary():
        return {
            "status": "ok",
            "artifact_count": 0,
            "artifacts": [],
            "benchmark_valid": False,
            "protocol": {
                "status": "ok",
                "errors": [],
                "benchmark_valid": False,
            },
        }

    def test_all_required_evidence_uses_no_follow_snapshot_and_revalidation(self):
        baseline = self.root / "baseline"
        reports, metrics_payloads = make_protocol_fixture(baseline)
        publication_guards = []
        protocol = self.verify_protocol(
            baseline, reports, metrics_payloads, publication_guards
        )
        self.assertEqual(protocol["status"], "ok", protocol["errors"])
        snapshotted_names = {
            guard.path.name
            for guard in publication_guards
            if isinstance(guard, VERIFIER._StableFileSnapshot)
        }
        self.assertTrue(
            {
                "environment.json",
                "run_config.yaml",
                "pre_run_gpu.json",
                "preflight.json",
                "environment.freeze.txt",
                "checkpoint_manifest.json",
                "timings.jsonl",
            }
            <= snapshotted_names
        )

        symlink_run = self.root / "symlink"
        reports, metrics_payloads = make_protocol_fixture(symlink_run)
        preflight = symlink_run / "preflight.json"
        target = self.root / "preflight-target.json"
        os.replace(preflight, target)
        preflight.symlink_to(target)
        protocol = self.verify_protocol(symlink_run, reports, metrics_payloads)
        self.assertEqual(protocol["status"], "failed")
        self.assertTrue(any("symlink" in error for error in protocol["errors"]))

        replacement_run = self.root / "replacement"
        reports, metrics_payloads = make_protocol_fixture(replacement_run)
        original_snapshot = VERIFIER._stable_file_snapshot
        replaced = False

        def replace_checkpoint_after_snapshot(path):
            nonlocal replaced
            snapshot = original_snapshot(path)
            if Path(path).name == "checkpoint_manifest.json" and not replaced:
                replacement = replacement_run / "checkpoint.replacement"
                replacement.write_bytes(snapshot.data)
                os.replace(replacement, path)
                replaced = True
            return snapshot

        with mock.patch.object(
            VERIFIER,
            "_stable_file_snapshot",
            side_effect=replace_checkpoint_after_snapshot,
        ):
            protocol = self.verify_protocol(
                replacement_run, reports, metrics_payloads
            )
        self.assertTrue(replaced)
        self.assertEqual(protocol["status"], "failed")
        self.assertTrue(
            any("stable snapshot failed" in error for error in protocol["errors"]),
            protocol["errors"],
        )

    def test_success_replace_is_followed_by_full_post_publish_revalidation(self):
        run_dir = self.root / "post-replace"
        run_dir.mkdir()
        evidence_path = run_dir / "environment.json"
        evidence_path.write_text("{}\n")
        snapshot = VERIFIER._stable_file_snapshot(evidence_path)
        output_path = run_dir / "verification.json"
        original_revalidate = VERIFIER._revalidate_publication_guard
        calls = 0

        def drift_on_post_publish(guard):
            nonlocal calls
            calls += 1
            if calls == 2:
                replacement = run_dir / "environment.replacement"
                replacement.write_bytes(evidence_path.read_bytes())
                os.replace(replacement, evidence_path)
            return original_revalidate(guard)

        with mock.patch.object(
            VERIFIER,
            "_revalidate_publication_guard",
            side_effect=drift_on_post_publish,
        ):
            published = VERIFIER._publish_verified_summary(
                output_path, self.ok_summary(), [snapshot]
            )
        persisted = json.loads(output_path.read_text())
        self.assertGreaterEqual(calls, 2)
        self.assertEqual(published["status"], "failed")
        self.assertEqual(persisted["status"], "failed")
        self.assertTrue(
            any("post-publish" in error for error in persisted["publication_errors"])
        )

    def test_zero_warmup_absence_guard_checks_pre_and_post_publish(self):
        pre_dir = self.root / "absence-pre"
        pre_dir.mkdir()
        pre_warmup = pre_dir / "warmup_timings.jsonl"
        pre_guard = VERIFIER._AbsentPathGuard(pre_warmup)
        pre_warmup.write_text("{}\n")
        pre_output = pre_dir / "verification.json"
        published = VERIFIER._publish_verified_summary(
            pre_output, self.ok_summary(), [pre_guard]
        )
        self.assertEqual(published["status"], "failed")
        self.assertTrue(
            any("pre-publish" in error for error in published["publication_errors"])
        )

        post_dir = self.root / "absence-post"
        post_dir.mkdir()
        post_warmup = post_dir / "warmup_timings.jsonl"
        post_guard = VERIFIER._AbsentPathGuard(post_warmup)
        post_output = post_dir / "verification.json"
        original_replace = VERIFIER._replace_json_temp
        created = False

        def create_after_success_replace(temporary_path, output_path):
            nonlocal created
            original_replace(temporary_path, output_path)
            if not created:
                post_warmup.write_text("{}\n")
                created = True

        with mock.patch.object(
            VERIFIER,
            "_replace_json_temp",
            side_effect=create_after_success_replace,
        ):
            published = VERIFIER._publish_verified_summary(
                post_output, self.ok_summary(), [post_guard]
            )
        self.assertTrue(created)
        self.assertEqual(published["status"], "failed")
        self.assertEqual(json.loads(post_output.read_text())["status"], "failed")
        self.assertTrue(
            any("post-publish" in error for error in published["publication_errors"])
        )

    def test_metrics_sidecar_must_strictly_equal_same_path_timing_record(self):
        run_dir = self.root / "metrics-mismatch"
        reports, metrics_payloads = make_protocol_fixture(run_dir)
        artifact_path = reports[0]["path"]
        metrics_payloads[artifact_path]["denoise_seconds"] = 999.0
        protocol = self.verify_protocol(run_dir, reports, metrics_payloads)
        self.assertEqual(protocol["status"], "failed")
        self.assertTrue(
            any(
                "metrics sidecar differs from same-path timing record" in error
                for error in protocol["errors"]
            ),
            protocol["errors"],
        )


class VerifierIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name).resolve()
        self.prompts = [f"fixed prompt {index}" for index in range(6)]

    def write_json(self, name, payload):
        path = self.run_dir / name
        path.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path

    def test_verify_protocol_persists_formal_warmup_binding_for_18_records(self):
        run_config = self.run_dir / "run_config.yaml"
        run_config.write_text("fixture: true\n", encoding="utf-8")
        pre_run = self.write_json("pre_run_gpu.json", {})
        preflight = self.write_json("preflight.json", {"errors": []})
        checkpoint = self.write_json("checkpoint_manifest.json", {})
        freeze = self.run_dir / "environment.freeze.txt"
        freeze.write_text("fixture\n", encoding="utf-8")

        timing_records = measurements(self.prompts)
        reports = []
        metrics_payloads = {}
        for index, record in enumerate(timing_records):
            artifact_path = self.run_dir / f"artifact-{index:02d}.mp4"
            artifact_path.write_bytes(f"artifact-{index}".encode("utf-8"))
            metrics_path = artifact_path.with_suffix(".metrics.json")
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            record.update(
                {
                    "benchmark_candidate": True,
                    "benchmark_valid": False,
                    "output_sha256": digest,
                    "output_path": str(artifact_path),
                    "gpu_process_monitor": {},
                    "use_cfg_cache": False,
                    "use_block_cache": False,
                }
            )
            metrics_path.write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
            )
            metrics_payloads[str(artifact_path)] = copy.deepcopy(record)
            reports.append(
                {
                    "path": str(artifact_path),
                    "sha256": digest,
                    "measurement_index": record["measurement_index"],
                    "prompt_index": record["prompt_index"],
                    "sample_index": record["sample_index"],
                    "prompt": record["prompt"],
                    "seed": record["seed"],
                    "metrics_path": str(metrics_path),
                    "artifact_binding": {
                        "path": str(artifact_path),
                        "bytes": artifact_path.stat().st_size,
                        "sha256": digest,
                    },
                    "metrics_binding": {
                        "path": str(metrics_path),
                        "bytes": metrics_path.stat().st_size,
                        "sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                    },
                    "errors": [],
                }
            )
        (self.run_dir / "timings.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in timing_records),
            encoding="utf-8",
        )
        warmup_record = warmup(self.prompts[0])
        warmup_record.update(
            {
                "benchmark_candidate": True,
                "gpu_process_monitor": {},
                "use_cfg_cache": False,
                "use_block_cache": False,
            }
        )
        warmup_path = self.run_dir / "warmup_timings.jsonl"
        warmup_path.write_text(
            json.dumps(warmup_record, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        payload = {
            **environment(self.prompts),
            "attention_method": "dense",
            "benchmark_eligible": True,
            "debug_forward": False,
            "git_dirty": False,
            "use_cfg_cache": False,
            "use_block_cache": False,
            "evidence_file_sha256": {
                "pre_run_gpu.json": VERIFIER.sha256(pre_run),
                "preflight.json": VERIFIER.sha256(preflight),
                "environment.freeze.txt": VERIFIER.sha256(freeze),
                "checkpoint_manifest.json": VERIFIER.sha256(checkpoint),
            },
            "pre_run_gpu_sha256": VERIFIER.sha256(pre_run),
            "run_config_sha256": VERIFIER.sha256(run_config),
        }
        self.write_json("environment.json", payload)

        with (
            mock.patch.object(VERIFIER, "validate_run_protocol"),
            mock.patch.object(
                VERIFIER,
                "validate_pre_run_gpu",
                return_value=(0, "GPU-fixture", "A100"),
            ),
            mock.patch.object(VERIFIER, "validate_gpu_monitor"),
        ):
            protocol = VERIFIER.verify_run_protocol(
                self.run_dir, reports, metrics_payloads=metrics_payloads
            )

        self.assertEqual(protocol["status"], "ok", protocol["errors"])
        self.assertTrue(protocol["benchmark_valid"])
        self.assertEqual(protocol["observed_measurement_records"], 18)
        self.assertEqual(protocol["observed_warmup_records"], 1)
        timings_path = self.run_dir / "timings.jsonl"
        self.assertEqual(
            protocol["timings_binding"],
            {
                "path": str(timings_path),
                "bytes": timings_path.stat().st_size,
                "sha256": hashlib.sha256(timings_path.read_bytes()).hexdigest(),
                "record_count": 18,
            },
        )
        binding = protocol["warmup_timings_binding"]
        self.assertEqual(
            binding,
            {
                "path": str(warmup_path),
                "bytes": warmup_path.stat().st_size,
                "sha256": hashlib.sha256(warmup_path.read_bytes()).hexdigest(),
                "record_count": 1,
            },
        )

    def test_main_real_one_by_one_by_one_smoke_protocol(self):
        run_config = self.run_dir / "run_config.yaml"
        run_config.write_text("fixture: true\n", encoding="utf-8")
        pre_run = self.write_json("pre_run_gpu.json", {})
        preflight = self.write_json("preflight.json", {"errors": []})
        checkpoint = self.write_json("checkpoint_manifest.json", {})
        freeze = self.run_dir / "environment.freeze.txt"
        freeze.write_text("fixture\n", encoding="utf-8")

        artifact = self.run_dir / "smoke.mp4"
        if VERIFIER.shutil.which("ffmpeg") is None or VERIFIER.shutil.which("ffprobe") is None:
            self.skipTest("real 1x1x1 smoke requires ffmpeg and ffprobe")
        VERIFIER.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=1/5:duration=5",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=880:sample_rate=16000:duration=5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(artifact),
            ]
        )
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        metrics = artifact_metrics(artifact, digest)
        self.write_json("smoke.metrics.json", metrics)
        timings_path = self.run_dir / "timings.jsonl"
        timings_path.write_text(
            json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        payload = {
            "measurement_runs": 1,
            "prompt_count": 1,
            "each_example_n_times": 1,
            "expected_measurement_records": 1,
            "warmup_runs": 0,
            "expected_warmup_records": 0,
            "seed": 103,
            "prompts_sha256": prompt_sequence_sha256(["one smoke prompt"]),
            "attention_method": "dense",
            "benchmark_eligible": False,
            "debug_forward": False,
            "git_dirty": False,
            "use_cfg_cache": False,
            "use_block_cache": False,
            "evidence_file_sha256": {
                "pre_run_gpu.json": VERIFIER.sha256(pre_run),
                "preflight.json": VERIFIER.sha256(preflight),
                "environment.freeze.txt": VERIFIER.sha256(freeze),
                "checkpoint_manifest.json": VERIFIER.sha256(checkpoint),
            },
            "pre_run_gpu_sha256": VERIFIER.sha256(pre_run),
            "run_config_sha256": VERIFIER.sha256(run_config),
        }
        self.write_json("environment.json", payload)

        with (
            mock.patch.object(
                sys, "argv", [str(SCRIPT_PATH), str(self.run_dir), "--expected-video-frames", "1"]
            ),
            mock.patch.object(VERIFIER, "validate_run_protocol"),
            mock.patch.object(
                VERIFIER,
                "validate_pre_run_gpu",
                return_value=(0, "GPU-fixture", "A100"),
            ),
            mock.patch.object(VERIFIER, "validate_gpu_monitor"),
            mock.patch("builtins.print"),
        ):
            exit_code = VERIFIER.main()

        self.assertEqual(exit_code, 0)
        persisted = json.loads((self.run_dir / "verification.json").read_text())
        self.assertEqual(persisted["status"], "ok", persisted)
        self.assertEqual(persisted["artifact_count"], 1)
        self.assertIs(persisted["benchmark_valid"], False)
        self.assertEqual(
            set(persisted["protocol"]["timings_binding"]),
            {"path", "bytes", "sha256", "record_count"},
        )
        self.assertIsNone(persisted["protocol"]["warmup_timings_binding"])
        report = persisted["artifacts"][0]
        self.assertEqual(
            set(report["artifact_binding"]), {"path", "bytes", "sha256"}
        )
        self.assertEqual(
            set(report["metrics_binding"]), {"path", "bytes", "sha256"}
        )
        self.assertFalse(
            {"inode", "device", "mtime_ns", "ctime_ns"}
            & set(report["artifact_binding"])
        )


if __name__ == "__main__":
    unittest.main()
