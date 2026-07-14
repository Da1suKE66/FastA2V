import copy
import csv
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "select_ovi_sparse_winner.py"
SPEC = importlib.util.spec_from_file_location("select_ovi_sparse_winner", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SELECT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SELECT
SPEC.loader.exec_module(SELECT)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tagged(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def make_pinned_protocol(path: Path) -> None:
    protocol = copy.deepcopy(
        json.loads((REPO_ROOT / "configs" / "quality_protocol.json").read_text(
            encoding="utf-8"
        ))
    )
    lpips = protocol["lpips"]
    lpips["trusted_lock_status"] = "pinned"
    lpips["trusted_environment_lock_sha256"] = tagged("environment-lock")
    locked = []
    for package in lpips["packages"]:
        package["trusted_archive_sha256"] = tagged(
            f"archive-{package['distribution']}"
        )
        locked.append({
            "distribution": package["distribution"],
            "version": package["version"],
            "source_index": package["source_index"],
            "archive_url": (
                "https://download.pytorch.org/whl/cpu/"
                if package["distribution"] in {"torch", "torchvision"}
                else "https://files.pythonhosted.org/packages/"
            ) + f"{package['distribution']}.whl",
            "archive_sha256": package["trusted_archive_sha256"],
        })
    lpips["trusted_environment_packages"] = locked
    for weight in lpips["weights"]:
        weight["trusted_sha256"] = tagged(f"weight-{weight['weight_id']}")
    write_json(path, protocol)


class WinnerFixture:
    def __init__(self, root: Path):
        self.root = root
        self.final_csv = root / "final.csv"
        self.output = root / "winner.json"
        self.protocol = json.loads(SELECT.PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.matrix = json.loads(SELECT.MATRIX_PATH.read_text(encoding="utf-8"))
        self.method_map = {
            method["method_id"]: method for method in self.matrix["methods"]
        }
        self.artifact_count = (
            self.matrix["fixed_protocol"]["measurement_runs"]
            * self.matrix["fixed_protocol"]["prompt_count"]
            * self.matrix["fixed_protocol"]["each_example_n_times"]
        )
        self.rows = [
            self.row(method_id, index)
            for index, method_id in enumerate(SELECT.METHOD_IDS)
        ]
        self.write()

    def row(self, method_id: str, index: int) -> dict[str, str]:
        row = {field: "" for field in SELECT.FINAL_FIELDS}
        method = self.method_map[method_id]
        row.update({
            "schema_version": "1",
            "method_id": method_id,
            "label": method["label"],
            "status": "complete",
            "timing_status": "valid",
            "quality_status": "reference" if method_id == "dense" else "complete",
            "manual_review_status": "reference" if method_id == "dense" else "complete",
            "timing_csv_path": str(self.root / "formal-timing.csv"),
            "timing_csv_sha256": tagged("formal-timing"),
            "quality_protocol_id": self.protocol["protocol_id"],
            "quality_protocol_sha256": digest(SELECT.PROTOCOL_PATH),
            "evaluation_matrix_id": self.matrix["matrix_id"],
            "evaluation_matrix_sha256": digest(SELECT.MATRIX_PATH),
            "evaluator_git_commit": "b" * 40,
            "run_dir": str(self.root / "runs" / method_id),
            "run_id": f"formal-{method_id}",
            "git_commit": "b" * 40,
            "verification_sha256": tagged(f"verification-{method_id}"),
            "timings_sha256": tagged(f"timings-{method_id}"),
            "checkpoint_manifest_sha256": tagged("checkpoint-manifest"),
            "checkpoint_fingerprint_sha256": tagged("checkpoint-fingerprint"),
            "gpu_uuid": "GPU-fixture",
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "prompt_set_sha256": self.matrix["fixed_protocol"]["prompts_sha256"],
            "prompt_count": str(self.matrix["fixed_protocol"]["prompt_count"]),
            "seed_count": str(self.matrix["fixed_protocol"]["each_example_n_times"]),
            "seeds": "103;104;105",
            "sample_steps": str(self.matrix["fixed_protocol"]["sample_steps"]),
            "measurement_count": str(self.matrix["fixed_protocol"]["measurement_runs"]),
            "artifact_count": str(self.artifact_count),
            "total_generation_seconds_median": str((60, 50, 40, 30, 20, 25)[index]),
        })
        if method_id != "dense":
            row.update({
                "quality_median_path": str(self.root / "quality" / f"{method_id}.json"),
                "quality_median_sha256": tagged(f"quality-{method_id}"),
                "manual_review_row_count": str(self.artifact_count),
                "manual_pass_count": str(self.artifact_count),
                "manual_fail_count": "0",
                "manual_uncertain_count": "0",
                "manual_validation_path": str(self.root / "manual" / f"{method_id}.json"),
                "manual_validation_sha256": tagged(f"manual-{method_id}"),
                "manual_reviews_csv_path": str(self.root / "manual" / f"{method_id}.csv"),
                "manual_reviews_csv_sha256": tagged(f"manual-csv-{method_id}"),
            })
        return row

    def by_id(self, method_id: str) -> dict[str, str]:
        return next(row for row in self.rows if row["method_id"] == method_id)

    def write(self, *, fields: tuple[str, ...] = SELECT.FINAL_FIELDS) -> None:
        with self.final_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n",
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)


class SelectSparseWinnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.protocol_path = self.root / "quality_protocol.json"
        make_pinned_protocol(self.protocol_path)
        self.protocol_patch = mock.patch.object(
            SELECT, "PROTOCOL_PATH", self.protocol_path
        )
        self.protocol_patch.start()
        self.repository_patch = mock.patch.object(
            SELECT, "_audit_repository", return_value="b" * 40
        )
        self.repository_audit = self.repository_patch.start()
        self.fixture = WinnerFixture(self.root)

    def tearDown(self):
        self.repository_patch.stop()
        self.protocol_patch.stop()
        self.temporary.cleanup()

    def select(self, output: Path | None = None) -> Path:
        return SELECT.select_sparse_winner(
            final_csv=self.fixture.final_csv,
            output=output or self.fixture.output,
        )

    def test_selects_fastest_eligible_candidate_and_binds_all_inputs(self):
        output = self.select()
        receipt = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(receipt["record_type"], "ovi_sparse_winner_selection")
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["winner"]["method_id"], "radial_conservative")
        self.assertEqual(receipt["winner"]["formal_slot"], "E")
        self.assertEqual(
            receipt["winner"]["selected_sparse_profile"], "radial_conservative"
        )
        self.assertEqual(
            receipt["winner"]["profile"],
            SELECT.METHOD_ENVIRONMENT_CONTRACTS["radial_conservative"],
        )
        self.assertEqual(receipt["final_csv"]["sha256"], digest(self.fixture.final_csv))
        self.assertEqual(receipt["quality_protocol"]["sha256"], digest(SELECT.PROTOCOL_PATH))
        self.assertEqual(receipt["evaluation_matrix"]["sha256"], digest(SELECT.MATRIX_PATH))
        self.assertEqual(receipt["selector"]["sha256"], digest(SELECT.SELECTOR_PATH))
        self.assertEqual(receipt["repository"]["head_commit"], "b" * 40)
        self.assertEqual(receipt["trust_model"]["runtime_evidence_root"], "final_csv")
        self.assertEqual(len(receipt["method_bindings"]), 6)
        self.assertEqual(
            [item["method_id"] for item in receipt["candidate_bindings"]],
            list(SELECT.CANDIDATE_METHOD_IDS),
        )
        self.assertTrue(all(item["eligible"] for item in receipt["candidate_bindings"]))
        self.assertEqual(
            self.repository_audit.call_args_list,
            [mock.call(), mock.call(expected_head="b" * 40)],
        )

    def test_exact_tie_uses_lexicographic_method_id(self):
        self.fixture.by_id("radial_aggressive")["total_generation_seconds_median"] = "10"
        self.fixture.by_id("sparge_topk50")["total_generation_seconds_median"] = "10.0"
        self.fixture.write()
        receipt = json.loads(self.select().read_text(encoding="utf-8"))
        self.assertEqual(receipt["winner"]["method_id"], "radial_aggressive")

    def test_fastest_failed_candidate_is_ineligible_but_does_not_block_winner(self):
        row = self.fixture.by_id("radial_aggressive")
        row["total_generation_seconds_median"] = "1"
        row["manual_pass_count"] = str(self.fixture.artifact_count - 1)
        row["manual_fail_count"] = "1"
        self.fixture.write()
        receipt = json.loads(self.select().read_text(encoding="utf-8"))
        self.assertEqual(receipt["winner"]["method_id"], "radial_conservative")
        failed = next(
            item for item in receipt["candidate_bindings"]
            if item["method_id"] == "radial_aggressive"
        )
        self.assertIs(failed["eligible"], False)
        self.assertEqual(
            failed["ineligibility_reason"],
            "one_or_more_manual_reviews_not_pass",
        )

    def test_no_all_pass_candidate_fails_without_output(self):
        for method_id in SELECT.CANDIDATE_METHOD_IDS:
            row = self.fixture.by_id(method_id)
            row["manual_pass_count"] = str(self.fixture.artifact_count - 1)
            row["manual_uncertain_count"] = "1"
        self.fixture.write()
        with self.assertRaisesRegex(SELECT.SelectionError, "no C--F candidate"):
            self.select()
        self.assertFalse(self.fixture.output.exists())

    def test_incomplete_candidate_still_fails_closed(self):
        self.fixture.by_id("sparge_topk75")["quality_status"] = "pending"
        self.fixture.write()
        with self.assertRaisesRegex(SELECT.SelectionError, "quality status"):
            self.select()
        self.assertFalse(self.fixture.output.exists())

    def test_unpinned_protocol_cannot_authorize_a_selection(self):
        protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        protocol["lpips"]["trusted_lock_status"] = "bootstrap_unpinned"
        write_json(self.protocol_path, protocol)
        new_hash = digest(self.protocol_path)
        for row in self.fixture.rows:
            row["quality_protocol_sha256"] = new_hash
        self.fixture.write()
        with self.assertRaisesRegex(SELECT.SelectionError,
                                    "trusted_lock_status must be pinned"):
            self.select()
        self.assertFalse(self.fixture.output.exists())

    def test_any_incomplete_a_to_f_status_fails_closed(self):
        self.fixture.by_id("dense_cfg_cache")["status"] = "pending"
        self.fixture.write()
        with self.assertRaisesRegex(SELECT.SelectionError, "status is not complete"):
            self.select()

    def test_final_commits_must_equal_current_clean_head(self):
        self.fixture.by_id("radial_conservative")["git_commit"] = "c" * 40
        self.fixture.write()
        with self.assertRaisesRegex(SELECT.SelectionError,
                                    "git_commit differs from current clean HEAD"):
            self.select()

    def test_formal8_prompt_and_cardinality_constants_are_hard_bound(self):
        row = self.fixture.by_id("sparge_topk50")
        row["prompt_set_sha256"] = tagged("wrong-prompts")
        self.fixture.write()
        with self.assertRaisesRegex(SELECT.SelectionError, "formal8"):
            self.select()

    def test_manual_counts_must_be_complete_even_for_ineligible_candidate(self):
        row = self.fixture.by_id("sparge_topk50")
        row["manual_pass_count"] = str(self.fixture.artifact_count - 2)
        row["manual_fail_count"] = "1"
        self.fixture.write()
        with self.assertRaisesRegex(SELECT.SelectionError, "counts differ"):
            self.select()

    def test_row_order_and_exact_a_to_f_cardinality_are_required(self):
        self.fixture.rows[2], self.fixture.rows[3] = (
            self.fixture.rows[3], self.fixture.rows[2]
        )
        self.fixture.write()
        with self.assertRaisesRegex(SELECT.SelectionError, "method order"):
            self.select()
        self.fixture.rows.pop()
        self.fixture.write()
        with self.assertRaisesRegex(SELECT.SelectionError, "exactly the A--F"):
            self.select(self.root / "second.json")

    def test_required_selection_field_cannot_disappear(self):
        fields = tuple(
            field for field in SELECT.FINAL_FIELDS
            if field != "total_generation_seconds_median"
        )
        self.fixture.write(fields=fields)
        with self.assertRaisesRegex(SELECT.SelectionError, "omits required"):
            self.select()

    def test_unrelated_new_schema_column_is_hash_bound_and_tolerated(self):
        for row in self.fixture.rows:
            row["future_audit_field"] = f"bound-{row['method_id']}"
        self.fixture.write(fields=SELECT.FINAL_FIELDS + ("future_audit_field",))
        receipt = json.loads(self.select().read_text(encoding="utf-8"))
        self.assertEqual(receipt["final_csv"]["sha256"], digest(self.fixture.final_csv))

    def test_noncanonical_or_nonpositive_ranking_value_is_rejected(self):
        self.fixture.by_id("radial_conservative")[
            "total_generation_seconds_median"
        ] = " 1.0"
        self.fixture.write()
        with self.assertRaisesRegex(SELECT.SelectionError, "canonical decimal"):
            self.select()
        self.fixture.by_id("radial_conservative")[
            "total_generation_seconds_median"
        ] = "0"
        self.fixture.write()
        with self.assertRaisesRegex(SELECT.SelectionError, "finite and positive"):
            self.select(self.root / "second.json")

    def test_symlinked_input_is_rejected(self):
        link = self.root / "final-link.csv"
        link.symlink_to(self.fixture.final_csv)
        with self.assertRaisesRegex(SELECT.SelectionError, "symlink or alias"):
            SELECT.select_sparse_winner(final_csv=link, output=self.fixture.output)
        self.assertFalse(self.fixture.output.exists())

    def test_nonregular_input_and_changed_snapshot_are_rejected(self):
        with self.assertRaisesRegex(SELECT.SelectionError, "not a regular file"):
            SELECT.select_sparse_winner(final_csv=self.root,
                                        output=self.fixture.output)
        snapshot = SELECT._snapshot_file(self.fixture.final_csv, "fixture")
        self.fixture.final_csv.write_bytes(self.fixture.final_csv.read_bytes() + b"\n")
        with self.assertRaisesRegex(SELECT.SelectionError, "changed after"):
            snapshot.revalidate("fixture")

    def test_input_mutation_during_final_repo_audit_prevents_publication(self):
        def audit_side_effect(expected_head=None):
            if expected_head is not None:
                self.fixture.final_csv.write_bytes(
                    self.fixture.final_csv.read_bytes() + b"\n"
                )
            return "b" * 40

        self.repository_audit.side_effect = audit_side_effect
        with self.assertRaisesRegex(SELECT.SelectionError, "changed after"):
            self.select()
        self.assertFalse(self.fixture.output.exists())
        self.assertFalse(any(self.root.glob(".winner.json.*.tmp")))

    def test_repository_source_scan_rejects_bytecode_and_symlinks(self):
        clean = self.root / "clean-repo"
        clean.mkdir()
        (clean / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        SELECT._scan_repository_tree(clean)

        bytecode = self.root / "bytecode-repo"
        (bytecode / "__pycache__").mkdir(parents=True)
        (bytecode / "__pycache__" / "module.pyc").write_bytes(b"pyc")
        with self.assertRaisesRegex(SELECT.SelectionError, "bytecode"):
            SELECT._scan_repository_tree(bytecode)

        linked = self.root / "linked-repo"
        linked.mkdir()
        target = linked / "real.py"
        target.write_text("VALUE = 1\n", encoding="utf-8")
        (linked / "alias.py").symlink_to(target)
        with self.assertRaisesRegex(SELECT.SelectionError, "symlink"):
            SELECT._scan_repository_tree(linked)

    def test_existing_output_symlink_is_not_followed_or_replaced(self):
        target = self.root / "target.json"
        target.write_text("sentinel", encoding="utf-8")
        self.fixture.output.symlink_to(target)
        with self.assertRaisesRegex(SELECT.SelectionError, "refusing to overwrite"):
            self.select()
        self.assertEqual(target.read_text(encoding="utf-8"), "sentinel")

    def test_output_parent_identity_drift_removes_partial_publication(self):
        output = self.root / "parent-drift.json"
        target = SELECT._output_path(output)
        real_lstat = os.lstat
        parent_calls = 0

        def drifting_lstat(path):
            nonlocal parent_calls
            result = real_lstat(path)
            if Path(path) == self.root:
                parent_calls += 1
                if parent_calls == 2:
                    fields = list(result)
                    fields[1] = result.st_ino + 1
                    return os.stat_result(fields)
            return result

        # Keep realpath deterministic so the call counter covers the two
        # explicit parent-identity checks rather than posixpath internals.
        with mock.patch.object(SELECT.os.path, "realpath", side_effect=os.fspath), \
             mock.patch.object(SELECT.os, "lstat", side_effect=drifting_lstat):
            with self.assertRaisesRegex(SELECT.SelectionError,
                                        "parent changed during publication"):
                SELECT._write_atomic_exclusive(
                    target, b"{}\n", (), "b" * 40
                )
        self.assertFalse(output.exists())
        self.assertFalse(any(self.root.glob(".parent-drift.json.*.tmp")))

    def test_output_parent_replacement_between_preflight_and_writer_is_rejected(self):
        parent = self.root / "publish"
        parent.mkdir()
        output = parent / "winner.json"
        target = SELECT._output_path(output)
        moved = self.root / "publish-moved"
        parent.rename(moved)
        parent.mkdir()

        with self.assertRaisesRegex(
            SELECT.SelectionError, "output parent changed before publication"
        ):
            SELECT._write_atomic_exclusive(target, b"{}\n", (), "b" * 40)
        self.assertFalse(output.exists())
        self.assertFalse((moved / "winner.json").exists())
        self.assertFalse(any(moved.glob(".winner.json.*.tmp")))

    def test_existing_output_is_never_overwritten(self):
        self.select()
        original = self.fixture.output.read_bytes()
        with self.assertRaisesRegex(SELECT.SelectionError, "refusing to overwrite"):
            self.select()
        self.assertEqual(self.fixture.output.read_bytes(), original)

    def test_cli_exposes_only_the_two_required_paths(self):
        parser = SELECT._build_parser()
        args = parser.parse_args([
            "--final-csv", str(self.fixture.final_csv),
            "--output", str(self.fixture.output),
        ])
        self.assertEqual(args.final_csv, str(self.fixture.final_csv))
        self.assertEqual(args.output, str(self.fixture.output))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--final-csv", str(self.fixture.final_csv)])


if __name__ == "__main__":
    unittest.main()
