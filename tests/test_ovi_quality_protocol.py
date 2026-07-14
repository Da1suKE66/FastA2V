import base64
import csv
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
import sys
import tempfile
import unittest
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "compare_ovi_quality.py"
SPEC = importlib.util.spec_from_file_location("compare_ovi_quality_test", SCRIPT_PATH)
QUALITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALITY
SPEC.loader.exec_module(QUALITY)

URL_POLICY_PATH = REPO_ROOT / "scripts" / "quality_archive_urls.py"
URL_POLICY_SPEC = importlib.util.spec_from_file_location(
    "quality_archive_urls_test",
    URL_POLICY_PATH,
)
URL_POLICY = importlib.util.module_from_spec(URL_POLICY_SPEC)
sys.modules[URL_POLICY_SPEC.name] = URL_POLICY
URL_POLICY_SPEC.loader.exec_module(URL_POLICY)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class QualityProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol, self.protocol_sha = QUALITY.load_quality_protocol()

    def make_run(
        self,
        method_id,
        label,
        *,
        fixture_prompt_count=1,
        fixture_sample_count=1,
        **overrides,
    ):
        run_dir = (self.root / f"run-{label}").resolve()
        run_dir.mkdir()
        for name in (
            "environment.json",
            "verification.json",
            "timings.jsonl",
            "warmup_timings.jsonl",
            "checkpoint_manifest.json",
        ):
            (run_dir / name).write_text(
                f"{label}-{name}\n",
                encoding="utf-8",
            )
        artifacts = {}
        prompts = tuple(
            f"A fixed audiovisual benchmark prompt {prompt_index}."
            for prompt_index in range(fixture_prompt_count)
        )
        for index, prompt_index, sample_index in (
            (index, prompt_index, sample_index)
            for index in QUALITY.EXPECTED_INDICES
            for prompt_index in range(fixture_prompt_count)
            for sample_index in range(fixture_sample_count)
        ):
            identity = (index, prompt_index, sample_index)
            artifact_path = run_dir / (
                f"measurement-{index}-prompt-{prompt_index}-sample-{sample_index}.mp4"
            )
            artifact_path.write_bytes(f"{label}-artifact-{identity}".encode("utf-8"))
            metrics_path = artifact_path.with_suffix(".metrics.json")
            metrics_path.write_text(
                f"{label}-metrics-{identity}\n",
                encoding="utf-8",
            )
            artifact_key = index if fixture_prompt_count == fixture_sample_count == 1 else identity
            artifacts[artifact_key] = QUALITY.MeasurementArtifact(
                measurement_index=index,
                prompt_index=prompt_index,
                sample_index=sample_index,
                path=artifact_path,
                sha256=digest(artifact_path),
                metrics_sidecar_path=metrics_path,
                metrics_sidecar_sha256=digest(metrics_path),
                prompt=prompts[prompt_index],
                seed=103 + sample_index,
                requested_shape=(720, 720),
                actual_shape=(704, 704),
                generated_video_shape=(3, 121, 704, 704),
                generated_audio_shape=(80000,),
                sample_steps=50,
            )
        (run_dir / "timings.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "measurement_index": artifact.measurement_index,
                        "prompt_index": artifact.prompt_index,
                        "sample_index": artifact.sample_index,
                    },
                    sort_keys=True,
                )
                + "\n"
                for artifact in artifacts.values()
            ),
            encoding="utf-8",
        )
        (run_dir / "warmup_timings.jsonl").write_text(
            json.dumps({"record_type": "warmup", "warmup_index": 0}) + "\n",
            encoding="utf-8",
        )
        preflight_path = run_dir / "preflight.json"
        preflight_path.write_text('{"status":"ok"}\n', encoding="utf-8")
        run_config_path = run_dir / "run_config.yaml"
        run_config_path.write_text("model: ovi\n", encoding="utf-8")
        write_json(
            run_dir / "environment.json",
            {
                "evidence_file_sha256": {
                    "checkpoint_manifest.json": digest(
                        run_dir / "checkpoint_manifest.json"
                    ),
                    "preflight.json": digest(preflight_path),
                },
                "run_config_sha256": digest(run_config_path),
            },
        )
        evidence_bindings = {
            name: {
                "path": str(run_dir / name),
                "bytes": (run_dir / name).stat().st_size,
                "sha256": digest(run_dir / name),
            }
            for name in (
                "environment.json",
                "verification.json",
                "timings.jsonl",
                "warmup_timings.jsonl",
                "checkpoint_manifest.json",
                "preflight.json",
                "run_config.yaml",
            )
        }
        values = {
            "method_id": method_id,
            "run_dir": run_dir,
            "run_id": run_dir.name,
            "verification_sha256": digest(run_dir / "verification.json"),
            "timings_path": run_dir / "timings.jsonl",
            "timings_bytes": (run_dir / "timings.jsonl").stat().st_size,
            "timings_sha256": digest(run_dir / "timings.jsonl"),
            "timings_record_count": len(artifacts),
            "warmup_timings_path": run_dir / "warmup_timings.jsonl",
            "warmup_timings_bytes": (
                run_dir / "warmup_timings.jsonl"
            ).stat().st_size,
            "warmup_timings_sha256": digest(
                run_dir / "warmup_timings.jsonl"
            ),
            "warmup_record_count": 1,
            "environment_sha256": digest(run_dir / "environment.json"),
            "git_commit": "a" * 40,
            "checkpoint_manifest_sha256": digest(
                run_dir / "checkpoint_manifest.json"
            ),
            "checkpoint_fingerprint_sha256": "4" * 64,
            "gpu_identity": (
                0,
                "GPU-11111111-2222-3333-4444-555555555555",
                "NVIDIA A100-SXM4-80GB",
            ),
            "prompt_set_sha256": hashlib.sha256(
                json.dumps(
                    list(prompts),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "prompt_count": fixture_prompt_count,
            "prompts": prompts,
            "base_seed": 103,
            "sample_count": fixture_sample_count,
            "sample_seeds": tuple(
                103 + sample_index
                for sample_index in range(fixture_sample_count)
            ),
            "selected_sparse_profile": "",
            "requested_shape": (720, 720),
            "actual_shape": (704, 704),
            "generated_video_shape": (3, 121, 704, 704),
            "generated_audio_shape": (80000,),
            "sample_steps": 50,
            "acceleration_environment": {
                "run_kind": (
                    "dense_baseline"
                    if method_id == "dense"
                    else "cfg_cache_benchmark"
                ),
                "attention_method": "dense",
                "use_cfg_cache": method_id != "dense",
                "use_block_cache": False,
            },
            "artifacts": artifacts,
            "evidence_bindings": evidence_bindings,
        }
        values.update(overrides)
        return QUALITY.AuditedRun(**values)

    @staticmethod
    def metrics(**overrides):
        payload = {
            "compared_video_frames": 121,
            "video_psnr_db": 41.0,
            "video_ssim": 0.97,
            "reference_audio_samples": 80000,
            "candidate_audio_samples": 80000,
            "audio_sample_count_compared": 80000,
            "audio_rmse": 0.01,
            "audio_max_abs_difference": 0.05,
            "audio_snr_db": 32.0,
            "audio_correlation": 0.94,
            "lpips_alex": 0.12,
            "lpips_frame_count": 121,
        }
        payload.update(overrides)
        return payload

    @staticmethod
    def receipt():
        return {
            "receipt_path": "/cache/liluchen/FastA2V/checkpoints/eval/receipt.json",
            "receipt_sha256": "5" * 64,
            "packages": [],
            "weights": [],
        }

    @staticmethod
    def evaluator_source_receipt():
        paths = {
            "comparison_script": REPO_ROOT / "scripts" / "compare_ovi_quality.py",
            "compare_media_script": REPO_ROOT / "scripts" / "compare_media.py",
            "run_validator_script": REPO_ROOT / "scripts" / "build_ovi_eval_csv.py",
            "archive_url_policy": REPO_ROOT / "scripts" / "quality_archive_urls.py",
            "quality_protocol": REPO_ROOT / "configs" / "quality_protocol.json",
            "evaluation_matrix": REPO_ROOT / "configs" / "ovi_eval_matrix.json",
        }
        return {
            "git_commit": "a" * 40,
            "files": {
                role: {
                    "path": str(path.resolve()),
                    "sha256": digest(path),
                }
                for role, path in paths.items()
            },
        }

    def make_dependency_fixture(self, *, weight_bytes=b"receipted-weight", create_weight=True):
        cache_root = self.root / "cache-root"
        environment_root = cache_root / "envs" / "eval"
        site_packages = (
            environment_root / "lib" / "python3.11" / "site-packages"
        )
        dist_info = site_packages / "fake_dist-1.0.dist-info"
        dist_info.mkdir(parents=True)
        module_file = site_packages / "fake_module.py"
        module_file.write_text("# fake\n", encoding="utf-8")
        record_path = dist_info / "RECORD"
        record_path.write_text("fake_module.py,,\n", encoding="utf-8")
        wheelhouse = cache_root / "checkpoints" / "eval" / "wheels"
        wheelhouse.mkdir(parents=True)
        archive = wheelhouse / "fake_dist-1.0-py3-none-any.whl"
        archive.write_bytes(b"fixed-fake-wheel")
        weight = cache_root / "checkpoints" / "eval" / "weight.pth"
        weight.parent.mkdir(parents=True, exist_ok=True)
        if create_weight:
            weight.write_bytes(weight_bytes)
            trusted_weight_hash = digest(weight)
            weight_size = weight.stat().st_size
        else:
            trusted_weight_hash = "0" * 64
            weight_size = 1
        package = {
            "distribution": "fake-dist",
            "version": "1.0",
            "module": "fake_module",
            "module_path": str(module_file),
            "source_index": "https://pypi.org/simple",
            "trusted_archive_sha256": digest(archive),
        }
        weight_contract = {
            "weight_id": "weight",
            "path": str(weight),
            "source_type": "url",
            "source": "https://example.invalid/weight.pth",
            "trusted_sha256": trusted_weight_hash,
        }
        protocol = {
            "environment_root": str(environment_root),
            "python_executable": os.path.abspath(sys.executable),
            "packages": [package],
            "weights": [weight_contract],
        }
        receipt_path = self.root / f"receipt-{len(list(self.root.glob('receipt-*')))}.json"
        receipt_package = {
            **package,
            "archive_url": "https://files.pythonhosted.org/packages/fake.whl",
            "archive_sha256": digest(archive),
            "archive_path": str(archive),
            "module_sha256": digest(module_file),
            "record_path": str(record_path),
            "record_sha256": digest(record_path),
        }
        lock_payload = [
            {
                key: receipt_package[key]
                for key in (
                    "distribution",
                    "version",
                    "source_index",
                    "archive_url",
                    "archive_sha256",
                )
            }
        ]
        environment_lock = QUALITY.dependency_environment_lock_sha256(lock_payload)
        protocol["trusted_environment_packages"] = lock_payload
        protocol["trusted_environment_lock_sha256"] = environment_lock
        receipt_weight = {
            **weight_contract,
            "bytes": weight_size,
            "sha256": trusted_weight_hash,
        }
        write_json(
            receipt_path,
            {
                "schema_version": 2,
                "environment_root": str(environment_root),
                "python_executable": os.path.abspath(sys.executable),
                "sys_prefix": str(environment_root),
                "python_version": "3.11.9",
                "runtime_contract": {
                    "python_arguments": ["-I", "-S", "-B"],
                    "python_minor": "3.11",
                    "site_packages": str(site_packages.resolve()),
                },
                "environment_lock_sha256": environment_lock,
                "packages": [receipt_package],
                "weights": [receipt_weight],
            },
        )
        kwargs = {
            "receipt_path": receipt_path,
            "import_module": lambda _name: SimpleNamespace(
                __file__=str(module_file)
            ),
            "distribution_version": lambda _name: "1.0",
            "executable": sys.executable,
            "prefix": str(environment_root),
            "runtime_flags": SimpleNamespace(
                isolated=1,
                no_site=1,
                dont_write_bytecode=1,
                no_user_site=1,
                ignore_environment=1,
                safe_path=1,
            ),
            "installed_distributions": lambda _site: {"fake-dist": "1.0"},
            "installed_record_validator": lambda _path, _root: [],
            "wheel_record_validator": lambda _archive, _site, _root: [],
            "site_packages_validator": lambda _packages, _root: [],
            "site_packages_activator": lambda _site: None,
        }
        return protocol, receipt_path, weight, kwargs

    def build(self, dense=None, candidate=None, runner=None):
        dense = dense or self.make_run("dense", "dense")
        candidate = candidate or self.make_run("dense_cfg_cache", "candidate")
        return QUALITY.build_quality_report(
            dense,
            candidate,
            self.protocol,
            protocol_sha256=self.protocol_sha,
            lpips_receipt=self.receipt(),
            media_tool_receipt={"tools": []},
            evaluator_source_receipt=self.evaluator_source_receipt(),
            metric_runner=runner or (lambda _dense, _candidate: self.metrics()),
        )

    def test_fixed_protocol_has_no_sparse_acceptance_threshold_or_manual_values(self):
        self.assertIsNone(
            self.protocol["media_metrics"]["automatic_acceptance_thresholds"]
        )
        self.assertTrue(
            self.protocol["manual_reviews"]["automatic_population_forbidden"]
        )
        template = REPO_ROOT / self.protocol["manual_reviews"]["template"]
        with template.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows, [])
        self.assertIsNone(
            self.protocol["lpips"]["trusted_environment_packages"]
        )

    def test_protocol_rejects_dependency_version_and_source_mutations(self):
        for field, value in (
            ("version", "999.0+cpu"),
            ("source_index", "https://example.invalid/simple"),
            ("module_path", "/tmp/not-the-fixed-module.py"),
        ):
            mutated = copy.deepcopy(self.protocol)
            mutated["lpips"]["packages"][0][field] = value
            path = self.root / f"mutated-{field}.json"
            write_json(path, mutated)
            with self.assertRaisesRegex(QUALITY.QualityError, "package version/module/path/source"):
                QUALITY.load_quality_protocol(path)
        mutated = copy.deepcopy(self.protocol)
        mutated["lpips"]["weights"][0]["source"] = (
            "https://example.invalid/alex.pth"
        )
        path = self.root / "mutated-weight-source.json"
        write_json(path, mutated)
        with self.assertRaisesRegex(QUALITY.QualityError, "weight path/source"):
            QUALITY.load_quality_protocol(path)

    def test_dependency_lock_accepts_official_pytorch_r2_cdn(self):
        archive_url = (
            "https://download-r2.pytorch.org/whl/cpu/"
            "torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl"
        )
        records = QUALITY._dependency_lock_records(
            [
                {
                    "distribution": "torch",
                    "version": "2.6.0+cpu",
                    "source_index": "https://download.pytorch.org/whl/cpu",
                    "archive_url": archive_url,
                    "archive_sha256": "a" * 64,
                }
            ],
            context="test dependency URL policy",
        )
        self.assertEqual(records[0]["archive_url"], archive_url)
        self.assertEqual(
            records[0]["source_index"],
            "https://download.pytorch.org/whl/cpu",
        )

    def test_dependency_lock_rejects_noncanonical_archive_urls(self):
        invalid_urls = {
            "lookalike host": (
                "https://download-r2.pytorch.org.evil.example/whl/cpu/torch.whl"
            ),
            "userinfo": (
                "https://user@download-r2.pytorch.org/whl/cpu/torch.whl"
            ),
            "non-443 port": (
                "https://download-r2.pytorch.org:444/whl/cpu/torch.whl"
            ),
            "non-https scheme": (
                "http://download-r2.pytorch.org/whl/cpu/torch.whl"
            ),
            "query": (
                "https://download-r2.pytorch.org/whl/cpu/torch.whl?mirror=1"
            ),
            "fragment": (
                "https://download-r2.pytorch.org/whl/cpu/torch.whl#sha256=x"
            ),
            "path lookalike": (
                "https://download-r2.pytorch.org/whl/cpu.evil/torch.whl"
            ),
        }
        for label, archive_url in invalid_urls.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                QUALITY.QualityError,
                "archive URL violates the fixed source policy",
            ):
                QUALITY._dependency_lock_records(
                    [
                        {
                            "distribution": "torch",
                            "version": "2.6.0+cpu",
                            "source_index": (
                                "https://download.pytorch.org/whl/cpu"
                            ),
                            "archive_url": archive_url,
                            "archive_sha256": "a" * 64,
                        }
                    ],
                    context="test dependency URL policy",
                )

    def test_dependency_lock_keeps_exact_pythonhosted_rule(self):
        accepted = {
            "distribution": "numpy",
            "version": "1.26.4",
            "source_index": "https://pypi.org/simple",
            "archive_url": (
                "https://files.pythonhosted.org:443/packages/aa/bb/numpy.whl"
            ),
            "archive_sha256": "b" * 64,
        }
        records = QUALITY._dependency_lock_records(
            [accepted],
            context="test PyPI dependency URL policy",
        )
        self.assertEqual(records[0]["source_index"], "https://pypi.org/simple")
        for archive_url in (
            "https://files.pythonhosted.org.evil.example/packages/numpy.whl",
            "https://user@files.pythonhosted.org/packages/numpy.whl",
            "https://files.pythonhosted.org/simple/numpy.whl",
        ):
            with self.subTest(archive_url=archive_url), self.assertRaises(
                QUALITY.QualityError
            ):
                QUALITY._dependency_lock_records(
                    [{**accepted, "archive_url": archive_url}],
                    context="test PyPI dependency URL policy",
                )

    def test_unpinned_bootstrap_hashes_fail_before_receipt_or_score(self):
        with self.assertRaisesRegex(QUALITY.QualityError, "not pinned"):
            QUALITY.validate_lpips_receipt(
                self.protocol["lpips"],
                receipt_path=self.root / "does-not-exist.json",
            )

    def test_lpips_decoder_disables_ffmpeg_frame_duplication(self):
        source = (REPO_ROOT / "scripts" / "compare_ovi_quality.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"-fps_mode",\n                    "passthrough"', source)

    def test_matrix_method_cannot_be_relabelled_as_dense(self):
        matrix = {
            "methods": [
                {
                    "method_id": "sparge_topk50",
                    "implementation_status": "ready",
                    "expected_environment": {
                        "run_kind": "dense_baseline",
                        "attention_method": "dense",
                        "use_cfg_cache": False,
                        "use_block_cache": False,
                    },
                }
            ]
        }
        with self.assertRaisesRegex(QUALITY.QualityError, "relabeled"):
            QUALITY._find_method(matrix, "sparge_topk50")

    def test_quality_matrix_contracts_bind_all_eight_sparse_run_kinds(self):
        matrix = json.loads(
            (REPO_ROOT / "configs" / "ovi_eval_matrix.json").read_text()
        )
        validator = QUALITY._run_validator_module()
        for method_id, run_kinds in validator.COMBO_METHOD_RUN_KINDS.items():
            method = QUALITY._find_method(matrix, method_id)
            self.assertEqual(tuple(method["allowed_run_kinds"]), run_kinds)
            self.assertIs(method["selection_required"], True)
            for run_kind in run_kinds:
                with self.subTest(method_id=method_id, run_kind=run_kind):
                    environment = {
                        "run_kind": run_kind,
                        **validator.SPARSE_COMBO_RUN_KIND_CONTRACTS[run_kind],
                    }
                    self.assertEqual(
                        QUALITY._selected_sparse_profile_for_environment(
                            method,
                            environment,
                            "fixture",
                        ),
                        validator.SPARSE_PROFILE_BY_RUN_KIND[run_kind],
                    )

    def test_quality_rejects_old_generic_dense_combo_contracts(self):
        matrix = json.loads(
            (REPO_ROOT / "configs" / "ovi_eval_matrix.json").read_text()
        )
        old_cfg = copy.deepcopy(matrix)
        cfg = next(
            method
            for method in old_cfg["methods"]
            if method["method_id"] == "best_sparse_cfg"
        )
        cfg["expected_environment"]["run_kind"] = "best_sparse_cfg_benchmark"
        with self.assertRaisesRegex(QUALITY.QualityError, "generic cache contract"):
            QUALITY._find_method(old_cfg, "best_sparse_cfg")

        old_block = copy.deepcopy(matrix)
        block = next(
            method
            for method in old_block["methods"]
            if method["method_id"] == "block_cache"
        )
        block["expected_environment"].update(
            {
                "run_kind": "block_cache_benchmark",
                "attention_method": "dense",
            }
        )
        with self.assertRaisesRegex(QUALITY.QualityError, "generic cache contract"):
            QUALITY._find_method(old_block, "block_cache")

    def test_quality_dense_cache_runs_cannot_masquerade_as_g_or_h(self):
        matrix = json.loads(
            (REPO_ROOT / "configs" / "ovi_eval_matrix.json").read_text()
        )
        cases = (
            (
                "best_sparse_cfg",
                {
                    "run_kind": "cfg_cache_benchmark",
                    "attention_method": "dense",
                    "use_cfg_cache": True,
                    "use_block_cache": False,
                },
            ),
            (
                "block_cache",
                {
                    "run_kind": "block_cache_benchmark",
                    "attention_method": "dense",
                    "block_cache_policy": "fixed",
                    "use_cfg_cache": False,
                    "use_block_cache": True,
                },
            ),
        )
        for method_id, environment in cases:
            with self.subTest(method_id=method_id):
                method = QUALITY._find_method(matrix, method_id)
                with self.assertRaisesRegex(QUALITY.QualityError, "run_kind"):
                    QUALITY._selected_sparse_profile_for_environment(
                        method,
                        environment,
                        "fixture",
                    )

    def test_quality_requires_g_and_h_to_bind_the_same_selected_profile(self):
        cfg = self.make_run(
            "best_sparse_cfg",
            "quality-selection-g",
            selected_sparse_profile="sparge_topk50",
        )
        block = self.make_run(
            "block_cache",
            "quality-selection-h",
            selected_sparse_profile="sparge_topk50",
        )
        self.assertEqual(
            QUALITY.validate_selected_sparse_profile_consistency((cfg, block)),
            "sparge_topk50",
        )
        mismatched = QUALITY.AuditedRun(
            **{
                **block.__dict__,
                "selected_sparse_profile": "radial_conservative",
            }
        )
        with self.assertRaisesRegex(
            QUALITY.QualityError,
            "selected different sparse profiles",
        ):
            QUALITY.validate_selected_sparse_profile_consistency(
                (cfg, mismatched)
            )

    def test_builds_three_index_pairs_and_numeric_medians_without_acceptance_claim(self):
        values = iter((0.2, 0.1, 0.3))

        def runner(_dense, _candidate):
            return self.metrics(lpips_alex=next(values))

        report = self.build(runner=runner)
        self.assertEqual(
            [pair["measurement_index"] for pair in report["pairs"]],
            [0, 1, 2],
        )
        self.assertAlmostEqual(report["metric_medians"]["lpips_alex"], 0.2)
        self.assertIsNone(report["automatic_acceptance"])
        self.assertEqual(report["manual_review"]["status"], "not_provided")
        for pair in report["pairs"]:
            self.assertEqual(
                pair["dense"]["git_commit"], pair["candidate"]["git_commit"]
            )
            self.assertEqual(
                pair["dense"]["checkpoint_fingerprint_sha256"],
                pair["candidate"]["checkpoint_fingerprint_sha256"],
            )
            self.assertIsNone(pair["automatic_acceptance"])

    def test_builds_eighteen_identity_pairs_for_six_prompt_matrix(self):
        dense = self.make_run(
            "dense",
            "dense-six",
            fixture_prompt_count=6,
        )
        candidate = self.make_run(
            "dense_cfg_cache",
            "candidate-six",
            fixture_prompt_count=6,
        )
        lpips_values = iter(float(value) for value in range(18))

        report = self.build(
            dense=dense,
            candidate=candidate,
            runner=lambda _dense, _candidate: self.metrics(
                lpips_alex=next(lpips_values)
            ),
        )

        expected_identities = [
            (measurement_index, prompt_index, 0)
            for measurement_index in QUALITY.EXPECTED_MEASUREMENT_INDICES
            for prompt_index in range(6)
        ]
        self.assertEqual(report["pair_count"], 18)
        self.assertEqual(
            [
                (
                    pair["measurement_index"],
                    pair["prompt_index"],
                    pair["sample_index"],
                )
                for pair in report["pairs"]
            ],
            expected_identities,
        )
        self.assertEqual(report["metric_medians"]["lpips_alex"], 8.5)

    def test_quality_pairing_rejects_prompt_identity_relabel(self):
        dense = self.make_run(
            "dense",
            "dense-six-relabel",
            fixture_prompt_count=6,
        )
        candidate = self.make_run(
            "dense_cfg_cache",
            "candidate-six-relabel",
            fixture_prompt_count=6,
        )
        artifacts = dict(candidate.artifacts)
        identity = (0, 1, 0)
        artifacts[identity] = QUALITY.MeasurementArtifact(
            **{**artifacts[identity].__dict__, "prompt": "relabelled prompt"}
        )
        candidate = QUALITY.AuditedRun(
            **{**candidate.__dict__, "artifacts": artifacts}
        )
        with self.assertRaisesRegex(QUALITY.QualityError, "prompt differs"):
            self.build(dense=dense, candidate=candidate)

    def test_legacy_quality_protocol_schema_is_rejected(self):
        legacy = copy.deepcopy(self.protocol)
        legacy["schema_version"] = 1
        legacy["protocol_id"] = "ovi_720x720_5s_dense_pair_quality_v1"
        path = self.root / "legacy-quality-protocol.json"
        write_json(path, legacy)
        with self.assertRaisesRegex(QUALITY.QualityError, "schema_version"):
            QUALITY.load_quality_protocol(path)

    def test_artifact_hash_drift_during_metric_execution_is_rejected(self):
        dense = self.make_run("dense", "dense")
        candidate = self.make_run("dense_cfg_cache", "candidate")

        def mutating_runner(_dense, candidate_artifact):
            candidate_artifact.path.write_bytes(b"changed-during-metrics")
            return self.metrics()

        with self.assertRaisesRegex(QUALITY.QualityError, "after metrics.*SHA256 drift"):
            self.build(dense=dense, candidate=candidate, runner=mutating_runner)

    def test_earlier_artifact_drift_during_later_pair_is_rejected(self):
        dense = self.make_run("dense", "dense")
        candidate = self.make_run("dense_cfg_cache", "candidate")
        calls = 0

        def mutating_late_runner(_dense, _candidate):
            nonlocal calls
            calls += 1
            if calls == 3:
                dense.artifacts[0].path.write_bytes(b"late-drift")
            return self.metrics()

        with self.assertRaisesRegex(
            QUALITY.QualityError, "after all metrics.*SHA256 drift"
        ):
            self.build(
                dense=dense,
                candidate=candidate,
                runner=mutating_late_runner,
            )

    def test_preflight_drift_after_run_load_is_rejected(self):
        run = self.make_run("dense_cfg_cache", "preflight-drift")
        preflight = run.run_dir / "preflight.json"
        preflight.write_text('{"status":"ok"}\n', encoding="utf-8")
        bindings = dict(run.evidence_bindings)
        bindings["preflight.json"] = {
            "path": str(preflight),
            "bytes": preflight.stat().st_size,
            "sha256": digest(preflight),
        }
        guarded = QUALITY.AuditedRun(
            **{**run.__dict__, "evidence_bindings": bindings}
        )
        preflight.write_text('{"status":"forged"}\n', encoding="utf-8")
        with self.assertRaisesRegex(
            QUALITY.QualityError,
            "preflight.json evidence bytes or SHA256 drifted",
        ):
            QUALITY._assert_run_evidence(guarded, "after metrics")

    def test_persisted_binding_covers_environment_originals_and_run_config(self):
        for filename, expected_message in (
            (
                "preflight.json",
                "evidence_bindings omits environment evidence preflight.json",
            ),
            ("run_config.yaml", "evidence_bindings omits run_config.yaml"),
        ):
            with self.subTest(filename=filename):
                run = self.make_run(
                    "dense_cfg_cache", f"persisted-omits-{filename}"
                )
                binding = run.sidecar_binding()
                del binding["evidence_bindings"][filename]
                with self.assertRaisesRegex(
                    QUALITY.QualityError, expected_message
                ):
                    QUALITY._validate_persisted_run_binding(
                        binding, "candidate"
                    )

    def test_persisted_binding_rejects_bound_original_drift(self):
        for filename in ("preflight.json", "run_config.yaml"):
            with self.subTest(filename=filename):
                run = self.make_run(
                    "dense_cfg_cache", f"persisted-drift-{filename}"
                )
                binding = run.sidecar_binding()
                (run.run_dir / filename).write_text(
                    "forged persisted original\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    QUALITY.QualityError,
                    rf"{re.escape(filename)} evidence bytes or SHA256 drifted",
                ):
                    QUALITY._validate_persisted_run_binding(
                        binding, "candidate"
                    )

    def test_persisted_binding_cross_checks_environment_hashes(self):
        for filename, expected_message in (
            (
                "preflight.json",
                "preflight.json differs from bound environment evidence hash",
            ),
            (
                "run_config.yaml",
                "run_config.yaml differs from bound environment hash",
            ),
        ):
            with self.subTest(filename=filename):
                run = self.make_run(
                    "dense_cfg_cache", f"persisted-rehash-{filename}"
                )
                binding = run.sidecar_binding()
                path = run.run_dir / filename
                path.write_text(
                    "forged and locally rebound original\n", encoding="utf-8"
                )
                binding["evidence_bindings"][filename] = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
                with self.assertRaisesRegex(
                    QUALITY.QualityError, expected_message
                ):
                    QUALITY._validate_persisted_run_binding(
                        binding, "candidate"
                    )

    def test_non_string_evidence_binding_name_is_rejected_cleanly(self):
        run = self.make_run("dense_cfg_cache", "bad-evidence-name")
        bindings = dict(run.evidence_bindings)
        bindings[1] = bindings["environment.json"]
        guarded = QUALITY.AuditedRun(
            **{**run.__dict__, "evidence_bindings": bindings}
        )
        with self.assertRaisesRegex(
            QUALITY.QualityError,
            "invalid evidence binding filename",
        ):
            QUALITY._assert_run_evidence(guarded, "after metrics")

    def test_nan_metric_is_rejected_instead_of_becoming_zero(self):
        with self.assertRaisesRegex(QUALITY.QualityError, "lpips_alex must be finite"):
            self.build(
                runner=lambda _dense, _candidate: self.metrics(lpips_alex=float("nan"))
            )

    def test_infinite_exact_match_psnr_uses_explicit_string_sentinel(self):
        report = self.build(
            runner=lambda _dense, _candidate: self.metrics(video_psnr_db=float("inf"))
        )
        self.assertEqual(report["pairs"][0]["metrics"]["video_psnr_db"], "inf")
        self.assertEqual(report["metric_medians"]["video_psnr_db"], "inf")

    def test_wrong_measurement_pair_is_rejected(self):
        dense = self.make_run("dense", "dense")
        candidate = self.make_run("dense_cfg_cache", "candidate")
        wrong = dict(candidate.artifacts)
        item = wrong[0]
        wrong[0] = QUALITY.MeasurementArtifact(
            **{**item.__dict__, "measurement_index": 1}
        )
        candidate = QUALITY.AuditedRun(
            **{**candidate.__dict__, "artifacts": wrong}
        )
        with self.assertRaisesRegex(QUALITY.QualityError, "identity"):
            self.build(dense=dense, candidate=candidate)

    def test_cross_protocol_commit_is_rejected(self):
        dense = self.make_run("dense", "dense")
        candidate = self.make_run(
            "dense_cfg_cache", "candidate", git_commit="b" * 40
        )
        with self.assertRaisesRegex(QUALITY.QualityError, "git_commit"):
            self.build(dense=dense, candidate=candidate)

    def test_run_commit_must_match_hash_bound_evaluator_commit(self):
        dense = self.make_run("dense", "dense", git_commit="c" * 40)
        candidate = self.make_run(
            "dense_cfg_cache",
            "candidate",
            git_commit="c" * 40,
        )
        with self.assertRaisesRegex(
            QUALITY.QualityError,
            "hash-bound evaluator commit",
        ):
            self.build(dense=dense, candidate=candidate)

    def test_pair_and_median_sidecars_bind_hashes(self):
        report = self.build()
        output_dir = self.root / "quality-output"
        median_path = QUALITY.write_quality_sidecars(report, output_dir)
        self.assertTrue(median_path.is_file())
        median = json.loads(median_path.read_text(encoding="utf-8"))
        self.assertEqual(median["record_type"], "ovi_quality_median")
        self.assertEqual(len(median["pairs"]), 3)
        for binding in median["pairs"]:
            pair_path = Path(binding["pair_sidecar_path"])
            self.assertEqual(binding["pair_sidecar_sha256"], digest(pair_path))
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
            self.assertEqual(
                binding["candidate_artifact_sha256"],
                pair["candidate"]["artifact_sha256"],
            )
        with self.assertRaisesRegex(QUALITY.QualityError, "refusing to overwrite"):
            QUALITY.write_quality_sidecars(report, output_dir)

    def test_six_prompt_sidecars_and_manual_rows_bind_all_eighteen_identities(self):
        dense = self.make_run(
            "dense",
            "dense-six-loop",
            fixture_prompt_count=6,
        )
        candidate = self.make_run(
            "dense_cfg_cache",
            "candidate-six-loop",
            fixture_prompt_count=6,
        )
        report = self.build(dense=dense, candidate=candidate)
        median_path = QUALITY.write_quality_sidecars(
            report,
            self.root / "quality-six-loop",
        )
        median = json.loads(median_path.read_text(encoding="utf-8"))
        self.assertEqual(median["pair_count"], 18)
        self.assertEqual(len(median["pairs"]), 18)
        bindings = QUALITY._expected_manual_bindings_from_report(median)
        self.assertEqual(len(bindings), 18)

        manual = self.root / "manual-six-loop.csv"
        with manual.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QUALITY.MANUAL_FIELDS)
            writer.writeheader()
            for identity in sorted(bindings):
                writer.writerow(
                    {
                        "measurement_index": identity[0],
                        "prompt_index": identity[1],
                        "sample_index": identity[2],
                        "dense_artifact_sha256": bindings[identity][0],
                        "candidate_artifact_sha256": bindings[identity][1],
                        "reviewer": "human-reviewer",
                        "reviewed_at_utc": "2026-07-14T12:00:00Z",
                        "sync_rating": "pass",
                        "notes": "Reviewed exact identity pair.",
                    }
                )
        status = QUALITY.validate_manual_reviews(
            manual,
            bindings,
            self.protocol["manual_reviews"],
        )
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["row_count"], 18)

    def test_quality_evidence_rejects_symlinked_warmup_snapshot(self):
        dense = self.make_run("dense", "dense-warmup-symlink")
        candidate = self.make_run("dense_cfg_cache", "candidate-warmup-symlink")
        target = dense.run_dir / "warmup-target.jsonl"
        target.write_bytes(dense.warmup_timings_path.read_bytes())
        dense.warmup_timings_path.unlink()
        dense.warmup_timings_path.symlink_to(target)
        with self.assertRaisesRegex(QUALITY.QualityError, "symlink"):
            self.build(dense=dense, candidate=candidate)

    def test_manual_template_is_valid_but_pending_and_never_autofilled(self):
        dense = self.make_run("dense", "dense")
        candidate = self.make_run("dense_cfg_cache", "candidate")
        bindings = {
            index: (dense.artifacts[index].sha256, candidate.artifacts[index].sha256)
            for index in QUALITY.EXPECTED_INDICES
        }
        status = QUALITY.validate_manual_reviews(
            REPO_ROOT / "eval" / "manual_sync_reviews.csv",
            bindings,
            self.protocol["manual_reviews"],
        )
        self.assertEqual(status["status"], "empty")
        self.assertEqual(status["row_count"], 0)

    def test_manual_candidate_hash_mismatch_is_rejected(self):
        dense = self.make_run("dense", "dense")
        candidate = self.make_run("dense_cfg_cache", "candidate")
        bindings = {
            index: (dense.artifacts[index].sha256, candidate.artifacts[index].sha256)
            for index in QUALITY.EXPECTED_INDICES
        }
        manual = self.root / "manual.csv"
        with manual.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QUALITY.MANUAL_FIELDS)
            writer.writeheader()
            for index in QUALITY.EXPECTED_INDICES:
                writer.writerow(
                    {
                        "measurement_index": index,
                        "prompt_index": 0,
                        "sample_index": 0,
                        "dense_artifact_sha256": bindings[index][0],
                        "candidate_artifact_sha256": (
                            "f" * 64 if index == 1 else bindings[index][1]
                        ),
                        "reviewer": "human-reviewer",
                        "reviewed_at_utc": "2026-07-14T12:00:00Z",
                        "sync_rating": "pass",
                        "notes": "Reviewed audio/video synchronization manually.",
                    }
                )
        with self.assertRaisesRegex(QUALITY.QualityError, "candidate artifact hash"):
            QUALITY.validate_manual_reviews(
                manual, bindings, self.protocol["manual_reviews"]
            )

    def test_persisted_median_sidecar_drives_complete_manual_validation(self):
        report = self.build()
        median_path = QUALITY.write_quality_sidecars(
            report, self.root / "persisted-quality"
        )
        median = json.loads(median_path.read_text(encoding="utf-8"))
        bindings = QUALITY._expected_manual_bindings_from_report(median)
        manual = self.root / "complete-manual.csv"
        with manual.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QUALITY.MANUAL_FIELDS)
            writer.writeheader()
            for identity in sorted(bindings):
                writer.writerow(
                    {
                        "measurement_index": identity[0],
                        "prompt_index": identity[1],
                        "sample_index": identity[2],
                        "dense_artifact_sha256": bindings[identity][0],
                        "candidate_artifact_sha256": bindings[identity][1],
                        "reviewer": "human-reviewer",
                        "reviewed_at_utc": "2026-07-14T12:00:00Z",
                        "sync_rating": "uncertain" if identity[0] == 2 else "pass",
                        "notes": "Human-authored synchronization review.",
                    }
                )
        status = QUALITY.validate_manual_reviews(
            manual, bindings, self.protocol["manual_reviews"]
        )
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["row_count"], 3)
        validation_path = QUALITY.write_manual_validation_receipt(
            median_path.parent / "manual-review.validation.json",
            median_path=median_path,
            median_sha256=digest(median_path),
            manual_status=status,
            expected_bindings=bindings,
            protocol=self.protocol,
            protocol_sha256=self.protocol_sha,
        )
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        self.assertEqual(validation["quality_median_sha256"], digest(median_path))
        self.assertEqual(validation["manual_reviews_csv_sha256"], digest(manual))
        self.assertEqual(len(validation["pairs"]), 3)

    def test_forged_inline_three_pair_report_is_not_a_manual_trust_root(self):
        forged = {
            "schema_version": QUALITY.QUALITY_SCHEMA_VERSION,
            "record_type": "ovi_quality_report",
            "pairs": [
                {
                    "measurement_index": index,
                    "dense": {"artifact_sha256": "1" * 64},
                    "candidate": {"artifact_sha256": "2" * 64},
                }
                for index in QUALITY.EXPECTED_INDICES
            ],
        }
        with self.assertRaisesRegex(QUALITY.QualityError, "persisted ovi_quality_median"):
            QUALITY._expected_manual_bindings_from_report(forged)

    def test_complete_dependency_receipt_binds_module_version_weight_and_source(self):
        protocol, receipt_path, weight, kwargs = self.make_dependency_fixture()
        validated = QUALITY.validate_lpips_receipt(protocol, **kwargs)
        self.assertEqual(validated["receipt_sha256"], digest(receipt_path))
        self.assertEqual(validated["weights"][0]["sha256"], digest(weight))
        self.assertEqual(
            validated["weights"][0]["source"],
            "https://example.invalid/weight.pth",
        )

    def test_complete_environment_payload_rejects_receipt_source_drift(self):
        protocol, receipt_path, _weight, kwargs = self.make_dependency_fixture()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["packages"][0]["archive_url"] = (
            "https://files.pythonhosted.org/packages/tampered/fake.whl"
        )
        receipt["environment_lock_sha256"] = (
            QUALITY.dependency_environment_lock_sha256(receipt["packages"])
        )
        write_json(receipt_path, receipt)
        with self.assertRaisesRegex(
            QUALITY.QualityError,
            "environment lock differs|package payload differs",
        ):
            QUALITY.validate_lpips_receipt(protocol, **kwargs)

    def test_extra_installed_distribution_is_rejected(self):
        protocol, _receipt_path, _weight, kwargs = self.make_dependency_fixture()
        kwargs["installed_distributions"] = lambda _site: {
            "fake-dist": "1.0",
            "injected-package": "9.9",
        }
        with self.assertRaisesRegex(
            QUALITY.QualityError,
            "installed distribution set differs",
        ):
            QUALITY.validate_lpips_receipt(protocol, **kwargs)

    def test_runtime_must_use_isolated_no_site_no_bytecode_flags(self):
        protocol, _receipt_path, _weight, kwargs = self.make_dependency_fixture()
        kwargs["runtime_flags"] = SimpleNamespace(
            isolated=1,
            no_site=0,
            dont_write_bytecode=1,
            no_user_site=1,
            ignore_environment=1,
            safe_path=1,
        )
        with self.assertRaisesRegex(QUALITY.QualityError, "no_site must be enabled"):
            QUALITY.validate_lpips_receipt(protocol, **kwargs)

    def test_unhashed_generated_pyc_must_be_absent(self):
        environment_root = self.root / "record-env"
        site_packages = (
            environment_root / "lib" / "python3.11" / "site-packages"
        )
        record = site_packages / "fake-1.0.dist-info" / "RECORD"
        record.parent.mkdir(parents=True)
        pyc = site_packages / "fake" / "__pycache__" / "module.cpython-311.pyc"
        record.write_text(
            "fake/__pycache__/module.cpython-311.pyc,,\n"
            "fake-1.0.dist-info/RECORD,,\n",
            encoding="utf-8",
        )
        self.assertEqual(
            QUALITY._distribution_record_errors(record, environment_root.resolve()),
            [],
        )
        pyc.parent.mkdir(parents=True)
        pyc.write_bytes(b"untrusted-bytecode")
        self.assertTrue(
            any(
                "unhashed generated bytecode must be absent" in error
                for error in QUALITY._distribution_record_errors(
                    record, environment_root.resolve()
                )
            )
        )

    def test_actual_pip_no_compile_record_is_verifiable(self):
        environment_root = self.root / "pip-record-env"
        site_packages = (
            environment_root / "lib" / "python3.11" / "site-packages"
        )
        wheelhouse = self.root / "record-wheelhouse"
        wheelhouse.mkdir()
        wheel = wheelhouse / "record_probe-1.0-py3-none-any.whl"
        files = {
            "record_probe/__init__.py": b"VALUE = 1\n",
            "record_probe-1.0.dist-info/METADATA": (
                b"Metadata-Version: 2.1\nName: record-probe\nVersion: 1.0\n"
            ),
            "record_probe-1.0.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: FastA2V-test\n"
                b"Root-Is-Purelib: true\nTag: py3-none-any\n"
            ),
        }
        record_lines = []
        for name, content in files.items():
            encoded = base64.urlsafe_b64encode(
                hashlib.sha256(content).digest()
            ).decode("ascii").rstrip("=")
            record_lines.append(f"{name},sha256={encoded},{len(content)}")
        record_name = "record_probe-1.0.dist-info/RECORD"
        files[record_name] = (
            "\n".join([*record_lines, f"{record_name},,"]) + "\n"
        ).encode("utf-8")
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--no-compile",
                "--target",
                str(site_packages),
                "--find-links",
                str(wheelhouse),
                "record-probe==1.0",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        installed_record = site_packages / "record_probe-1.0.dist-info" / "RECORD"
        self.assertEqual(
            QUALITY._distribution_record_errors(
                installed_record,
                environment_root.resolve(),
            ),
            [],
        )
        self.assertEqual(
            QUALITY._site_packages_tree_errors(
                [
                    {
                        "archive_path": str(wheel),
                        "record_path": str(installed_record),
                    }
                ],
                environment_root.resolve(),
            ),
            [],
        )

    def test_run_validator_source_is_hash_bound(self):
        receipt = self.evaluator_source_receipt()
        QUALITY.validate_evaluator_source_receipt(receipt)
        receipt["files"]["run_validator_script"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(QUALITY.QualityError, "run_validator_script"):
            QUALITY.validate_evaluator_source_receipt(receipt)

    def test_archive_url_policy_source_is_hash_bound(self):
        receipt = self.evaluator_source_receipt()
        QUALITY.validate_evaluator_source_receipt(receipt)
        receipt["files"]["archive_url_policy"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(QUALITY.QualityError, "archive_url_policy"):
            QUALITY.validate_evaluator_source_receipt(receipt)

    def test_manual_validation_rejects_tampered_median_metrics(self):
        report = self.build()
        median_path = QUALITY.write_quality_sidecars(
            report, self.root / "tampered-median"
        )
        median = json.loads(median_path.read_text(encoding="utf-8"))
        median["metric_medians"]["lpips_alex"] = 0.999
        write_json(median_path, median)
        with self.assertRaisesRegex(
            QUALITY.QualityError,
            "median metrics differ",
        ):
            QUALITY._expected_manual_bindings_from_report(median)

    def test_installer_has_bootstrap_and_offline_pinned_paths(self):
        source = (REPO_ROOT / "scripts" / "install_ovi_quality_env.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("trusted_environment_packages", source)
        self.assertIn("from pip._vendor.packaging.utils", source)
        self.assertIn('--no-index \\\n', source)
        self.assertIn('--find-links "${WHEELHOUSE}"', source)
        self.assertIn("--no-compile", source)
        self.assertIn('EVAL_QUALITY_URL_POLICY="${QUALITY_URL_POLICY}"', source)
        self.assertIn('"classify_dependency_archive_url"', source)
        self.assertIn('url_policy["validate_dependency_archive_url"]', source)
        isolated_pip = (
            'PIP_CONFIG_FILE=/dev/null \\\n'
            '  "${EVAL_ENV}/bin/python" -I -B -m pip --isolated install \\\n'
        )
        self.assertEqual(source.count(isolated_pip), 4)
        self.assertEqual(source.count('--cache-dir "${PIP_CACHE_DIR}"'), 4)
        self.assertIn('PIP_NETWORK_TIMEOUT_SECONDS="300"', source)
        self.assertIn('PIP_NETWORK_RETRIES="10"', source)
        self.assertEqual(
            source.count('--timeout "${PIP_NETWORK_TIMEOUT_SECONDS}"'),
            4,
        )
        self.assertEqual(
            source.count('--retries "${PIP_NETWORK_RETRIES}"'),
            4,
        )
        self.assertEqual(source.count("--disable-pip-version-check"), 5)
        self.assertEqual(source.count("--no-input"), 5)
        self.assertNotIn("-m pip install", source)
        self.assertNotIn("export PIP_CACHE_DIR", source)
        self.assertIn('download_environment["PIP_CONFIG_FILE"] = os.devnull', source)
        self.assertIn('"--isolated",\n        "download",', source)
        self.assertIn('"--resume-retries",', source)
        self.assertIn('"--no-deps",\n        "--only-binary=:all:",\n        "--no-index",', source)
        self.assertIn("materialized wheelhouse differs from the exact pip reports", source)
        self.assertIn("materialized wheel hash differs from pip report", source)
        self.assertNotIn(
            'archive_url.startswith("https://download.pytorch.org/")',
            source,
        )

    def test_exact_wheelhouse_rejects_symlink_entry(self):
        wheelhouse = self.root / "exact-wheelhouse-symlink"
        wheelhouse.mkdir()
        expected = wheelhouse / "expected-1.0-py3-none-any.whl"
        expected.write_bytes(b"expected-wheel")
        (wheelhouse / "extra-link.whl").symlink_to(expected.name)

        with self.assertRaisesRegex(
            ValueError,
            "non-regular or symlink top-level entries",
        ):
            URL_POLICY.validate_exact_wheelhouse(wheelhouse, {expected})

    def test_exact_wheelhouse_rejects_directory_entry(self):
        wheelhouse = self.root / "exact-wheelhouse-directory"
        wheelhouse.mkdir()
        expected = wheelhouse / "expected-1.0-py3-none-any.whl"
        expected.write_bytes(b"expected-wheel")
        (wheelhouse / "extra-directory").mkdir()

        with self.assertRaisesRegex(
            ValueError,
            "non-regular or symlink top-level entries",
        ):
            URL_POLICY.validate_exact_wheelhouse(wheelhouse, {expected})

    def test_pip_dual_isolation_ignores_synthetic_ambient_configuration(self):
        pip_config = self.root / "synthetic-pip.conf"
        config_index = "https://config-index.invalid/simple"
        environment_index = "https://environment-index.invalid/simple"
        pip_config.write_text(
            "[global]\n"
            f"extra-index-url = {config_index}\n"
            f"find-links = {self.root / 'config-wheels'}\n",
            encoding="utf-8",
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PIP_")
        }
        environment.update(
            {
                "PIP_CONFIG_FILE": str(pip_config),
                "PIP_EXTRA_INDEX_URL": environment_index,
            }
        )

        isolated_only = subprocess.run(
            [sys.executable, "-m", "pip", "--isolated", "config", "list"],
            check=True,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertTrue(config_index in isolated_only.stdout)
        self.assertFalse(environment_index in isolated_only.stdout)

        environment["PIP_CONFIG_FILE"] = os.devnull
        config_file_only = subprocess.run(
            [sys.executable, "-m", "pip", "config", "list"],
            check=True,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertTrue(environment_index in config_file_only.stdout)

        dual_isolation = subprocess.run(
            [sys.executable, "-m", "pip", "--isolated", "config", "list"],
            check=True,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(dual_isolation.stdout.strip(), "")
        self.assertFalse(config_index in dual_isolation.stderr)
        self.assertFalse(environment_index in dual_isolation.stderr)

    def test_pip_dual_isolation_confines_pinned_find_links(self):
        ambient_wheelhouse = self.root / "ambient-wheels"
        intended_wheelhouse = self.root / "intended-wheels"
        cache_dir = self.root / "pip-cache"
        ambient_wheelhouse.mkdir()
        intended_wheelhouse.mkdir()
        wheel = (
            ambient_wheelhouse
            / "fasta2v_pip_isolation_probe-1.0-py3-none-any.whl"
        )
        files = {
            "fasta2v_pip_isolation_probe/__init__.py": b"VALUE = 1\n",
            "fasta2v_pip_isolation_probe-1.0.dist-info/METADATA": (
                b"Metadata-Version: 2.1\n"
                b"Name: fasta2v-pip-isolation-probe\n"
                b"Version: 1.0\n"
            ),
            "fasta2v_pip_isolation_probe-1.0.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\n"
                b"Generator: FastA2V-test\n"
                b"Root-Is-Purelib: true\n"
                b"Tag: py3-none-any\n"
            ),
            "fasta2v_pip_isolation_probe-1.0.dist-info/RECORD": b"",
        }
        with zipfile.ZipFile(
            wheel,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name, content in files.items():
                archive.writestr(name, content)

        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PIP_")
        }
        environment.update(
            {
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_FIND_LINKS": str(ambient_wheelhouse),
            }
        )
        common = [
            "install",
            "--dry-run",
            "--disable-pip-version-check",
            "--no-input",
            "--cache-dir",
            str(cache_dir),
            "--target",
            str(self.root / "pip-target"),
            "--no-index",
            "--no-deps",
            "--find-links",
            str(intended_wheelhouse),
            "fasta2v-pip-isolation-probe==1.0",
        ]

        config_file_only = subprocess.run(
            [sys.executable, "-I", "-B", "-m", "pip", *common],
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(
            config_file_only.returncode,
            0,
            config_file_only.stdout + config_file_only.stderr,
        )
        self.assertIn(str(wheel), config_file_only.stdout)

        dual_isolation = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-m",
                "pip",
                "--isolated",
                *common,
            ],
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        combined_output = dual_isolation.stdout + dual_isolation.stderr
        self.assertNotEqual(dual_isolation.returncode, 0)
        self.assertNotIn(str(ambient_wheelhouse), combined_output)
        self.assertIn("No matching distribution found", combined_output)

    def test_missing_weight_fails_before_any_lpips_score_exists(self):
        protocol, _receipt_path, _weight, kwargs = self.make_dependency_fixture(
            create_weight=False
        )
        with self.assertRaisesRegex(QUALITY.QualityError, "weight file is missing"):
            QUALITY.validate_lpips_receipt(protocol, **kwargs)

    def test_weight_hash_drift_is_rejected(self):
        protocol, _receipt_path, weight, kwargs = self.make_dependency_fixture(
            weight_bytes=b"original-weight"
        )
        weight.write_bytes(b"tampered-weight")
        with self.assertRaisesRegex(QUALITY.QualityError, "SHA256 drifted|byte count drifted"):
            QUALITY.validate_lpips_receipt(protocol, **kwargs)


if __name__ == "__main__":
    unittest.main()
