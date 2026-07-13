import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = REPO_ROOT / "ovi" / "modules" / "sparge_attention_backend.py"
SPEC = importlib.util.spec_from_file_location(
    "sparge_attention_backend_under_test", BACKEND_PATH
)
BACKEND_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BACKEND_MODULE
SPEC.loader.exec_module(BACKEND_MODULE)

SPARGEATTN_API = BACKEND_MODULE.SPARGEATTN_API
SPARGEATTN_COMMIT = BACKEND_MODULE.SPARGEATTN_COMMIT
SpargeAttentionDependencyError = BACKEND_MODULE.SpargeAttentionDependencyError
SpargeAttentionInputError = BACKEND_MODULE.SpargeAttentionInputError
SpargeVideoSelfAttentionBackend = (
    BACKEND_MODULE.SpargeVideoSelfAttentionBackend
)
load_official_sparge_kernel = BACKEND_MODULE.load_official_sparge_kernel
verify_sparge_install_receipt = BACKEND_MODULE.verify_sparge_install_receipt


class FakeDevice:
    def __init__(self, device_type="cuda", index=0):
        self.type = device_type
        self.index = index

    def __str__(self):
        return f"{self.type}:{self.index}"


class FakeTensor:
    def __init__(self, label, shape, *, device=None):
        self.label = label
        self.shape = tuple(shape)
        self.device = device or FakeDevice()

    def flatten(self, start_dim):
        if start_dim != 2 or len(self.shape) != 4:
            raise AssertionError(f"unexpected flatten({start_dim}) for {self.shape}")
        return FakeTensor(
            f"flatten({self.label})",
            (self.shape[0], self.shape[1], self.shape[2] * self.shape[3]),
            device=self.device,
        )


class FakeSequenceLengths:
    def __init__(self, values):
        self.values = list(values)
        self.tolist_calls = 0

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        self.tolist_calls += 1
        return list(self.values)


class RecordingOviAttention:
    def __init__(self, *, use_sp=False, qkv_shape=None):
        self.use_sp = use_sp
        self.window_size = (-1, -1)
        self.qkv_shape = qkv_shape
        self.qkv_inputs = []
        self.projection_inputs = []

    def qkv_fn(self, x):
        self.qkv_inputs.append(x)
        nhd_shape = self.qkv_shape or (x.shape[0], x.shape[1], 24, 128)
        return (
            FakeTensor("q_from_original_qkv_norm", nhd_shape, device=x.device),
            FakeTensor("k_from_original_qkv_norm", nhd_shape, device=x.device),
            FakeTensor("v_from_original_qkv", nhd_shape, device=x.device),
        )

    def o(self, x):
        self.projection_inputs.append(x)
        return FakeTensor("projected_by_original_o", x.shape, device=x.device)


class RecordingKernel:
    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, **kwargs):
        self.calls.append((q, k, v, kwargs))
        return FakeTensor("official_sparge_output", q.shape, device=q.device)


class SpargeAttentionBackendTests(unittest.TestCase):
    def make_backend(self, *, use_collectives=False):
        kernel = RecordingKernel()
        rope_calls = []
        collective_calls = []

        def rope_apply_fn(tensor, grid_sizes, freqs):
            rope_calls.append((tensor, grid_sizes, freqs))
            return FakeTensor(
                f"rope({tensor.label})", tensor.shape, device=tensor.device
            )

        def all_to_all_4d_fn(tensor, *, scatter_dim, gather_dim):
            collective_calls.append((tensor.label, scatter_dim, gather_dim))
            shape = list(tensor.shape)
            if (scatter_dim, gather_dim) == (2, 1):
                shape[1] *= 2
                shape[2] //= 2
            elif (scatter_dim, gather_dim) == (1, 2):
                shape[1] //= 2
                shape[2] *= 2
            else:
                raise AssertionError(
                    f"unexpected all_to_all dimensions: {scatter_dim}/{gather_dim}"
                )
            return FakeTensor(
                f"all_to_all({tensor.label})", shape, device=tensor.device
            )

        backend = SpargeVideoSelfAttentionBackend(
            kernel=kernel,
            rope_apply_fn=rope_apply_fn,
            all_to_all_4d_fn=all_to_all_4d_fn if use_collectives else None,
            topk=0.5,
            pvthreshd=50,
            smooth_k=True,
        )
        return backend, kernel, rope_calls, collective_calls

    def test_real_ovi_nhd_contract_reuses_qkv_rope_and_output_projection(self):
        backend, kernel, rope_calls, _ = self.make_backend()
        attention = RecordingOviAttention()
        hidden = FakeTensor("video_hidden", (1, 15004, 3072))
        seq_lens = FakeSequenceLengths([15004])
        grid_sizes = object()
        freqs = object()

        result = backend(
            attention,
            hidden,
            seq_lens,
            grid_sizes,
            freqs,
            block_index=17,
            debug_context={"branch": "conditional"},
        )

        self.assertEqual(result.label, "projected_by_original_o")
        self.assertEqual(attention.qkv_inputs, [hidden])
        self.assertEqual(len(attention.projection_inputs), 1)
        self.assertEqual(attention.projection_inputs[0].shape, (1, 15004, 3072))
        self.assertEqual(
            [call[0].label for call in rope_calls],
            ["q_from_original_qkv_norm", "k_from_original_qkv_norm"],
        )
        self.assertEqual(len(kernel.calls), 1)
        q, k, v, kwargs = kernel.calls[0]
        self.assertEqual(q.label, "rope(q_from_original_qkv_norm)")
        self.assertEqual(k.label, "rope(k_from_original_qkv_norm)")
        self.assertEqual(v.label, "v_from_original_qkv")
        self.assertEqual(
            kwargs,
            {
                "dropout_p": 0.0,
                "is_causal": False,
                "topk": 0.5,
                "pvthreshd": 50.0,
                "smooth_k": True,
                "tensor_layout": "NHD",
                "return_sparsity": False,
            },
        )
        self.assertEqual(seq_lens.tolist_calls, 1)
        details = backend.metrics()
        self.assertEqual(details["pinned_commit"], SPARGEATTN_COMMIT)
        self.assertEqual(details["api"], SPARGEATTN_API)
        self.assertEqual(details["last_nhd_shape"], [1, 15004, 24, 128])
        self.assertFalse(details["return_sparsity"])
        self.assertEqual(details["calls"], 1)

    def test_full_length_validation_runs_only_once_per_shape_and_device(self):
        backend, kernel, _, _ = self.make_backend()
        attention = RecordingOviAttention()
        hidden = FakeTensor("video_hidden", (1, 15004, 3072))
        seq_lens = FakeSequenceLengths([15004])

        for _ in range(2):
            backend(attention, hidden, seq_lens, object(), object())

        self.assertEqual(seq_lens.tolist_calls, 1)
        self.assertEqual(len(kernel.calls), 2)

    def test_new_generation_revalidates_lengths_even_for_same_shape(self):
        backend, kernel, _, _ = self.make_backend()
        attention = RecordingOviAttention()
        hidden = FakeTensor("video_hidden", (1, 15004, 3072))
        backend(
            attention,
            hidden,
            FakeSequenceLengths([15004]),
            object(),
            object(),
        )

        backend.reset_metrics()
        with self.assertRaisesRegex(SpargeAttentionInputError, "cannot represent padded"):
            backend(
                attention,
                hidden,
                FakeSequenceLengths([14900]),
                object(),
                object(),
            )
        self.assertEqual(len(kernel.calls), 1)

    def test_padded_input_fails_instead_of_ignoring_seq_lens(self):
        backend, kernel, _, _ = self.make_backend()
        attention = RecordingOviAttention()
        hidden = FakeTensor("video_hidden", (1, 15004, 3072))

        with self.assertRaisesRegex(SpargeAttentionInputError, "cannot represent padded"):
            backend(
                attention,
                hidden,
                FakeSequenceLengths([14900]),
                object(),
                object(),
            )
        self.assertEqual(kernel.calls, [])

    def test_official_kernel_shape_constraints_fail_fast(self):
        backend, kernel, _, _ = self.make_backend()
        short_attention = RecordingOviAttention(qkv_shape=(1, 127, 24, 128))
        with self.assertRaisesRegex(SpargeAttentionInputError, ">= 128"):
            backend(
                short_attention,
                FakeTensor("short_hidden", (1, 127, 3072)),
                FakeSequenceLengths([127]),
                object(),
                object(),
            )

        wrong_head_attention = RecordingOviAttention(
            qkv_shape=(1, 15004, 32, 96)
        )
        with self.assertRaisesRegex(SpargeAttentionInputError, "64 or 128"):
            backend(
                wrong_head_attention,
                FakeTensor("video_hidden", (1, 15004, 3072)),
                FakeSequenceLengths([15004]),
                object(),
                object(),
            )
        self.assertEqual(kernel.calls, [])

    def test_sequence_parallel_reuses_only_ovi_collectives(self):
        backend, kernel, _, collective_calls = self.make_backend(
            use_collectives=True
        )
        attention = RecordingOviAttention(use_sp=True)
        hidden = FakeTensor("video_hidden_local", (1, 7502, 3072))

        backend(
            attention,
            hidden,
            FakeSequenceLengths([15004]),
            object(),
            object(),
        )

        self.assertEqual(len(kernel.calls), 1)
        self.assertEqual(kernel.calls[0][0].shape, (1, 15004, 12, 128))
        self.assertEqual(attention.projection_inputs[0].shape, (1, 7502, 3072))
        self.assertEqual(
            [(scatter, gather) for _, scatter, gather in collective_calls],
            [(2, 1), (2, 1), (2, 1), (1, 2)],
        )

    def test_cpu_input_fails_before_official_kernel(self):
        backend, kernel, _, _ = self.make_backend()
        attention = RecordingOviAttention()
        hidden = FakeTensor(
            "video_hidden", (1, 15004, 3072), device=FakeDevice("cpu", 0)
        )

        with self.assertRaisesRegex(SpargeAttentionInputError, "CUDA backend"):
            backend(
                attention,
                hidden,
                FakeSequenceLengths([15004]),
                object(),
                object(),
            )
        self.assertEqual(kernel.calls, [])

    def test_pinned_upstream_rejects_smooth_k_false_before_inference(self):
        with self.assertRaisesRegex(ValueError, "sparge_smooth_k must be true"):
            SpargeVideoSelfAttentionBackend(
                kernel=lambda *args, **kwargs: None,
                rope_apply_fn=lambda tensor, grid, freqs: tensor,
                smooth_k=False,
            )

    def test_incompatible_kernel_return_fails_without_dense_fallback(self):
        hidden = FakeTensor("video_hidden", (1, 15004, 3072))
        attention = RecordingOviAttention()

        tuple_backend = SpargeVideoSelfAttentionBackend(
            kernel=lambda q, k, v, **kwargs: (q, 0.5),
            rope_apply_fn=lambda tensor, grid, freqs: tensor,
        )
        with self.assertRaisesRegex(RuntimeError, "returned a tuple"):
            tuple_backend(
                attention,
                hidden,
                FakeSequenceLengths([15004]),
                object(),
                object(),
            )

        wrong_shape_backend = SpargeVideoSelfAttentionBackend(
            kernel=lambda q, k, v, **kwargs: FakeTensor(
                "wrong", (1, 14999, 24, 128), device=q.device
            ),
            rope_apply_fn=lambda tensor, grid, freqs: tensor,
        )
        with self.assertRaisesRegex(RuntimeError, "output shape differs"):
            wrong_shape_backend(
                attention,
                hidden,
                FakeSequenceLengths([15004]),
                object(),
                object(),
            )

    def test_non_global_ovi_window_fails_instead_of_changing_semantics(self):
        backend, kernel, _, _ = self.make_backend()
        attention = RecordingOviAttention()
        attention.window_size = (128, 128)
        hidden = FakeTensor("video_hidden", (1, 15004, 3072))

        with self.assertRaisesRegex(SpargeAttentionInputError, "global attention"):
            backend(
                attention,
                hidden,
                FakeSequenceLengths([15004]),
                object(),
                object(),
            )
        self.assertEqual(kernel.calls, [])

    def test_missing_official_package_fails_fast_with_install_command(self):
        with mock.patch.object(
            BACKEND_MODULE,
            "import_module",
            side_effect=ModuleNotFoundError("spas_sage_attn"),
        ):
            with self.assertRaises(SpargeAttentionDependencyError) as caught:
                load_official_sparge_kernel()

        message = str(caught.exception)
        self.assertIn("scripts/install_sparge_attn.sh", message)
        self.assertIn(SPARGEATTN_COMMIT, message)

    def test_incompatible_public_api_fails_fast(self):
        incompatible = SimpleNamespace(
            spas_sage2_attn_meansim_topk_cuda=lambda q, k, v: None
        )
        with mock.patch.object(
            BACKEND_MODULE, "import_module", return_value=incompatible
        ):
            with self.assertRaisesRegex(
                SpargeAttentionDependencyError, "incompatible signature"
            ):
                load_official_sparge_kernel()

    def test_install_receipt_must_match_pinned_official_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "spargeattn-install.json"
            receipt_path.write_text(
                "{\n"
                '  "repository": "https://github.com/thu-ml/SpargeAttn.git",\n'
                '  "commit": "wrong",\n'
                '  "api": "spas_sage2_attn_meansim_topk_cuda"\n'
                "}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SpargeAttentionDependencyError, "does not match"
            ):
                verify_sparge_install_receipt(receipt_path)

    def test_install_receipt_fingerprints_detect_package_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "spas_sage_attn"
            root.mkdir()
            for name, content in {
                "core.py": b"official python core\n",
                "_qattn.test.so": b"official qattn extension\n",
                "_fused.test.so": b"official fused extension\n",
            }.items():
                (root / name).write_bytes(content)

            installed_files = {}
            for path in root.iterdir():
                installed_files[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            receipt = {
                "repository": "https://github.com/thu-ml/SpargeAttn.git",
                "commit": SPARGEATTN_COMMIT,
                "api": SPARGEATTN_API,
                "installed_package_root": str(root),
                "installed_files": installed_files,
            }
            receipt_path = Path(directory) / "spargeattn-install.json"
            receipt_path.write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            verified_path, verified = verify_sparge_install_receipt(receipt_path)
            self.assertEqual(verified_path, receipt_path)
            self.assertEqual(verified, receipt)

            (root / "core.py").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(
                SpargeAttentionDependencyError, "differ from the pinned"
            ):
                verify_sparge_install_receipt(receipt_path)

    def test_adapter_uses_public_package_api_not_private_kernel_modules(self):
        source = BACKEND_PATH.read_text(encoding="utf-8")
        self.assertIn("spas_sage_attn", source)
        self.assertNotIn("spas_sage_attn._qattn", source)
        self.assertNotIn("spas_sage_attn._fused", source)
        self.assertNotIn("flash_attention(", source)
        self.assertNotIn("scaled_dot_product_attention", source)

    def test_machine_readable_pin_matches_adapter_constant(self):
        pin = (REPO_ROOT / "third_party" / "SpargeAttn.commit").read_text(
            encoding="utf-8"
        ).strip()
        self.assertEqual(pin, SPARGEATTN_COMMIT)

    def test_engine_constructs_backend_only_for_sparge_method(self):
        engine_path = REPO_ROOT / "ovi" / "ovi_fusion_engine.py"
        tree = ast.parse(engine_path.read_text(encoding="utf-8"))
        builder_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_sparge_video_backend"
        ]
        self.assertEqual(len(builder_calls), 1)

    def test_sparse_receipt_is_allowed_in_fresh_run_directory(self):
        inference_source = (REPO_ROOT / "inference.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'allowed_pre_run_files.add("spargeattn-install.json")',
            inference_source,
        )
        self.assertIn(
            'evidence_filenames.append("spargeattn-install.json")',
            inference_source,
        )

    def test_formal_verifier_requires_sparge_backend_provenance_and_calls(self):
        verifier_path = REPO_ROOT / "scripts" / "verify_ovi_output.py"
        verifier_spec = importlib.util.spec_from_file_location(
            "verify_ovi_output_under_test", verifier_path
        )
        verifier_module = importlib.util.module_from_spec(verifier_spec)
        with mock.patch.dict(sys.modules, {"numpy": SimpleNamespace()}):
            verifier_spec.loader.exec_module(verifier_module)

        receipt = {"commit": SPARGEATTN_COMMIT, "installed_files": {}}
        dispatcher = {
            "calls_total": 2950,
            "backend_details": {
                **verifier_module.SPARGE_PROVENANCE,
                "calls": 2950,
                "topk": 0.5,
                "pvthreshd": 50.0,
                "smooth_k": True,
                "install_receipt": receipt,
            },
        }
        errors = []
        verifier_module.validate_sparge_dispatcher(
            dispatcher,
            errors,
            expected_receipt=receipt,
            expected_settings={
                "topk": 0.5,
                "pvthreshd": 50.0,
                "smooth_k": True,
            },
        )
        self.assertEqual(errors, [])

        dispatcher["backend_details"]["calls"] = 2949
        errors = []
        verifier_module.validate_sparge_dispatcher(dispatcher, errors)
        self.assertTrue(any("calls=" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
