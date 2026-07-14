import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from ovi.sparge_evidence import sparge_microtest_evidence_errors


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

GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"


def load_sparge_microtest_module():
    torch_module = ModuleType("torch")
    torch_nn_module = ModuleType("torch.nn")
    torch_functional_module = ModuleType("torch.nn.functional")
    torch_module.nn = torch_nn_module
    torch_nn_module.functional = torch_functional_module

    backend_module = ModuleType("ovi.modules.sparge_attention_backend")
    backend_module.SPARGEATTN_MICROTEST_MIN_COSINE = (
        BACKEND_MODULE.SPARGEATTN_MICROTEST_MIN_COSINE
    )
    backend_module.SPARGEATTN_MICROTEST_SHAPE = (
        BACKEND_MODULE.SPARGEATTN_MICROTEST_SHAPE
    )
    backend_module.SpargeAttentionDependencyError = (
        SpargeAttentionDependencyError
    )
    backend_module.load_official_sparge_kernel = lambda: None

    gpu_module = ModuleType("ovi.gpu_process_monitor")
    gpu_module.query_gpu_compute_processes = lambda _device_index: None

    path = REPO_ROOT / "scripts" / "sparge_attn_microtest.py"
    spec = importlib.util.spec_from_file_location(
        "sparge_attn_microtest_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "torch": torch_module,
            "torch.nn": torch_nn_module,
            "torch.nn.functional": torch_functional_module,
            "ovi.modules.sparge_attention_backend": backend_module,
            "ovi.gpu_process_monitor": gpu_module,
        },
    ):
        spec.loader.exec_module(module)
    return module


def complete_microtest():
    return {
        "status": "ok",
        "device": "NVIDIA A100-SXM4-80GB",
        "device_uuid": GPU_UUID,
        "shape": [1, 132, 24, 128],
        "compute_capability": [8, 0],
        "torch": "2.6.0+cu124",
        "torch_cuda": "12.4",
        "torch_cxx11_abi": False,
        "dtype": "torch.bfloat16",
        "tensor_layout": "NHD",
        "tested_topk": [0.5, 1.0],
        "cosine_vs_sdpa": 0.99,
        "max_abs_difference_vs_sdpa": 0.1,
    }


def complete_receipt_metadata():
    digest = "a" * 64
    core = {"bytes": 20, "sha256": digest}
    return {
        "repository": "https://github.com/thu-ml/SpargeAttn.git",
        "clone_url": "ssh://git@ssh.github.com:443/thu-ml/SpargeAttn.git",
        "commit": SPARGEATTN_COMMIT,
        "api": SPARGEATTN_API,
        "package": "spas_sage_attn",
        "package_version": "0.1.0",
        "python": "3.11.15",
        "torch": "2.6.0+cu124",
        "torch_cuda": "12.4",
        "torch_cxx11_abi": False,
        "triton": "3.2.0",
        "cuda_home": "/usr/local/cuda-12.1",
        "torch_cuda_arch_list": "8.0",
        "max_jobs": 2,
        "source_dir": (
            "/cache/liluchen/FastA2V/sources/SpargeAttn-" + SPARGEATTN_COMMIT
        ),
        "source_core": {
            "path": (
                "/cache/liluchen/FastA2V/sources/SpargeAttn-"
                + SPARGEATTN_COMMIT
                + "/spas_sage_attn/core.py"
            ),
            **core,
        },
        "installed_package_root": (
            "/cache/liluchen/FastA2V/envs/ovi/lib/python3.11/site-packages/"
            "spas_sage_attn"
        ),
        "installed_files": {
            "core.py": dict(core),
            "_qattn.test.so": {
                "bytes": 1,
                "sha256": "b" * 64,
                "ldd_not_found": [],
            },
            "_fused.test.so": {
                "bytes": 1,
                "sha256": "c" * 64,
                "ldd_not_found": [],
            },
        },
        "build_log": {
            "path": "/cache/liluchen/FastA2V/spargeattn-build.log",
            "bytes": 1,
            "sha256": digest,
        },
        "install_pre_run_gpu": {
            "path": "/cache/liluchen/FastA2V/spargeattn-pre_run_gpu.json",
            "bytes": 1,
            "sha256": digest,
            "device_uuid": GPU_UUID,
        },
        "microtest": complete_microtest(),
    }


class FakeDevice:
    def __init__(self, device_type="cuda", index=0):
        self.type = device_type
        self.index = index

    def __str__(self):
        return f"{self.type}:{self.index}"


class FakeTensor:
    def __init__(self, label, shape, *, device=None, dtype="torch.bfloat16"):
        self.label = label
        self.shape = tuple(shape)
        self.device = device or FakeDevice()
        self.dtype = dtype

    def flatten(self, start_dim):
        if start_dim != 2 or len(self.shape) != 4:
            raise AssertionError(f"unexpected flatten({start_dim}) for {self.shape}")
        return FakeTensor(
            f"flatten({self.label})",
            (self.shape[0], self.shape[1], self.shape[2] * self.shape[3]),
            device=self.device,
            dtype=self.dtype,
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
    def test_microtest_normalizes_torch_26_uuid_without_gpu_prefix(self):
        microtest = load_sparge_microtest_module()
        properties = SimpleNamespace(
            uuid="48F3AC25-1111-2222-3333-444444444444"
        )
        self.assertEqual(
            microtest._normalize_runtime_gpu_uuid(properties),
            "GPU-48f3ac25-1111-2222-3333-444444444444",
        )
        self.assertEqual(
            microtest._validate_runtime_gpu_binding(
                properties,
                "NVIDIA A100-SXM4-80GB",
                {
                    "available": True,
                    "device_index": 0,
                    "device_name": "NVIDIA A100-SXM4-80GB",
                    "device_uuid": (
                        "GPU-48f3ac25-1111-2222-3333-444444444444"
                    ),
                    "process_count": 1,
                },
            ),
            "GPU-48f3ac25-1111-2222-3333-444444444444",
        )

    def test_microtest_missing_or_malformed_runtime_uuid_fails_closed(self):
        microtest = load_sparge_microtest_module()

        class ExplodingUuid:
            def __str__(self):
                raise TypeError("uuid conversion failed")

        for label, properties in (
            ("missing", SimpleNamespace()),
            ("wrong_type", SimpleNamespace(uuid=17)),
            ("malformed", SimpleNamespace(uuid="GPU-not-a-uuid")),
            ("conversion_error", SimpleNamespace(uuid=ExplodingUuid())),
        ):
            with self.subTest(label=label), self.assertRaises(
                SpargeAttentionDependencyError
            ):
                microtest._normalize_runtime_gpu_uuid(properties)

    def test_same_name_different_uuid_cannot_bind_logical_cuda_zero(self):
        microtest = load_sparge_microtest_module()
        runtime_properties = SimpleNamespace(
            uuid="aaaaaaaa-1111-2222-3333-444444444444"
        )
        physical_zero = {
            "available": True,
            "device_index": 0,
            "device_name": "NVIDIA A100-SXM4-80GB",
            "device_uuid": "GPU-bbbbbbbb-1111-2222-3333-444444444444",
            "process_count": 0,
        }
        with self.assertRaisesRegex(
            SpargeAttentionDependencyError,
            "UUID",
        ):
            microtest._validate_runtime_gpu_binding(
                runtime_properties,
                "NVIDIA A100-SXM4-80GB",
                physical_zero,
            )

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

    def test_non_bf16_qkv_fails_before_official_kernel(self):
        backend, kernel, _, _ = self.make_backend()
        attention = RecordingOviAttention()
        original_qkv_fn = attention.qkv_fn

        def float16_qkv(x):
            q, k, v = original_qkv_fn(x)
            for tensor in (q, k, v):
                tensor.dtype = "torch.float16"
            return q, k, v

        attention.qkv_fn = float16_qkv
        with self.assertRaisesRegex(SpargeAttentionInputError, "torch.bfloat16"):
            backend(
                attention,
                FakeTensor("video_hidden", (1, 15004, 3072)),
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
            root = (
                Path(directory)
                / "envs"
                / "ovi"
                / "lib"
                / "python3.11"
                / "site-packages"
                / "spas_sage_attn"
            )
            root.mkdir(parents=True)
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
                if path.suffix == ".so":
                    installed_files[path.name]["ldd_not_found"] = []
            source_dir = (
                Path(directory)
                / "sources"
                / f"SpargeAttn-{SPARGEATTN_COMMIT}"
                / "spas_sage_attn"
            )
            source_dir.mkdir(parents=True)
            source_core_path = source_dir / "core.py"
            source_core_path.write_bytes((root / "core.py").read_bytes())
            build_log_path = Path(directory) / "spargeattn-build.log"
            build_log_path.write_text("official build log\n", encoding="utf-8")
            install_gpu_path = Path(directory) / "spargeattn-pre_run_gpu.json"
            install_gpu_path.write_text("{}\n", encoding="utf-8")

            receipt = complete_receipt_metadata()
            receipt.update(
                {
                    "source_dir": str(source_dir.parent),
                    "source_core": {
                        "path": str(source_core_path),
                        "bytes": source_core_path.stat().st_size,
                        "sha256": hashlib.sha256(
                            source_core_path.read_bytes()
                        ).hexdigest(),
                    },
                    "installed_package_root": str(root),
                    "installed_files": installed_files,
                    "build_log": {
                        "path": str(build_log_path),
                        "bytes": build_log_path.stat().st_size,
                        "sha256": hashlib.sha256(
                            build_log_path.read_bytes()
                        ).hexdigest(),
                    },
                    "install_pre_run_gpu": {
                        "path": str(install_gpu_path),
                        "bytes": install_gpu_path.stat().st_size,
                        "sha256": hashlib.sha256(
                            install_gpu_path.read_bytes()
                        ).hexdigest(),
                        "device_uuid": GPU_UUID,
                    },
                }
            )
            receipt_path = Path(directory) / "spargeattn-install.json"
            receipt_path.write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            with mock.patch.dict(
                BACKEND_MODULE.os.environ,
                {"FASTA2V_CACHE_ROOT": directory},
            ):
                verified_path, verified = verify_sparge_install_receipt(
                    receipt_path
                )
            self.assertEqual(verified_path, receipt_path)
            self.assertEqual(verified, receipt)

            (root / "core.py").write_text("changed\n", encoding="utf-8")
            with mock.patch.dict(
                BACKEND_MODULE.os.environ,
                {"FASTA2V_CACHE_ROOT": directory},
            ):
                with self.assertRaisesRegex(
                    SpargeAttentionDependencyError, "differ from the pinned"
                ):
                    verify_sparge_install_receipt(receipt_path)

    def test_install_receipt_requires_real_cuda_microtest(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "spargeattn-install.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "repository": "https://github.com/thu-ml/SpargeAttn.git",
                        "clone_url": "ssh://git@ssh.github.com:443/thu-ml/SpargeAttn.git",
                        "commit": SPARGEATTN_COMMIT,
                        "api": SPARGEATTN_API,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SpargeAttentionDependencyError, "microtest"):
                verify_sparge_install_receipt(receipt_path)

    def test_microtest_evidence_rejects_nonfinite_comparison(self):
        evidence = complete_microtest()
        evidence["cosine_vs_sdpa"] = float("nan")
        evidence["max_abs_difference_vs_sdpa"] = float("inf")
        errors = sparge_microtest_evidence_errors(
            evidence, expected_gpu_uuid=GPU_UUID
        )
        self.assertEqual(sum("must be finite" in error for error in errors), 2)

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
        for filename in (
            "spargeattn-install.json",
            "spargeattn-build.log",
            "spargeattn-install-pre_run_gpu.json",
        ):
            self.assertGreaterEqual(inference_source.count(f'"{filename}"'), 2)

    def test_formal_verifier_requires_sparge_backend_provenance_and_calls(self):
        verifier_path = REPO_ROOT / "scripts" / "verify_ovi_output.py"
        verifier_spec = importlib.util.spec_from_file_location(
            "verify_ovi_output_under_test", verifier_path
        )
        verifier_module = importlib.util.module_from_spec(verifier_spec)
        with mock.patch.dict(sys.modules, {"numpy": SimpleNamespace()}):
            verifier_spec.loader.exec_module(verifier_module)

        receipt = complete_receipt_metadata()
        dispatcher = {
            "calls_total": 2950,
            "calls_by_method": {
                "dense": 0,
                "sparge": 2950,
                "radial": 0,
                "svg": 0,
            },
            "backend_details": {
                **verifier_module.SPARGE_PROVENANCE,
                "calls": 2950,
                "last_nhd_shape": [1, 15004, 24, 128],
                "last_dtype": "torch.bfloat16",
                "last_device": "cuda:0",
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
            expected_gpu_uuid=GPU_UUID,
        )
        self.assertEqual(errors, [])

        dispatcher["backend_details"]["calls"] = 2949
        errors = []
        verifier_module.validate_sparge_dispatcher(dispatcher, errors)
        self.assertTrue(any("calls=" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
