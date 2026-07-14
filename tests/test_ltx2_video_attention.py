import ast
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import NamedTuple
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ltx2.video_attention import (  # noqa: E402
    DensePassthroughBackend,
    LTX2VideoAttentionInputError,
    LTX2VideoAttentionIntegrationError,
    LTX2VideoAttentionKernelError,
    OFFICIAL_LTX2_COMMIT,
    SPARGEATTN_API,
    SpargeVideoSelfAttentionBackend,
    create_ltx2_video_self_attention_module_ops,
    with_ltx2_video_self_attention,
    with_ltx2_video_self_attention_builder,
)


@dataclass(frozen=True)
class FakeDevice:
    type: str
    index: int | None = None

    def __str__(self):
        if self.index is None:
            return self.type
        return f"{self.type}:{self.index}"


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        dtype="torch.bfloat16",
        device=FakeDevice("cuda", 0),
        reshape_log=None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.reshape_log = [] if reshape_log is None else reshape_log

    def reshape(self, *shape):
        self.reshape_log.append((self.shape, tuple(shape)))
        source_size = 1
        target_size = 1
        for value in self.shape:
            source_size *= value
        for value in shape:
            target_size *= value
        if source_size != target_size:
            raise ValueError("fake reshape size mismatch")
        return FakeTensor(
            shape,
            dtype=self.dtype,
            device=self.device,
            reshape_log=self.reshape_log,
        )


class RecordingDense:
    def __init__(self, result=None):
        self.result = object() if result is None else result
        self.calls = []

    def __call__(self, q, k, v, heads):
        self.calls.append((q, k, v, heads))
        return self.result


class RecordingKernel:
    def __init__(self, output_factory=None, error=None):
        self.output_factory = output_factory
        self.error = error
        self.calls = []

    def __call__(self, q, k, v, **kwargs):
        self.calls.append((q, k, v, kwargs))
        if self.error is not None:
            raise self.error
        if self.output_factory is not None:
            return self.output_factory(q, k, v)
        return FakeTensor(q.shape, dtype=q.dtype, device=q.device)


class FakeModuleOps(NamedTuple):
    name: str
    matcher: object
    mutator: object


class FakeAttention:
    def __init__(self, prefix):
        self.attention_function = RecordingDense(f"{prefix}-unmasked")
        self.masked_attention_function = RecordingDense(f"{prefix}-masked")
        self.heads = 2
        self.dim_head = 128


class FakeBlock:
    def __init__(self, index):
        self.attn1 = FakeAttention(f"video-{index}")
        self.attn2 = FakeAttention(f"video-text-{index}")
        self.audio_attn1 = FakeAttention(f"audio-{index}")
        self.audio_attn2 = FakeAttention(f"audio-text-{index}")
        self.audio_to_video_attn = FakeAttention(f"a2v-{index}")
        self.video_to_audio_attn = FakeAttention(f"v2a-{index}")


class FakeModel:
    def __init__(self, block_count=2):
        self.transformer_blocks = [FakeBlock(i) for i in range(block_count)]


class OtherModel:
    pass


class FakeBuilder:
    def __init__(self, module_ops=()):
        self.module_ops = tuple(module_ops)

    def with_module_ops(self, module_ops):
        return FakeBuilder(module_ops)


class FakeStage:
    def __init__(self, builder):
        self._transformer_builder = builder

    def with_builder(self, builder):
        return FakeStage(builder)


def make_tensors(
    shape=(1, 128, 256),
    *,
    dtype="torch.bfloat16",
    device=FakeDevice("cuda", 0),
):
    return tuple(
        FakeTensor(shape, dtype=dtype, device=device) for _ in range(3)
    )


def make_sparge(kernel=None, **kwargs):
    return SpargeVideoSelfAttentionBackend(
        kernel=kernel or RecordingKernel(),
        expected_heads=kwargs.pop("expected_heads", 2),
        **kwargs,
    )


def make_module_op(backend):
    return create_ltx2_video_self_attention_module_ops(
        backend,
        module_ops_factory=FakeModuleOps,
        model_type=FakeModel,
        attention_type=FakeAttention,
    )


class LTX2VideoAttentionTests(unittest.TestCase):
    def test_package_exports_production_entrypoints(self):
        import ltx2

        self.assertIs(ltx2.DensePassthroughBackend, DensePassthroughBackend)
        self.assertIs(
            ltx2.SpargeVideoSelfAttentionBackend,
            SpargeVideoSelfAttentionBackend,
        )
        self.assertIs(
            ltx2.with_ltx2_video_self_attention,
            with_ltx2_video_self_attention,
        )

    def test_module_has_no_top_level_torch_or_ltx_import(self):
        source = (REPO_ROOT / "ltx2" / "video_attention.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported_roots = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertNotIn("torch", imported_roots)
        self.assertNotIn("ltx_core", imported_roots)
        self.assertEqual(
            OFFICIAL_LTX2_COMMIT,
            "9377758131b1ffde4b7f766804590a6617bf2ab9",
        )
        self.assertEqual(SPARGEATTN_API, "spas_sage2_attn_meansim_topk_cuda")

    def test_dense_passthrough_returns_exact_original_result_and_counts(self):
        expected = object()
        dense = RecordingDense(expected)
        backend = DensePassthroughBackend()
        bound = backend.bind(dense, block_index=7)
        q, k, v = make_tensors()

        actual = bound(q, k, v, 2)

        self.assertIs(actual, expected)
        self.assertEqual(dense.calls, [(q, k, v, 2)])
        self.assertEqual(backend.metrics()["calls"], 1)
        self.assertEqual(backend.metrics()["calls_by_block"], {"7": 1})

    def test_dense_direct_mode_requires_explicit_original(self):
        q, k, v = make_tensors()
        with self.assertRaisesRegex(
            LTX2VideoAttentionIntegrationError, "must be bound"
        ):
            DensePassthroughBackend()(q, k, v, 2)

        original = RecordingDense("dense")
        backend = DensePassthroughBackend(original)
        self.assertEqual(backend(q, k, v, 2), "dense")
        self.assertEqual(backend.metrics()["calls_by_block"], {"unbound": 1})

    def test_sparge_converts_ltx_flattened_qkv_to_nhd_and_back(self):
        kernel = RecordingKernel()
        backend = make_sparge(kernel, topk=0.75, pvthreshd=42)
        q, k, v = make_tensors()

        output = backend(q, k, v, 2)

        self.assertEqual(output.shape, (1, 128, 256))
        self.assertEqual(len(kernel.calls), 1)
        q_nhd, k_nhd, v_nhd, kwargs = kernel.calls[0]
        self.assertEqual(q_nhd.shape, (1, 128, 2, 128))
        self.assertEqual(k_nhd.shape, q_nhd.shape)
        self.assertEqual(v_nhd.shape, q_nhd.shape)
        self.assertEqual(
            kwargs,
            {
                "dropout_p": 0.0,
                "is_causal": False,
                "topk": 0.75,
                "pvthreshd": 42.0,
                "smooth_k": True,
                "tensor_layout": "NHD",
                "return_sparsity": False,
            },
        )
        metrics = backend.metrics()
        self.assertEqual(metrics["calls"], 1)
        self.assertEqual(metrics["sparse_calls"], 1)
        self.assertEqual(metrics["last_input_shape"], [1, 128, 256])
        self.assertEqual(metrics["last_nhd_shape"], [1, 128, 2, 128])
        self.assertFalse(metrics["fallback_used"])

    def test_configuration_validation_is_strict(self):
        kernel = RecordingKernel()
        invalid = (
            ({"topk": True}, TypeError),
            ({"topk": 0}, ValueError),
            ({"topk": 1.01}, ValueError),
            ({"topk": float("nan")}, ValueError),
            ({"pvthreshd": 0}, ValueError),
            ({"smooth_k": False}, ValueError),
            ({"fallback_to_dense": 1}, TypeError),
            ({"expected_heads": 0}, ValueError),
            ({"supported_head_dims": ()}, ValueError),
            ({"allowed_dtypes": ()}, ValueError),
            ({"required_device_type": ""}, TypeError),
            ({"min_sequence_length": 0}, ValueError),
        )
        for kwargs, error_type in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(error_type):
                    SpargeVideoSelfAttentionBackend(kernel=kernel, **kwargs)

    def test_invalid_inputs_fail_before_kernel_with_stable_reasons(self):
        cases = []

        q, k, v = make_tensors((1, 128, 2, 128))
        cases.append(("shape", (q, k, v, 2), {}))

        q, k, v = make_tensors()
        k = FakeTensor((1, 129, 256), dtype=k.dtype, device=k.device)
        cases.append(("shape", (q, k, v, 2), {}))

        cases.append(("heads", (*make_tensors(), True), {}))
        cases.append(("heads", (*make_tensors(), 4), {}))
        cases.append(
            (
                "heads",
                (*make_tensors((1, 128, 257)), 2),
                {"expected_heads": None},
            )
        )
        cases.append(
            (
                "head_dim",
                (*make_tensors((1, 128, 96)), 2),
                {"supported_head_dims": (64, 128)},
            )
        )
        cases.append(
            ("sequence_length", (*make_tensors((1, 127, 256)), 2), {})
        )
        cases.append(
            ("dtype", (*make_tensors(dtype="torch.float16"), 2), {})
        )
        q, k, v = make_tensors()
        k = FakeTensor(k.shape, dtype="torch.float16", device=k.device)
        cases.append(("dtype", (q, k, v, 2), {}))
        cases.append(
            (
                "device",
                (*make_tensors(device=FakeDevice("cpu")), 2),
                {},
            )
        )
        q, k, v = make_tensors()
        v = FakeTensor(v.shape, dtype=v.dtype, device=FakeDevice("cuda", 1))
        cases.append(("device", (q, k, v, 2), {}))

        for reason, args, kwargs in cases:
            with self.subTest(reason=reason, kwargs=kwargs):
                kernel = RecordingKernel()
                backend = make_sparge(kernel, **kwargs)
                with self.assertRaises(LTX2VideoAttentionInputError) as caught:
                    backend(*args)
                self.assertEqual(caught.exception.reason, reason)
                self.assertEqual(kernel.calls, [])
                self.assertEqual(backend.metrics()["errors"], 1)
                self.assertEqual(backend.metrics()["fallback_count"], 0)

    def test_explicit_dense_fallback_counts_each_reason_and_calls_original(self):
        kernel = RecordingKernel()
        backend = make_sparge(kernel, fallback_to_dense=True)
        dense = RecordingDense("fallback-result")
        bound = backend.bind(dense, block_index=3)

        for _ in range(2):
            q, k, v = make_tensors(device=FakeDevice("cpu"))
            self.assertEqual(bound(q, k, v, 2), "fallback-result")
        q, k, v = make_tensors(dtype="torch.float16")
        self.assertEqual(bound(q, k, v, 2), "fallback-result")

        metrics = backend.metrics()
        self.assertEqual(kernel.calls, [])
        self.assertEqual(len(dense.calls), 3)
        self.assertTrue(metrics["fallback_allowed"])
        self.assertTrue(metrics["fallback_used"])
        self.assertEqual(metrics["fallback_count"], 3)
        self.assertEqual(metrics["fallback_reasons"], {"device": 2, "dtype": 1})
        self.assertEqual(metrics["dense_fallback"]["calls"], 3)
        self.assertEqual(metrics["dense_fallback"]["calls_by_block"], {"3": 3})
        self.assertEqual(metrics["errors"], 0)

    def test_direct_sparse_call_cannot_hide_invalid_input_behind_fallback(self):
        backend = make_sparge(fallback_to_dense=True)
        q, k, v = make_tensors(device=FakeDevice("cpu"))
        with self.assertRaisesRegex(
            LTX2VideoAttentionIntegrationError, "requires a backend bound"
        ):
            backend(q, k, v, 2)
        self.assertEqual(backend.metrics()["fallback_count"], 0)
        self.assertEqual(backend.metrics()["errors"], 1)

    def test_kernel_exception_never_falls_back(self):
        kernel = RecordingKernel(error=RuntimeError("cuda exploded"))
        backend = make_sparge(kernel, fallback_to_dense=True)
        dense = RecordingDense("must-not-run")
        q, k, v = make_tensors()

        with self.assertRaisesRegex(
            LTX2VideoAttentionKernelError, SPARGEATTN_API
        ):
            backend.bind(dense)(q, k, v, 2)

        self.assertEqual(dense.calls, [])
        self.assertEqual(backend.metrics()["fallback_count"], 0)
        self.assertEqual(backend.metrics()["errors"], 1)

    def test_incompatible_kernel_outputs_never_fall_back(self):
        factories = {
            "tuple": lambda q, k, v: (q, 0.5),
            "shape": lambda q, k, v: FakeTensor(
                (1, 129, 2, 128), dtype=q.dtype, device=q.device
            ),
            "dtype": lambda q, k, v: FakeTensor(
                q.shape, dtype="torch.float16", device=q.device
            ),
            "device": lambda q, k, v: FakeTensor(
                q.shape, dtype=q.dtype, device=FakeDevice("cuda", 1)
            ),
        }
        for label, factory in factories.items():
            with self.subTest(label=label):
                backend = make_sparge(
                    RecordingKernel(output_factory=factory),
                    fallback_to_dense=True,
                )
                dense = RecordingDense("must-not-run")
                with self.assertRaises(LTX2VideoAttentionKernelError):
                    backend.bind(dense)(*make_tensors(), 2)
                self.assertEqual(dense.calls, [])
                self.assertEqual(backend.metrics()["fallback_count"], 0)
                self.assertEqual(backend.metrics()["errors"], 1)

    def test_reset_metrics_resets_sparse_and_fallback_counts(self):
        backend = make_sparge(fallback_to_dense=True)
        dense = RecordingDense("fallback")
        bound = backend.bind(dense)
        bound(*make_tensors(), 2)
        bound(*make_tensors(device=FakeDevice("cpu")), 2)
        backend.reset_metrics()
        metrics = backend.metrics()
        self.assertEqual(metrics["calls"], 0)
        self.assertEqual(metrics["sparse_calls"], 0)
        self.assertEqual(metrics["fallback_count"], 0)
        self.assertEqual(metrics["fallback_reasons"], {})
        self.assertEqual(metrics["dense_fallback"]["calls"], 0)

    def test_module_op_patches_only_each_video_attn1_unmasked_slot(self):
        model = FakeModel(3)
        backend = DensePassthroughBackend()
        op = make_module_op(backend)
        snapshots = []
        for block in model.transformer_blocks:
            snapshots.append(
                {
                    "attn1_unmasked": block.attn1.attention_function,
                    "attn1_masked": block.attn1.masked_attention_function,
                    "attn2": block.attn2.attention_function,
                    "audio1": block.audio_attn1.attention_function,
                    "audio2": block.audio_attn2.attention_function,
                    "a2v": block.audio_to_video_attn.attention_function,
                    "v2a": block.video_to_audio_attn.attention_function,
                }
            )

        self.assertTrue(op.matcher(model))
        self.assertFalse(op.matcher(OtherModel()))
        self.assertIs(op.mutator(model), model)

        for index, (block, snapshot) in enumerate(
            zip(model.transformer_blocks, snapshots, strict=True)
        ):
            self.assertIsNot(
                block.attn1.attention_function, snapshot["attn1_unmasked"]
            )
            self.assertIs(
                block.attn1.masked_attention_function, snapshot["attn1_masked"]
            )
            self.assertIs(block.attn2.attention_function, snapshot["attn2"])
            self.assertIs(block.audio_attn1.attention_function, snapshot["audio1"])
            self.assertIs(block.audio_attn2.attention_function, snapshot["audio2"])
            self.assertIs(
                block.audio_to_video_attn.attention_function, snapshot["a2v"]
            )
            self.assertIs(
                block.video_to_audio_attn.attention_function, snapshot["v2a"]
            )
            self.assertEqual(
                block.attn1.attention_function(*make_tensors(), 2),
                f"video-{index}-unmasked",
            )
        self.assertEqual(
            backend.metrics()["calls_by_block"], {"0": 1, "1": 1, "2": 1}
        )

    def test_module_op_preflight_is_atomic_on_structural_drift(self):
        model = FakeModel(2)
        original_first = model.transformer_blocks[0].attn1.attention_function
        model.transformer_blocks[1].attn1 = object()
        op = make_module_op(DensePassthroughBackend())

        with self.assertRaisesRegex(
            LTX2VideoAttentionIntegrationError,
            r"transformer_blocks\[1\]\.attn1",
        ):
            op.mutator(model)

        self.assertIs(
            model.transformer_blocks[0].attn1.attention_function,
            original_first,
        )

    def test_module_op_rejects_empty_missing_and_double_patch(self):
        op = make_module_op(DensePassthroughBackend())
        empty = FakeModel(0)
        with self.assertRaisesRegex(
            LTX2VideoAttentionIntegrationError, "is empty"
        ):
            op.mutator(empty)

        model = FakeModel(1)
        op.mutator(model)
        with self.assertRaisesRegex(
            LTX2VideoAttentionIntegrationError, "already patched"
        ):
            op.mutator(model)

    def test_builder_helper_appends_functionally_without_mutating_original(self):
        existing = FakeModuleOps("existing", lambda _: False, lambda module: module)
        builder = FakeBuilder((existing,))
        backend = DensePassthroughBackend()

        updated = with_ltx2_video_self_attention_builder(
            builder,
            backend,
            module_ops_factory=FakeModuleOps,
            model_type=FakeModel,
            attention_type=FakeAttention,
        )

        self.assertIsNot(updated, builder)
        self.assertEqual(builder.module_ops, (existing,))
        self.assertEqual(updated.module_ops[0], existing)
        self.assertEqual(
            updated.module_ops[1].name, "fasta2v_ltx2_video_self_attention"
        )

    def test_diffusion_stage_helper_replaces_only_its_builder_functionally(self):
        builder = FakeBuilder()
        stage = FakeStage(builder)
        updated = with_ltx2_video_self_attention(
            stage,
            DensePassthroughBackend(),
            module_ops_factory=FakeModuleOps,
            model_type=FakeModel,
            attention_type=FakeAttention,
        )

        self.assertIsNot(updated, stage)
        self.assertIs(stage._transformer_builder, builder)
        self.assertIsNot(updated._transformer_builder, builder)
        self.assertEqual(len(updated._transformer_builder.module_ops), 1)

    def test_builder_and_stage_interfaces_fail_loudly_on_drift(self):
        with self.assertRaisesRegex(
            LTX2VideoAttentionIntegrationError, "with_module_ops"
        ):
            with_ltx2_video_self_attention_builder(
                object(), DensePassthroughBackend()
            )
        with self.assertRaisesRegex(
            LTX2VideoAttentionIntegrationError, "DiffusionStage"
        ):
            with_ltx2_video_self_attention(
                object(), DensePassthroughBackend()
            )


if __name__ == "__main__":
    unittest.main()
