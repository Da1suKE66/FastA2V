"""Selective video self-attention adapters for official LTX-2.3.

The official LTX-2 attention callable consumes Q/K/V in flattened
``[B, T, H*D]`` form.  This module patches only
``model.transformer_blocks[i].attn1.attention_function`` and deliberately
leaves the masked callable and every audio, text, and audio/video cross
attention module untouched.

There is intentionally no top-level ``torch`` or ``ltx_core`` import.  The
adapter can therefore be imported and unit-tested without an LTX installation;
official ``ModuleOps``/model/attention types are imported lazily when a real
module operation is requested.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Callable, Iterable


OFFICIAL_LTX2_COMMIT = "9377758131b1ffde4b7f766804590a6617bf2ab9"
SPARGEATTN_API = "spas_sage2_attn_meansim_topk_cuda"


class LTX2VideoAttentionError(RuntimeError):
    """Base error for the selective LTX-2 video attention adapter."""


class LTX2VideoAttentionIntegrationError(LTX2VideoAttentionError):
    """Raised when the pinned official LTX model/builder shape has drifted."""


class LTX2VideoAttentionKernelError(LTX2VideoAttentionError):
    """Raised when SpargeAttn fails or returns an incompatible tensor."""


class LTX2VideoAttentionInputError(ValueError):
    """An input unsupported by the selected sparse kernel.

    ``reason`` is a stable, low-cardinality value used for explicit fallback
    accounting.  The human-readable exception message retains the exact input
    detail.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _shape(tensor: object, name: str) -> tuple[int, ...]:
    raw_shape = getattr(tensor, "shape", None)
    if raw_shape is None:
        raise LTX2VideoAttentionInputError(
            "shape", f"{name} must expose a tensor shape"
        )
    try:
        values = tuple(raw_shape)
    except TypeError as exc:
        raise LTX2VideoAttentionInputError(
            "shape", f"{name}.shape must be iterable"
        ) from exc

    shape: list[int] = []
    for dimension in values:
        if isinstance(dimension, bool):
            raise LTX2VideoAttentionInputError(
                "shape", f"{name}.shape contains boolean dimension {dimension!r}"
            )
        try:
            integer = operator.index(dimension)
        except TypeError as exc:
            raise LTX2VideoAttentionInputError(
                "shape", f"{name}.shape contains non-integer dimension {dimension!r}"
            ) from exc
        if integer <= 0:
            raise LTX2VideoAttentionInputError(
                "shape", f"{name}.shape dimensions must be positive, got {values!r}"
            )
        shape.append(integer)
    return tuple(shape)


def _dtype_name(tensor: object, name: str) -> str:
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        raise LTX2VideoAttentionInputError(
            "dtype", f"{name} must expose a tensor dtype"
        )
    return str(dtype)


def _device_identity(tensor: object, name: str) -> tuple[str, str]:
    device = getattr(tensor, "device", None)
    device_type = getattr(device, "type", None)
    if device is None or not isinstance(device_type, str) or not device_type:
        raise LTX2VideoAttentionInputError(
            "device", f"{name} must expose a typed tensor device"
        )
    return str(device), device_type


def _number_in_range(value: object, name: str, *, upper: float | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        range_text = "positive" if upper is None else "in (0, 1]"
        raise TypeError(f"{name} must be numeric and {range_text}")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        range_text = "positive" if upper is None else "in (0, 1]"
        raise ValueError(f"{name} must be {range_text}, got {value!r}")
    if upper is not None and normalized > upper:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    return normalized


def _positive_integer_or_none(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer or None")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer or None") from exc
    if integer <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return integer


def _positive_integer_tuple(values: Iterable[object], name: str) -> tuple[int, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of positive integers") from exc
    if not raw_values:
        raise ValueError(f"{name} must not be empty")
    normalized: list[int] = []
    for value in raw_values:
        integer = _positive_integer_or_none(value, name)
        assert integer is not None
        normalized.append(integer)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate values")
    return tuple(normalized)


class _BoundVideoAttention:
    """Official ``AttentionCallable``-shaped wrapper bound to one attn1 slot."""

    _ltx2_video_attention_bound = True

    def __init__(
        self,
        backend: object,
        original_attention: Callable,
        block_index: int | None,
    ) -> None:
        self.backend = backend
        self.original_attention = original_attention
        self.block_index = block_index
        self.label = getattr(backend, "label", type(backend).__name__)

    def __call__(self, q: object, k: object, v: object, heads: int) -> object:
        return self.backend._call(  # type: ignore[attr-defined]
            self.original_attention,
            q,
            k,
            v,
            heads,
            block_index=self.block_index,
        )


class DensePassthroughBackend:
    """Call and count the exact dense callable already installed by LTX.

    A backend is normally shared by every video ``attn1`` and bound to each
    slot with :meth:`bind`.  Passing ``original_attention`` also makes the
    backend directly callable, which is convenient for isolated checks.
    """

    label = "FastA2V-LTX2-DensePassthrough"
    method = "dense"

    def __init__(self, original_attention: Callable | None = None) -> None:
        if original_attention is not None and not callable(original_attention):
            raise TypeError("original_attention must be callable or None")
        self._original_attention = original_attention
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self._calls = 0
        self._calls_by_block: Counter[int | None] = Counter()

    def metrics(self) -> dict[str, object]:
        return {
            "backend": "dense_passthrough",
            "method": self.method,
            "calls": self._calls,
            "calls_by_block": {
                "unbound" if key is None else str(key): value
                for key, value in sorted(
                    self._calls_by_block.items(),
                    key=lambda item: (-1 if item[0] is None else item[0]),
                )
            },
            "fallback_allowed": False,
            "fallback_count": 0,
        }

    def bind(
        self, original_attention: Callable, *, block_index: int | None = None
    ) -> _BoundVideoAttention:
        if not callable(original_attention):
            raise TypeError("the original LTX attention function must be callable")
        return _BoundVideoAttention(self, original_attention, block_index)

    def _call(
        self,
        original_attention: Callable,
        q: object,
        k: object,
        v: object,
        heads: int,
        *,
        block_index: int | None,
    ) -> object:
        self._calls += 1
        self._calls_by_block[block_index] += 1
        return original_attention(q, k, v, heads)

    def __call__(self, q: object, k: object, v: object, heads: int) -> object:
        if self._original_attention is None:
            raise LTX2VideoAttentionIntegrationError(
                "DensePassthroughBackend must be bound to an original LTX "
                "attention callable before direct use"
            )
        return self._call(
            self._original_attention,
            q,
            k,
            v,
            heads,
            block_index=None,
        )


class SpargeVideoSelfAttentionBackend:
    """Adapter for the official SpargeAttn top-k NHD callable.

    Inputs follow LTX-2's unmasked ``AttentionCallable`` contract:
    ``q/k/v: [B, T, H*D]`` plus ``heads``.  The adapter validates that this is
    an unpadded CUDA self-attention shape, reshapes to ``[B, T, H, D]``, calls
    the injected official ``spas_sage2_attn_meansim_topk_cuda`` function with
    ``tensor_layout='NHD'``, validates its result, and flattens back to
    ``[B, T, H*D]``.

    Unsupported inputs raise by default.  ``fallback_to_dense=True`` is an
    explicit opt-in and is available only when the backend is bound to an
    original dense callable; every fallback is counted by a stable reason.
    Kernel exceptions and malformed kernel outputs never fall back.
    """

    label = "FastA2V-LTX2-SpargeAttn"
    method = "sparge"

    def __init__(
        self,
        *,
        kernel: Callable,
        topk: float = 0.5,
        pvthreshd: float = 50,
        smooth_k: bool = True,
        fallback_to_dense: bool = False,
        expected_heads: int | None = 32,
        supported_head_dims: Iterable[int] = (64, 128),
        allowed_dtypes: Iterable[object] = ("torch.bfloat16",),
        required_device_type: str = "cuda",
        min_sequence_length: int = 128,
    ) -> None:
        if not callable(kernel):
            raise TypeError("kernel must be the callable official SpargeAttn API")
        if smooth_k is not True:
            raise ValueError(
                "smooth_k must be true because the pinned official SpargeAttn "
                "top-k implementation uses km unconditionally"
            )
        if not isinstance(fallback_to_dense, bool):
            raise TypeError("fallback_to_dense must be a bool")
        if not isinstance(required_device_type, str) or not required_device_type:
            raise TypeError("required_device_type must be a non-empty string")

        self._kernel = kernel
        self.topk = _number_in_range(topk, "topk", upper=1.0)
        self.pvthreshd = _number_in_range(
            pvthreshd, "pvthreshd", upper=None
        )
        self.smooth_k = True
        self.fallback_to_dense = fallback_to_dense
        self.expected_heads = _positive_integer_or_none(
            expected_heads, "expected_heads"
        )
        self.supported_head_dims = _positive_integer_tuple(
            supported_head_dims, "supported_head_dims"
        )
        try:
            dtype_values = tuple(allowed_dtypes)
        except TypeError as exc:
            raise TypeError("allowed_dtypes must be an iterable") from exc
        if not dtype_values:
            raise ValueError("allowed_dtypes must not be empty")
        self.allowed_dtypes = tuple(str(value) for value in dtype_values)
        if any(not value for value in self.allowed_dtypes):
            raise ValueError("allowed_dtypes must not contain empty names")
        if len(set(self.allowed_dtypes)) != len(self.allowed_dtypes):
            raise ValueError("allowed_dtypes must not contain duplicates")
        self.required_device_type = required_device_type
        minimum = _positive_integer_or_none(
            min_sequence_length, "min_sequence_length"
        )
        assert minimum is not None
        self.min_sequence_length = minimum
        self._dense_fallback = DensePassthroughBackend()
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self._calls = 0
        self._sparse_calls = 0
        self._errors = 0
        self._fallback_count = 0
        self._fallback_reasons: Counter[str] = Counter()
        self._calls_by_block: Counter[int | None] = Counter()
        self._last_input_shape: tuple[int, ...] | None = None
        self._last_nhd_shape: tuple[int, ...] | None = None
        self._last_dtype: str | None = None
        self._last_device: str | None = None
        self._dense_fallback.reset_metrics()

    def metrics(self) -> dict[str, object]:
        return {
            "backend": "official_spargeattn",
            "method": self.method,
            "api": SPARGEATTN_API,
            "tensor_layout": "NHD",
            "topk": self.topk,
            "pvthreshd": self.pvthreshd,
            "smooth_k": self.smooth_k,
            "expected_heads": self.expected_heads,
            "supported_head_dims": list(self.supported_head_dims),
            "allowed_dtypes": list(self.allowed_dtypes),
            "required_device_type": self.required_device_type,
            "min_sequence_length": self.min_sequence_length,
            "calls": self._calls,
            "sparse_calls": self._sparse_calls,
            "errors": self._errors,
            "calls_by_block": {
                "unbound" if key is None else str(key): value
                for key, value in sorted(
                    self._calls_by_block.items(),
                    key=lambda item: (-1 if item[0] is None else item[0]),
                )
            },
            "fallback_allowed": self.fallback_to_dense,
            "fallback_used": self._fallback_count > 0,
            "fallback_count": self._fallback_count,
            "fallback_reasons": dict(sorted(self._fallback_reasons.items())),
            "dense_fallback": self._dense_fallback.metrics(),
            "last_input_shape": (
                list(self._last_input_shape)
                if self._last_input_shape is not None
                else None
            ),
            "last_nhd_shape": (
                list(self._last_nhd_shape)
                if self._last_nhd_shape is not None
                else None
            ),
            "last_dtype": self._last_dtype,
            "last_device": self._last_device,
        }

    def bind(
        self, original_attention: Callable, *, block_index: int | None = None
    ) -> _BoundVideoAttention:
        if not callable(original_attention):
            raise TypeError("the original LTX attention function must be callable")
        return _BoundVideoAttention(self, original_attention, block_index)

    def _validate_inputs(
        self,
        q: object,
        k: object,
        v: object,
        heads: object,
    ) -> tuple[int, int, int, int, str, str]:
        shapes = {
            "q": _shape(q, "q"),
            "k": _shape(k, "k"),
            "v": _shape(v, "v"),
        }
        if any(len(shape) != 3 for shape in shapes.values()):
            raise LTX2VideoAttentionInputError(
                "shape",
                "official LTX attention inputs must have shape [B, T, H*D]; "
                f"got {shapes}",
            )
        if len(set(shapes.values())) != 1:
            raise LTX2VideoAttentionInputError(
                "shape",
                "LTX video self-attention q/k/v shapes must match; "
                f"got {shapes}",
            )
        batch, sequence, hidden = shapes["q"]

        if isinstance(heads, bool):
            raise LTX2VideoAttentionInputError(
                "heads", f"heads must be a positive integer, got {heads!r}"
            )
        try:
            normalized_heads = operator.index(heads)
        except TypeError as exc:
            raise LTX2VideoAttentionInputError(
                "heads", f"heads must be a positive integer, got {heads!r}"
            ) from exc
        if normalized_heads <= 0:
            raise LTX2VideoAttentionInputError(
                "heads", f"heads must be positive, got {normalized_heads}"
            )
        if self.expected_heads is not None and normalized_heads != self.expected_heads:
            raise LTX2VideoAttentionInputError(
                "heads",
                f"LTX-2.3 video attention requires {self.expected_heads} heads; "
                f"got {normalized_heads}",
            )
        if hidden % normalized_heads:
            raise LTX2VideoAttentionInputError(
                "heads",
                f"hidden size {hidden} is not divisible by heads={normalized_heads}",
            )
        head_dim = hidden // normalized_heads
        if head_dim not in self.supported_head_dims:
            raise LTX2VideoAttentionInputError(
                "head_dim",
                "official SpargeAttn supports head dimensions "
                f"{self.supported_head_dims}; got {head_dim}",
            )
        if sequence < self.min_sequence_length:
            raise LTX2VideoAttentionInputError(
                "sequence_length",
                "official SpargeAttn requires sequence length >= "
                f"{self.min_sequence_length}; got {sequence}",
            )

        dtype_names = {
            name: _dtype_name(tensor, name)
            for name, tensor in (("q", q), ("k", k), ("v", v))
        }
        if len(set(dtype_names.values())) != 1:
            raise LTX2VideoAttentionInputError(
                "dtype", f"q/k/v dtypes must match; got {dtype_names}"
            )
        dtype_name = dtype_names["q"]
        if dtype_name not in self.allowed_dtypes:
            raise LTX2VideoAttentionInputError(
                "dtype",
                f"SpargeAttn dtype must be one of {self.allowed_dtypes}; "
                f"got {dtype_name!r}",
            )

        devices = {
            name: _device_identity(tensor, name)
            for name, tensor in (("q", q), ("k", k), ("v", v))
        }
        if len(set(devices.values())) != 1:
            raise LTX2VideoAttentionInputError(
                "device", f"q/k/v devices must match exactly; got {devices}"
            )
        device_name, device_type = devices["q"]
        if device_type != self.required_device_type:
            raise LTX2VideoAttentionInputError(
                "device",
                f"SpargeAttn requires device type {self.required_device_type!r}; "
                f"got {device_type!r}",
            )

        return (
            batch,
            sequence,
            hidden,
            head_dim,
            dtype_name,
            device_name,
        )

    def _validate_output(
        self,
        output: object,
        expected_shape: tuple[int, int, int, int],
        expected_dtype: str,
        expected_device: str,
    ) -> None:
        if isinstance(output, tuple):
            raise LTX2VideoAttentionKernelError(
                "official SpargeAttn returned a tuple despite "
                "return_sparsity=False"
            )
        try:
            output_shape = _shape(output, "SpargeAttn output")
            output_dtype = _dtype_name(output, "SpargeAttn output")
            output_device, _ = _device_identity(output, "SpargeAttn output")
        except LTX2VideoAttentionInputError as exc:
            raise LTX2VideoAttentionKernelError(str(exc)) from exc
        if output_shape != expected_shape:
            raise LTX2VideoAttentionKernelError(
                "official SpargeAttn output shape differs from NHD input: "
                f"output={output_shape}, input={expected_shape}"
            )
        if output_dtype != expected_dtype:
            raise LTX2VideoAttentionKernelError(
                "official SpargeAttn changed dtype: "
                f"output={output_dtype!r}, input={expected_dtype!r}"
            )
        if output_device != expected_device:
            raise LTX2VideoAttentionKernelError(
                "official SpargeAttn changed device: "
                f"output={output_device!r}, input={expected_device!r}"
            )

    def _call(
        self,
        original_attention: Callable | None,
        q: object,
        k: object,
        v: object,
        heads: int,
        *,
        block_index: int | None,
    ) -> object:
        self._calls += 1
        self._calls_by_block[block_index] += 1
        try:
            (
                batch,
                sequence,
                hidden,
                head_dim,
                dtype_name,
                device_name,
            ) = self._validate_inputs(q, k, v, heads)
        except LTX2VideoAttentionInputError as exc:
            if self.fallback_to_dense:
                if original_attention is None:
                    self._errors += 1
                    raise LTX2VideoAttentionIntegrationError(
                        "explicit dense fallback requires a backend bound to the "
                        "original LTX attention callable"
                    ) from exc
                self._fallback_count += 1
                self._fallback_reasons[exc.reason] += 1
                return self._dense_fallback._call(
                    original_attention,
                    q,
                    k,
                    v,
                    heads,
                    block_index=block_index,
                )
            self._errors += 1
            raise

        nhd_shape = (batch, sequence, operator.index(heads), head_dim)
        try:
            q_nhd = q.reshape(*nhd_shape)  # type: ignore[attr-defined]
            k_nhd = k.reshape(*nhd_shape)  # type: ignore[attr-defined]
            v_nhd = v.reshape(*nhd_shape)  # type: ignore[attr-defined]
        except Exception as exc:
            self._errors += 1
            raise LTX2VideoAttentionInputError(
                "shape", "q/k/v could not be reshaped from [B,T,H*D] to [B,T,H,D]"
            ) from exc

        try:
            output = self._kernel(
                q_nhd,
                k_nhd,
                v_nhd,
                dropout_p=0.0,
                is_causal=False,
                topk=self.topk,
                pvthreshd=self.pvthreshd,
                smooth_k=self.smooth_k,
                tensor_layout="NHD",
                return_sparsity=False,
            )
        except Exception as exc:
            self._errors += 1
            raise LTX2VideoAttentionKernelError(
                f"official {SPARGEATTN_API} failed"
            ) from exc

        try:
            self._validate_output(output, nhd_shape, dtype_name, device_name)
            flattened = output.reshape(batch, sequence, hidden)  # type: ignore[attr-defined]
        except LTX2VideoAttentionKernelError:
            self._errors += 1
            raise
        except Exception as exc:
            self._errors += 1
            raise LTX2VideoAttentionKernelError(
                "SpargeAttn output could not be flattened to [B,T,H*D]"
            ) from exc

        self._sparse_calls += 1
        self._last_input_shape = (batch, sequence, hidden)
        self._last_nhd_shape = nhd_shape
        self._last_dtype = dtype_name
        self._last_device = device_name
        return flattened

    def __call__(self, q: object, k: object, v: object, heads: int) -> object:
        """Run sparse attention directly (valid inputs do not need dense state)."""

        return self._call(None, q, k, v, heads, block_index=None)


def _load_official_ltx_types() -> tuple[object, type, type]:
    """Import the pinned official integration types only when they are needed."""

    try:
        from ltx_core.loader.module_ops import ModuleOps
        from ltx_core.model.transformer.attention import Attention
        from ltx_core.model.transformer.model import LTXModel
    except ImportError as exc:
        raise LTX2VideoAttentionIntegrationError(
            "official LTX-2 is not importable; install the checkout pinned at "
            f"{OFFICIAL_LTX2_COMMIT} or inject test types explicitly"
        ) from exc
    return ModuleOps, LTXModel, Attention


def create_ltx2_video_self_attention_module_ops(
    backend: DensePassthroughBackend | SpargeVideoSelfAttentionBackend,
    *,
    module_ops_factory: Callable | None = None,
    model_type: type | None = None,
    attention_type: type | None = None,
) -> object:
    """Create a builder ``ModuleOps`` targeting only LTX video ``attn1``.

    The three type/factory arguments exist solely to make the structural logic
    testable without importing LTX or torch.  Production callers omit them and
    receive the pinned official ``ModuleOps`` matcher/mutator.
    """

    bind = getattr(backend, "bind", None)
    if not callable(bind):
        raise TypeError("backend must expose bind(original_attention, block_index=...)")

    if (
        module_ops_factory is None
        or model_type is None
        or attention_type is None
    ):
        official_factory, official_model, official_attention = (
            _load_official_ltx_types()
        )
        module_ops_factory = module_ops_factory or official_factory  # type: ignore[assignment]
        model_type = model_type or official_model
        attention_type = attention_type or official_attention

    if not callable(module_ops_factory):
        raise TypeError("module_ops_factory must be callable")
    if not isinstance(model_type, type):
        raise TypeError("model_type must be a type")
    if not isinstance(attention_type, type):
        raise TypeError("attention_type must be a type")

    def matcher(module: object) -> bool:
        return isinstance(module, model_type)

    def mutator(module: object) -> object:
        blocks = getattr(module, "transformer_blocks", None)
        if blocks is None:
            raise LTX2VideoAttentionIntegrationError(
                "official LTX model is missing transformer_blocks"
            )
        try:
            block_list = list(blocks)
        except TypeError as exc:
            raise LTX2VideoAttentionIntegrationError(
                "official LTX transformer_blocks is not iterable"
            ) from exc
        if not block_list:
            raise LTX2VideoAttentionIntegrationError(
                "official LTX transformer_blocks is empty"
            )

        # Preflight the whole model before changing any slot, so a structural
        # mismatch cannot leave a partially patched transformer.
        slots: list[tuple[object, Callable, object, int]] = []
        for index, block in enumerate(block_list):
            attn1 = getattr(block, "attn1", None)
            if not isinstance(attn1, attention_type):
                raise LTX2VideoAttentionIntegrationError(
                    f"transformer_blocks[{index}].attn1 is not the pinned "
                    "official Attention type"
                )
            original = getattr(attn1, "attention_function", None)
            if not callable(original):
                raise LTX2VideoAttentionIntegrationError(
                    f"transformer_blocks[{index}].attn1.attention_function "
                    "is not callable"
                )
            if getattr(original, "_ltx2_video_attention_bound", False):
                raise LTX2VideoAttentionIntegrationError(
                    f"transformer_blocks[{index}].attn1 is already patched"
                )
            sentinel = object()
            masked = getattr(attn1, "masked_attention_function", sentinel)
            if masked is sentinel or not callable(masked):
                raise LTX2VideoAttentionIntegrationError(
                    f"transformer_blocks[{index}].attn1 is missing the official "
                    "masked attention callable"
                )
            slots.append((attn1, original, masked, index))

        replacements = [
            bind(original, block_index=index)
            for _attn1, original, _masked, index in slots
        ]
        if any(not callable(replacement) for replacement in replacements):
            raise LTX2VideoAttentionIntegrationError(
                "backend.bind returned a non-callable replacement"
            )

        for (attn1, _original, masked, index), replacement in zip(
            slots, replacements, strict=True
        ):
            setattr(attn1, "attention_function", replacement)
            if getattr(attn1, "masked_attention_function") is not masked:
                raise LTX2VideoAttentionIntegrationError(
                    f"transformer_blocks[{index}].attn1 masked path changed"
                )
        return module

    return module_ops_factory(
        name="fasta2v_ltx2_video_self_attention",
        matcher=matcher,
        mutator=mutator,
    )


def with_ltx2_video_self_attention_builder(
    builder: object,
    backend: DensePassthroughBackend | SpargeVideoSelfAttentionBackend,
    **module_op_kwargs: object,
) -> object:
    """Return an official builder copy with the selective ``ModuleOps`` appended."""

    with_module_ops = getattr(builder, "with_module_ops", None)
    if not callable(with_module_ops):
        raise LTX2VideoAttentionIntegrationError(
            "LTX transformer builder does not expose with_module_ops"
        )
    existing = getattr(builder, "module_ops", None)
    if existing is None:
        raise LTX2VideoAttentionIntegrationError(
            "LTX transformer builder does not expose module_ops"
        )
    try:
        existing_tuple = tuple(existing)
    except TypeError as exc:
        raise LTX2VideoAttentionIntegrationError(
            "LTX transformer builder module_ops is not iterable"
        ) from exc
    op = create_ltx2_video_self_attention_module_ops(
        backend, **module_op_kwargs
    )
    return with_module_ops((*existing_tuple, op))


def with_ltx2_video_self_attention(
    diffusion_stage: object,
    backend: DensePassthroughBackend | SpargeVideoSelfAttentionBackend,
    **module_op_kwargs: object,
) -> object:
    """Return a functional ``DiffusionStage`` copy using the selective adapter.

    Official ``DiffusionStage`` intentionally exposes no builder property, but
    stores it in ``_transformer_builder`` and provides the public functional
    ``with_builder`` method.  This helper is pinned to that official interface.
    """

    builder = getattr(diffusion_stage, "_transformer_builder", None)
    with_builder = getattr(diffusion_stage, "with_builder", None)
    if builder is None or not callable(with_builder):
        raise LTX2VideoAttentionIntegrationError(
            "object is not the pinned official DiffusionStage interface"
        )
    updated_builder = with_ltx2_video_self_attention_builder(
        builder, backend, **module_op_kwargs
    )
    return with_builder(updated_builder)


# Small singular alias for callers that use the generic builder-op vocabulary.
build_ltx2_video_attention_module_op = (
    create_ltx2_video_self_attention_module_ops
)


__all__ = [
    "DensePassthroughBackend",
    "LTX2VideoAttentionError",
    "LTX2VideoAttentionInputError",
    "LTX2VideoAttentionIntegrationError",
    "LTX2VideoAttentionKernelError",
    "OFFICIAL_LTX2_COMMIT",
    "SPARGEATTN_API",
    "SpargeVideoSelfAttentionBackend",
    "build_ltx2_video_attention_module_op",
    "create_ltx2_video_self_attention_module_ops",
    "with_ltx2_video_self_attention",
    "with_ltx2_video_self_attention_builder",
]
