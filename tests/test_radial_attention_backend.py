import base64
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zlib

from ovi.gpu_process_monitor import (
    TRUSTED_NVIDIA_SMI_BYTES,
    TRUSTED_NVIDIA_SMI_PATH,
    TRUSTED_NVIDIA_SMI_SHA256,
    build_pre_run_gpu_report,
    query_gpu_compute_processes,
)
from ovi.radial_evidence import (
    FLASHINFER_VERSION,
    FLASHINFER_WHEEL_BYTES,
    FLASHINFER_WHEEL_FILENAME,
    FLASHINFER_WHEEL_SHA256,
    FLASHINFER_WHEEL_URL,
    RADIAL_COMMIT,
    RADIAL_DERIVED_MODULE_SHA256,
    RADIAL_FORBIDDEN_LOADER_VARIABLES,
    RADIAL_QKV_STORAGE_BYTES,
    RADIAL_OPTIONAL_IMPORTS_PATCH_SHA256,
    RADIAL_PROFILE_AUDITS,
    RADIAL_SOURCE_MODULE_SHA256,
    expected_flashinfer_manifest,
    deterministic_ldd_environment,
    flashinfer_manifest_evidence_errors,
    ldd_resolved_libraries,
    ldd_resolved_library_paths,
    normalize_ldd_output,
    parse_nvidia_smi_pmon_output,
    radial_gpu_process_binding_errors,
    radial_profile,
    radial_ldd_search_paths,
    radial_microtest_evidence_errors,
    radial_receipt_evidence_errors,
    radial_runtime_loader_environment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = REPO_ROOT / "ovi" / "modules" / "radial_attention_backend.py"
SPEC = importlib.util.spec_from_file_location(
    "radial_attention_backend_under_test", BACKEND_PATH
)
BACKEND_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BACKEND_MODULE
SPEC.loader.exec_module(BACKEND_MODULE)

RadialAttentionInputError = BACKEND_MODULE.RadialAttentionInputError
RadialAttentionDependencyError = BACKEND_MODULE.RadialAttentionDependencyError
RadialVideoSelfAttentionBackend = BACKEND_MODULE.RadialVideoSelfAttentionBackend
audit_and_repair_radial_mask = BACKEND_MODULE.audit_and_repair_radial_mask
summarize_bool_rows = BACKEND_MODULE.summarize_bool_rows

_COMPRESSED_RAW_MASKS = {
    "aggressive": (
        "eNrFmtuOwzAIRJn//+l9SRNfwIDNeCOtqqpSj/AEhkkXeC6R35+8L++rRN4O1+LTH/Ih4gp1rPMK9YXK8/fSpH8NvZXYp2ivW7W2yGvUsU554FQqhqvrHAlR88JiomI+YfQYnNY6Iy/oqp/uTHM6J0d9UWe6IqXrJ+SRrkjVqiPJuuqnKzu6SpjaiWjpWl1rP3TbE/Z13aY2IuoTkVFrI2IHpTpdf8Ne0nVukwudE6jzzt5Ev363jXrCxgZz3DkPVNTOYemK9vtWuq4nYo6Kl9o6Kdvp0NTajj+u032Hd03Xbww6ulY63dsyza1Fn4hNn97TFcoX0XVtOqZrHG+DOXK6oU/7icjSFUsq6YTbPpXgRDymdn0qjWLMDWbuU1/X41o/a7uoK4wvSDpdjvrdu6J80atibVZvof1wszeY46zeQG1dqydiC9XOrMzpYEEtXas3mBGq6lrtdJg+u6ArrBvN0fUoq8Om8jYYmFS4E3G7XztrS+h65HS9tYlqWuVZfbA2r9YaXUdrc6kVnTNZm0W9ldXZSXz3hHez+mxt9ImoWtuervGsrlsb2el0ayM7nWVtVF0RfChUusHAowrB6YBgrZW6rqyNpuvS2ha6HmX1tbWRnM6xtgKnQ97aKE43PV1x1seSrO5b246ugVrlkJrunIi1lThd2tqkOKuHrC21+wcmYtDaaicicr931DjdBHWfjBRsMBqU7nSA9kCLrKsGDf7Us5/VVWiu1vwGo0OlQNdFVjegBbounM6CJp0ul9VNKHeDWfx6QtxgIDXUzAbzv0mcc8LaBrOGkiYisv++VLHBwKMKw+lQUmvW6ebURtdVTW171PgGo6e2FDWf1YffxM1mjj06DuqK2ICq1dVMbbnGyWV1O7VRna6Blver7XQIPv6tm4j9PCoYiJGsvk5tNF2XqY2zwbiPBhm6+qmtwOmmrO6mtvrOQSC11U/ESGord7pQahNCVne9TAhZPVJNvdMFWqM0q/8B5awQQA=="
    ),
    "conservative": (
        "eNrFmttuhDAMRD3//9Ot1G2BxIkvzKR5QYjdjJzBOTYAjMO+x+dgn8Pv2J8OY3MV8FShVfVF/0LVqI6CU4jDH/enlrsKd4h93YqqVNeiQl+xGve/BDJlY7FVHe9lM+e0HmsoqvB1ITj7us+cmip2I+8rSr4iVE35ilKsKVG2rxvBqq+WVkU0FDsiMqrwfW2rohQqKdZAUEM6ZAbbV2RVmZmDcqiqak09bovX3RHrsVZFKb7e58sQBxm0RarDjIdI91y8U6Sri76O9bPZJaoIIumuu/cg6WZ4tUmX97Uj+npHnFPhAOmuBLRWsK1Yn5tDbkd8TbohZc74+jMH7Kivf6t10tdrghZee6pOynRIV1O9MsaeK8wknde1zaUamXRe1/a8dIB0d1Hr7Yj1Xh3PldD5OnZtfgkurGAwXTvgKxY3mrSCwUpVWcEA61jpFcxjaveSsIJ5oG1KHBHpBrT5vrJJN6DN95VdwQS/PFqZHunEjetrGW1tX/M7ooe2pq/5Xt1Hm5h0PtrEpPPRJiYdcg+FuKRDpGoC0gHpWHmkW6NNSLod2t752kabiHR7tDFIhzLaNKRDrXzk+BqjTeJrhDYB6RJo45CujDY26VJoY5MuhTY26ZLbPZd0k2j4ZITgqycq9xXwHmiJffVEc696XvjqitZ2xHoF44uatoJZiGormJWolHRLUS3pNm9PhBUM7KVqw9f/7cTP+boXFVUwqH6+xKhgQlVNBcOKtQqd96od0hU+1WKQzu/aSqr1Xn14Eb9M5uyjY6t8+HPU12XXVkucmq/rrk27IyL7wQmbdHbc19wHJ8QKZt+1CX01rmrOV+OrxqFa5WUCi3T7rk1EumBeTeZESyghXTivCXr1kGUm6NUz0dB9zaQGtVf/AgfyFNs="
    ),
}


def fixture_rows(profile):
    raw = zlib.decompress(base64.b64decode(_COMPRESSED_RAW_MASKS[profile]))
    assert len(raw) == 117 * 117
    return [
        [bool(value) for value in raw[offset : offset + 117]]
        for offset in range(0, len(raw), 117)
    ]


class FakeMask:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)
        self.shape = (len(rows), len(rows[0]))

    def clone(self):
        return FakeMask(self.rows)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.rows)

    def __setitem__(self, key, value):
        row, columns = key
        if columns != slice(None):
            raise AssertionError(f"unexpected mask slice: {columns!r}")
        self.rows[row] = [bool(value)] * len(self.rows[row])


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

    def __getitem__(self, key):
        if isinstance(key, int):
            return FakeTensor(
                f"{self.label}[{key}]",
                self.shape[1:],
                device=self.device,
                dtype=self.dtype,
            )
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            length = len(range(start, stop, step))
            return FakeTensor(
                f"{self.label}[{start}:{stop}:{step}]",
                (length, *self.shape[1:]),
                device=self.device,
                dtype=self.dtype,
            )
        raise AssertionError(f"unexpected tensor key: {key!r}")

    def unsqueeze(self, dimension):
        if dimension != 0:
            raise AssertionError(f"unexpected unsqueeze: {dimension}")
        return FakeTensor(
            f"unsqueeze({self.label})",
            (1, *self.shape),
            device=self.device,
            dtype=self.dtype,
        )

    def flatten(self, start_dimension):
        if start_dimension != 2 or len(self.shape) != 4:
            raise AssertionError("unexpected flatten")
        return FakeTensor(
            f"flatten({self.label})",
            (self.shape[0], self.shape[1], self.shape[2] * self.shape[3]),
            device=self.device,
            dtype=self.dtype,
        )


class FakeHostTensor:
    def __init__(self, values):
        self.values = copy.deepcopy(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.values)


class FakeCuda:
    def __init__(self):
        self.synchronize_calls = []

    def synchronize(self, device):
        self.synchronize_calls.append(str(device))


class FakeTorch:
    uint8 = "torch.uint8"

    def __init__(self):
        self.empty_calls = []
        self.cuda = FakeCuda()

    def empty(self, size, *, device, dtype):
        self.empty_calls.append((size, str(device), dtype))
        return FakeTensor("workspace", (size,), device=device, dtype=dtype)

    def cat(self, tensors, dim):
        if dim != 0:
            raise AssertionError(f"unexpected cat dim: {dim}")
        tensors = tuple(tensors)
        return FakeTensor(
            "cat",
            (sum(item.shape[0] for item in tensors), *tensors[0].shape[1:]),
            device=tensors[0].device,
            dtype=tensors[0].dtype,
        )


class FakeWrapper:
    def __init__(self, workspace, *, backend):
        self.workspace = workspace
        self.backend = backend
        self.plan_kwargs = None
        self.run_calls = []

    def plan(self, **kwargs):
        self.plan_kwargs = kwargs

    def run(self, q, k, v, *, return_lse):
        self.run_calls.append((q, k, v, return_lse))
        return (
            FakeTensor("prefix_sparse", q.shape, device=q.device),
            FakeTensor("prefix_sparse_lse", q.shape[:2], device=q.device),
        )


class FakeFlashInfer:
    def __init__(self):
        self.wrappers = []
        self.dense_calls = []
        self.merge_calls = []

    def BlockSparseAttentionWrapper(self, workspace, *, backend):
        wrapper = FakeWrapper(workspace, backend=backend)
        self.wrappers.append(wrapper)
        return wrapper

    def single_prefill_with_kv_cache(self, *, q, k, v, causal, return_lse):
        self.dense_calls.append((q, k, v, causal, return_lse))
        output = FakeTensor("dense", q.shape, device=q.device)
        if return_lse:
            return output, FakeTensor("dense_lse", q.shape[:2], device=q.device)
        return output

    def merge_state(self, **kwargs):
        self.merge_calls.append(kwargs)
        return kwargs["v_a"], kwargs["s_a"]


class RecordingAttention:
    def __init__(self, *, shape=(1, 15004, 24, 128), use_sp=False):
        self.shape = shape
        self.use_sp = use_sp
        self.window_size = (-1, -1)
        self.projection_inputs = []

    def qkv_fn(self, hidden):
        return tuple(
            FakeTensor(label, self.shape, device=hidden.device)
            for label in ("q", "k", "v")
        )

    def o(self, value):
        self.projection_inputs.append(value)
        return FakeTensor("projected", value.shape, device=value.device)


def complete_receipt():
    root = f"/cache/liluchen/FastA2V"
    source = f"{root}/sources/radial-attention-{RADIAL_COMMIT}"
    derived = f"{root}/derived/radial-attention-{RADIAL_COMMIT}"
    flashinfer_root = (
        f"{root}/envs/ovi/lib/python3.11/site-packages/flashinfer"
    )
    flashinfer_init = {"bytes": 1, "sha256": "f" * 64}
    native_ldd = "libtorch.so => /fixed/libtorch.so (0x0000)\n"
    native_ldd_normalized = normalize_ldd_output(native_ldd)
    ldd_search_paths = list(radial_ldd_search_paths(root))
    return {
        "repository": "https://github.com/mit-han-lab/radial-attention.git",
        "clone_url": (
            "ssh://git@ssh.github.com:443/mit-han-lab/radial-attention.git"
        ),
        "commit": RADIAL_COMMIT,
        "mask_api": "gen_log_mask_shrinked",
        "source_dir": source,
        "derived_dir": derived,
        "source_module": {
            "path": f"{source}/radial_attn/attn_mask.py",
            "bytes": 1,
            "sha256": RADIAL_SOURCE_MODULE_SHA256,
        },
        "derived_module": {
            "path": f"{derived}/radial_attn/attn_mask.py",
            "bytes": 1,
            "sha256": RADIAL_DERIVED_MODULE_SHA256,
        },
        "optional_imports_patch": {
            "path": (
                "/workspace/liluchen/FastA2V/third_party/"
                "radial-attention-optional-imports.patch"
            ),
            "bytes": 1,
            "sha256": RADIAL_OPTIONAL_IMPORTS_PATCH_SHA256,
        },
        "patch_scope": ["radial_attn/attn_mask.py"],
        "patch_purpose": "optional_imports_only",
        "flashinfer_distribution": "flashinfer-python",
        "flashinfer_version": FLASHINFER_VERSION,
        "flashinfer_wheel_index": "https://flashinfer.ai/whl/cu124/torch2.6/",
        "flashinfer_wheel_url": FLASHINFER_WHEEL_URL,
        "flashinfer_wheel": {
            "path": f"{root}/wheels/{FLASHINFER_WHEEL_FILENAME}",
            "bytes": FLASHINFER_WHEEL_BYTES,
            "sha256": FLASHINFER_WHEEL_SHA256,
        },
        "flashinfer_required_apis": [
            "BlockSparseAttentionWrapper",
            "single_prefill_with_kv_cache",
            "merge_state",
        ],
        "cuda_home": "/usr/local/cuda-12.1",
        "ldd_executable": {
            "path": "/usr/bin/ldd",
            "bytes": 1,
            "sha256": "d" * 64,
        },
        "ldd_search_paths": ldd_search_paths,
        "ldd_dependencies": {
            "/fixed/libtorch.so": {
                "path": "/fixed/libtorch.so",
                "bytes": 1,
                "sha256": "c" * 64,
            }
        },
        "runtime_loaded_dependencies": {
            "flashinfer_kernels.so": [
                {
                    "path": f"{flashinfer_root}/flashinfer_kernels.so",
                    "bytes": 2,
                    "sha256": "a" * 64,
                }
            ],
            "libtorch.so": [
                {
                    "path": "/fixed/libtorch.so",
                    "bytes": 1,
                    "sha256": "c" * 64,
                }
            ],
        },
        "runtime_loader_environment": {
            "LD_LIBRARY_PATH": (
                ":".join(ldd_search_paths)
            ),
            "forbidden_prefixes": ["LD_"],
            "unset": list(RADIAL_FORBIDDEN_LOADER_VARIABLES),
        },
        "installed_flashinfer_package_root": flashinfer_root,
        "flashinfer_module": {
            "path": f"{flashinfer_root}/__init__.py",
            **flashinfer_init,
        },
        "installed_flashinfer_files": {
            "__init__.py": dict(flashinfer_init),
            "flashinfer_kernels.so": {
                "bytes": 2,
                "sha256": "a" * 64,
                "ldd_not_found": [],
                "ldd_output": native_ldd,
                "ldd_normalized_output": native_ldd_normalized,
                "ldd_sha256": hashlib.sha256(
                    native_ldd_normalized.encode("utf-8")
                ).hexdigest(),
                "ldd_dependency_paths": ["/fixed/libtorch.so"],
                "ldd_dependency_libraries": [
                    {"name": "libtorch.so", "path": "/fixed/libtorch.so"}
                ],
            },
        },
        "flashinfer_manifest": {
            "path": f"{root}/radial-flashinfer-manifest.json",
            "bytes": 1,
            "sha256": "e" * 64,
        },
        "python": "3.11.15",
        "torch": "2.6.0+cu124",
        "torch_cuda": "12.4",
        "model_type": "wan",
        "block_size": 128,
        "sequence": 15004,
        "prefix_sequence": 14976,
        "tail_sequence": 28,
        "grid": [31, 22, 22],
        "cpu_mask_audits": copy.deepcopy(RADIAL_PROFILE_AUDITS),
        "cuda_kernel_launched": False,
    }


class RadialMaskAuditTests(unittest.TestCase):
    def test_profile_copy_does_not_expose_global_empty_rows(self):
        profile = radial_profile("conservative")
        profile["empty_rows"].append(116)
        self.assertEqual(
            radial_profile("conservative")["empty_rows"], [22, 56, 90]
        )

    def test_installed_flashinfer_inventory_and_native_ldd_are_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            package_root = Path(directory) / "flashinfer"
            package_root.mkdir()
            init_path = package_root / "__init__.py"
            native_path = package_root / "kernels.so"
            torch_lib = Path(directory) / "torch" / "lib"
            cuda_home = (Path(directory) / "cuda").resolve()
            cuda_lib = cuda_home / "lib64"
            ldd_path = (Path(directory) / "ldd").resolve()
            dependency_path = (Path(directory) / "libtorch.so").resolve()
            init_path.write_bytes(b"init")
            native_path.write_bytes(b"native")
            torch_lib.mkdir(parents=True)
            cuda_lib.mkdir(parents=True)
            ldd_path.write_bytes(b"ldd")
            dependency_path.write_bytes(b"dependency")
            ldd_output = (
                f"libtorch.so => {dependency_path} (0x1234)\n"
            )
            normalized = normalize_ldd_output(ldd_output)

            def file_metadata(path):
                payload = path.read_bytes()
                return {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }

            receipt = {
                "cuda_home": str(cuda_home),
                "ldd_executable": {
                    "path": str(ldd_path),
                    **file_metadata(ldd_path),
                },
                "ldd_search_paths": [
                    str(torch_lib.resolve()),
                    str(cuda_lib.resolve()),
                ],
                "ldd_dependencies": {
                    str(dependency_path): {
                        "path": str(dependency_path),
                        **file_metadata(dependency_path),
                    }
                },
                "runtime_loader_environment": {
                    "LD_LIBRARY_PATH": (
                        f"{torch_lib.resolve()}:{cuda_lib.resolve()}"
                    ),
                    "forbidden_prefixes": ["LD_"],
                    "unset": list(RADIAL_FORBIDDEN_LOADER_VARIABLES),
                },
                "installed_flashinfer_package_root": str(package_root),
                "flashinfer_module": {
                    "path": str(init_path),
                    **file_metadata(init_path),
                },
                "installed_flashinfer_files": {
                    "__init__.py": file_metadata(init_path),
                    "kernels.so": {
                        **file_metadata(native_path),
                        "ldd_not_found": [],
                        "ldd_output": ldd_output,
                        "ldd_normalized_output": normalized,
                        "ldd_sha256": hashlib.sha256(
                            normalized.encode("utf-8")
                        ).hexdigest(),
                        "ldd_dependency_paths": [str(dependency_path)],
                    },
                },
            }
            with (
                mock.patch.object(
                    BACKEND_MODULE, "RADIAL_CUDA_HOME", str(cuda_home)
                ),
                mock.patch.object(
                    BACKEND_MODULE, "RADIAL_LDD_EXECUTABLE", str(ldd_path)
                ),
                mock.patch.object(
                    BACKEND_MODULE.subprocess,
                    "check_output",
                    return_value=ldd_output,
                ) as run_ldd,
                mock.patch.dict(
                    os.environ,
                    {
                        "CUDA_HOME": "/ambient/cuda",
                        "LD_LIBRARY_PATH": "/ambient/lib",
                        "LD_PRELOAD": "/ambient/preload.so",
                    },
                ),
            ):
                verified = BACKEND_MODULE._verify_installed_flashinfer_files(
                    receipt
                )
                self.assertEqual(verified, package_root.resolve())
                self.assertEqual(
                    run_ldd.call_args.kwargs["env"],
                    {
                        "PATH": "/usr/bin:/bin",
                        "LANG": "C",
                        "LC_ALL": "C",
                        "LD_LIBRARY_PATH": (
                            f"{torch_lib.resolve()}:{cuda_lib.resolve()}"
                        ),
                    },
                )
                dependency_path.write_bytes(b"dependency drift")
                with self.assertRaises(RadialAttentionDependencyError):
                    BACKEND_MODULE._verify_installed_flashinfer_files(receipt)
                dependency_path.write_bytes(b"dependency")
                (package_root / "unexpected.py").write_text("drift")
                with self.assertRaises(RadialAttentionDependencyError):
                    BACKEND_MODULE._verify_installed_flashinfer_files(receipt)

    def test_runtime_loader_contract_rejects_ambient_injection(self):
        receipt = complete_receipt()
        expected = receipt["runtime_loader_environment"]
        with mock.patch.dict(
            os.environ,
            {"LD_LIBRARY_PATH": expected["LD_LIBRARY_PATH"]},
            clear=True,
        ):
            self.assertEqual(
                BACKEND_MODULE.verify_radial_runtime_loader_environment(receipt),
                expected,
            )
            for variable in (
                "LD_PRELOAD",
                "LD_HWCAP_MASK",
                "GLIBC_TUNABLES",
            ):
                with self.subTest(variable=variable):
                    os.environ[variable] = "ambient"
                    with self.assertRaises(RadialAttentionDependencyError):
                        BACKEND_MODULE.verify_radial_runtime_loader_environment(
                            receipt
                        )
                    os.environ.pop(variable)

    def test_runtime_mappings_are_hashed_and_post_cuda_alias_drift_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            search_a = root / "search-a"
            search_b = root / "search-b"
            allowed = root / "allowed"
            for path in (search_a, search_b, allowed):
                path.mkdir()
            dependency = root / "libfoo.so.1"
            conflict = root / "libfoo.so.1.9"
            dependency.write_bytes(b"fixed")
            conflict.write_bytes(b"conflict")

            def metadata(path):
                payload = path.read_bytes()
                return {
                    "path": str(path),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }

            search_paths = [str(search_a), str(search_b)]
            receipt = {
                "ldd_search_paths": search_paths,
                "runtime_loader_environment": (
                    radial_runtime_loader_environment(search_paths)
                ),
                "runtime_loaded_dependencies": {
                    "libfoo.so.1": [metadata(dependency)]
                },
            }
            with mock.patch.object(
                BACKEND_MODULE,
                "loaded_shared_object_paths",
                return_value=(str(dependency),),
            ):
                evidence = (
                    BACKEND_MODULE.verify_radial_runtime_loaded_dependencies(
                        receipt
                    )
                )
                self.assertEqual(evidence["status"], "ok")
                with mock.patch.dict(
                    os.environ,
                    {
                        "LD_LIBRARY_PATH": (
                            f"{allowed}:{':'.join(search_paths)}"
                        )
                    },
                    clear=True,
                ):
                    restored = (
                        BACKEND_MODULE.restore_radial_loader_after_preloaded_optional_imports(
                            receipt,
                            allowed_prepend_paths=(allowed,),
                        )
                    )
                    self.assertTrue(restored["restored"])
                    self.assertEqual(
                        os.environ["LD_LIBRARY_PATH"], ":".join(search_paths)
                    )
            with mock.patch.object(
                BACKEND_MODULE,
                "loaded_shared_object_paths",
                return_value=(str(dependency), str(conflict)),
            ):
                with self.assertRaises(RadialAttentionDependencyError):
                    BACKEND_MODULE.verify_radial_runtime_loaded_dependencies(
                        receipt
                    )

    def test_ldd_helpers_reject_ambiguous_search_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            canonical = str(Path(directory).resolve())
            self.assertEqual(
                deterministic_ldd_environment([canonical])["LD_LIBRARY_PATH"],
                canonical,
            )
        for invalid in (
            [],
            [""],
            ["relative"],
            ["/absolute:split"],
            ["/tmp/../tmp"],
            ["/tmp", "/tmp"],
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    deterministic_ldd_environment(invalid)
        self.assertEqual(
            ldd_resolved_library_paths(
                "linux-vdso.so.1 (0x1)\n"
                "libtorch.so => /fixed/libtorch.so (0x2)\n"
                "/lib64/ld-linux-x86-64.so.2 (0x3)\n"
            ),
            ("/fixed/libtorch.so", "/lib64/ld-linux-x86-64.so.2"),
        )
        self.assertEqual(
            ldd_resolved_libraries(
                "linux-vdso.so.1 (0x1)\n"
                "libtorch.so => /fixed/libtorch.so (0x2)\n"
                "/lib64/ld-linux-x86-64.so.2 (0x3)\n"
            ),
            (
                ("libtorch.so", "/fixed/libtorch.so"),
                ("ld-linux-x86-64.so.2", "/lib64/ld-linux-x86-64.so.2"),
            ),
        )

    def test_cpu_fixtures_bind_hash_count_and_official_empty_rows(self):
        for profile, expected in RADIAL_PROFILE_AUDITS.items():
            with self.subTest(profile=profile):
                summary = summarize_bool_rows(fixture_rows(profile))
                self.assertEqual(summary["true_blocks"], expected["raw_true_blocks"])
                self.assertEqual(summary["sha256"], expected["raw_sha256"])
                self.assertEqual(summary["empty_rows"], [22, 56, 90])

    def test_empty_rows_are_made_dense_and_repaired_hash_is_fixed(self):
        for profile, expected in RADIAL_PROFILE_AUDITS.items():
            with self.subTest(profile=profile):
                repaired, audit = audit_and_repair_radial_mask(
                    FakeMask(fixture_rows(profile)), profile
                )
                self.assertEqual(audit, expected)
                repaired_summary = summarize_bool_rows(repaired.tolist())
                self.assertEqual(repaired_summary["empty_rows"], [])
                self.assertEqual(
                    repaired_summary["true_blocks"],
                    expected["repaired_true_blocks"],
                )
                self.assertEqual(
                    repaired_summary["sha256"], expected["repaired_sha256"]
                )

    def test_receipt_validator_rejects_dependency_or_mask_drift(self):
        receipt = complete_receipt()
        self.assertEqual(radial_receipt_evidence_errors(receipt), [])
        ldd_drift = copy.deepcopy(receipt)
        ldd_drift["ldd_search_paths"].append("/ambient/lib")
        self.assertTrue(
            any(
                "ldd_search_paths" in error
                for error in radial_receipt_evidence_errors(ldd_drift)
            )
        )
        manifest = expected_flashinfer_manifest(receipt)
        self.assertEqual(
            flashinfer_manifest_evidence_errors(manifest, receipt), []
        )
        manifest["version"] = "drifted"
        self.assertTrue(
            flashinfer_manifest_evidence_errors(manifest, receipt)
        )
        receipt["commit"] = "0" * 40
        receipt["cpu_mask_audits"]["aggressive"]["raw_true_blocks"] += 1
        errors = radial_receipt_evidence_errors(receipt)
        self.assertTrue(any("commit" in error for error in errors))
        self.assertTrue(any("CPU mask audit" in error for error in errors))

    def test_receipt_accepts_legitimate_zero_byte_package_markers(self):
        receipt = complete_receipt()
        receipt["installed_flashinfer_files"]["py.typed"] = {
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
        self.assertEqual(radial_receipt_evidence_errors(receipt), [])

    def test_radial_preflight_passes_persisted_idle_evidence_path(self):
        preflight_source = (
            REPO_ROOT / "scripts" / "preflight_ovi.py"
        ).read_text(encoding="utf-8")
        microtest_source = (
            REPO_ROOT / "scripts" / "radial_flashinfer_microtest.py"
        ).read_text(encoding="utf-8")
        self.assertIn("pre_run_gpu_path=pre_run_path", preflight_source)
        self.assertIn("pre_run_gpu_path=pre_run_path", microtest_source)
        self.assertIn("_read_and_bind_pre_run_gpu", microtest_source)

    def test_pmon_parser_uses_dynamic_header_and_retains_idle_rows(self):
        parsed = parse_nvidia_smi_pmon_output(
            "# gpu pid type sm mem enc dec jpg ofa command\n"
            "0 - - - - - - - - -\n"
            "0 42 C 1 2 - - - - python\n"
        )
        self.assertEqual(parsed["errors"], [])
        self.assertEqual(parsed["header_columns"][-3:], ["jpg", "ofa", "command"])
        self.assertIsNone(parsed["rows"][0]["host_pid"])
        self.assertEqual(parsed["rows"][1]["host_pid"], 42)
        self.assertEqual(parsed["rows"][1]["process_type"], "C")
        self.assertTrue(
            parse_nvidia_smi_pmon_output("0 42 C python\n")["errors"]
        )

    def test_cuda_microtest_evidence_is_cross_bound_and_finite(self):
        gpu_uuid = "GPU-11111111-2222-3333-4444-555555555555"
        boot_id = "11111111-2222-3333-4444-555555555555"
        host_pid = 9876
        container_pid = 4321
        binary_metadata = {
            "requested_path": TRUSTED_NVIDIA_SMI_PATH,
            "resolved_path": TRUSTED_NVIDIA_SMI_PATH,
            "owner_uid": 0,
            "owner_gid": 0,
            "mode": 0o755,
            "device": 2050,
            "inode": 2490545,
            "bytes": TRUSTED_NVIDIA_SMI_BYTES,
            "sha256": TRUSTED_NVIDIA_SMI_SHA256,
        }

        def gpu_sample(start, count):
            outputs = iter(
                (
                    f"0, {gpu_uuid}, NVIDIA A100-SXM4-80GB\n",
                    (
                        f"{host_pid}, [Not Found], 600\n"
                        if count
                        else ""
                    ),
                )
            )
            sample = query_gpu_compute_processes(
                0,
                command_fn=lambda _command: next(outputs),
                include_process_name=True,
                binary_metadata_fn=lambda: copy.deepcopy(binary_metadata),
            )
            sample["boot_id"] = boot_id
            receipt = sample["query_receipt"]
            for clock in ("unix", "monotonic"):
                sample[f"sampled_at_{clock}_seconds"] = start
                sample[f"query_started_at_{clock}_seconds"] = start
                sample[f"query_finished_at_{clock}_seconds"] = start + 0.01
                receipt[f"query_started_at_{clock}_seconds"] = start
                receipt[f"query_finished_at_{clock}_seconds"] = start + 0.01
                receipt["commands"][0][f"started_at_{clock}_seconds"] = start
                receipt["commands"][0][f"finished_at_{clock}_seconds"] = (
                    start + 0.004
                )
                receipt["commands"][1][f"started_at_{clock}_seconds"] = (
                    start + 0.005
                )
                receipt["commands"][1][f"finished_at_{clock}_seconds"] = (
                    start + 0.009
                )
            return sample

        pre_run_gpu = build_pre_run_gpu_report(
            gpu_sample(100.0, 0),
            cuda_visible_devices=gpu_uuid,
        )
        pre_run_gpu["checked_at_utc"] = "2026-07-14T00:00:00+00:00"
        pre_run_gpu["run_nonce"] = "a" * 32
        raw_stdout = (
            "#Date Time gpu pid type sm mem enc dec command\n"
            "#YYYYMMDD HH:MM:SS Idx # C/G % % % % name\n"
            "19700101 00:01:42 0 - - - - - - -\n"
            "19700101 00:01:43 0 9876 C 5 4 - - [Not Found]\n"
            "19700101 00:01:44 0 9876 C 0 4 - - [Not Found]\n"
        )
        parsed_pmon = parse_nvidia_smi_pmon_output(raw_stdout)
        self.assertEqual(parsed_pmon["errors"], [])
        rows_by_line = {
            row["line_number"]: row for row in parsed_pmon["rows"]
        }
        line_times = [101.12, 101.13, 102.1, 103.2, 104.1]
        line_records = [
            {
                "line_index": index,
                "raw_line": line,
                "received_at_unix_seconds": line_times[index],
                "received_at_monotonic_seconds": line_times[index],
                "parsed_row": rows_by_line.get(index + 1),
            }
            for index, line in enumerate(raw_stdout.splitlines(keepends=True))
        ]
        pmon = {
            "status": "ok",
            "collection_status": "ok",
            "observation_mode": "direct_c_observed",
            "direct_compute_type_observed": True,
            "host_pid_observed": True,
            "continuous_exclusivity_proven": False,
            "compute_observation_timeout_seconds": 20.0,
            "command": [
                "/usr/bin/nvidia-smi",
                "pmon",
                "-i",
                "0",
                "-s",
                "um",
                "-d",
                "1",
                "-o",
                "DT",
            ],
            "resolved_executable": "/usr/bin/nvidia-smi",
            "nvidia_smi_binary": copy.deepcopy(binary_metadata),
            "locale": {"LC_ALL": "C", "LANG": "C", "TZ": "UTC"},
            "process_pid": 5555,
            "raw_stdout": raw_stdout,
            "raw_stderr": "",
            "raw_stdout_bytes": len(raw_stdout.encode()),
            "raw_stdout_sha256": hashlib.sha256(raw_stdout.encode()).hexdigest(),
            "raw_stderr_bytes": 0,
            "raw_stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "exit_code": -15,
            "termination_method": "terminate",
            "timed_out": False,
            "header_columns": parsed_pmon["header_columns"],
            "rows": parsed_pmon["rows"],
            "parser_errors": [],
            "stdout_line_records": line_records,
            "stderr_line_records": [],
            "errors": [],
            "expected_host_pid": host_pid,
            "spawn_started_at_unix_seconds": 101.1,
            "spawn_started_at_monotonic_seconds": 101.1,
            "process_started_at_unix_seconds": 101.11,
            "process_started_at_monotonic_seconds": 101.11,
            "header_ready_at_unix_seconds": 101.12,
            "header_ready_at_monotonic_seconds": 101.12,
            "idle_baseline_ready_at_unix_seconds": 102.1,
            "idle_baseline_ready_at_monotonic_seconds": 102.1,
            "idle_baseline_line_number": 3,
            "host_pid_bound_at_unix_seconds": 102.62,
            "host_pid_bound_at_monotonic_seconds": 102.62,
            "backend_window_started_at_unix_seconds": 102.7,
            "backend_window_started_at_monotonic_seconds": 102.7,
            "backend_window_start_line_number": 3,
            "window_compute_ready_at_unix_seconds": 103.2,
            "window_compute_ready_at_monotonic_seconds": 103.2,
            "window_compute_line_number": 4,
            "compute_observation_deadline_reached_at_unix_seconds": None,
            "compute_observation_deadline_reached_at_monotonic_seconds": None,
            "final_sync_covered_at_unix_seconds": 104.1,
            "final_sync_covered_at_monotonic_seconds": 104.1,
            "final_sync_covered_line_number": 5,
            "stop_requested_at_unix_seconds": 104.2,
            "stop_requested_at_monotonic_seconds": 104.2,
            "process_exited_at_unix_seconds": 104.25,
            "process_exited_at_monotonic_seconds": 104.25,
        }
        binding = {
            "schema_version": 2,
            "binding_method": "sampled_temporal_association_after_idle_guard",
            "claim_scope": (
                "sampled_temporal_association_not_pid_ownership_or_"
                "continuous_exclusivity"
            ),
            "pmon_observation_mode": "direct_c_observed",
            "host_pid_ownership": (
                "unknown_sampled_temporal_association_only"
            ),
            "pre_run_gpu_path": "/runs/test/pre_run_gpu.json",
            "pre_run_gpu_sha256": "b" * 64,
            "pre_run_gpu": copy.deepcopy(pre_run_gpu),
            "microtest_started_at_unix_seconds": 100.5,
            "microtest_started_at_monotonic_seconds": 100.5,
            "cuda_touch_started_at_unix_seconds": 102.2,
            "cuda_touch_started_at_monotonic_seconds": 102.2,
            "setup_cuda_synchronize_started_at_unix_seconds": 102.3,
            "setup_cuda_synchronize_started_at_monotonic_seconds": 102.3,
            "setup_cuda_synchronized_at_unix_seconds": 102.5,
            "setup_cuda_synchronized_at_monotonic_seconds": 102.5,
            "exact_backend_started_at_unix_seconds": 102.7,
            "exact_backend_started_at_monotonic_seconds": 102.7,
            "exact_backend_returned_at_unix_seconds": 103.5,
            "exact_backend_returned_at_monotonic_seconds": 103.5,
            "cuda_synchronize_started_at_unix_seconds": 103.6,
            "cuda_synchronize_started_at_monotonic_seconds": 103.6,
            "cuda_synchronized_at_unix_seconds": 103.7,
            "cuda_synchronized_at_monotonic_seconds": 103.7,
            "exact_kernel_completed": True,
            "exact_backend_call_count": 2,
            "cuda_synchronize_completed": True,
            "current_cuda_device_uuid": gpu_uuid,
            "current_cuda_device_name": "NVIDIA A100-SXM4-80GB",
            "current_cuda_device_index": 0,
            "cuda_visible_devices": gpu_uuid,
            "qkv_storage_bytes": RADIAL_QKV_STORAGE_BYTES,
            "allocator_memory_bytes": RADIAL_QKV_STORAGE_BYTES + 1024,
            "reserved_memory_bytes": 512 * 1024 * 1024,
            "python_executable": "/cache/liluchen/FastA2V/envs/ovi/bin/python",
            "python_executable_resolved": (
                "/cache/liluchen/FastA2V/envs/ovi/bin/python3.11"
            ),
            "container_pid": container_pid,
            "nvidia_smi_host_pid": host_pid,
            "nspid": {"status": "ok", "error": None, "chain": [container_pid]},
            "host_pid_proc_visibility": {"status": "not_visible", "error": None},
            "mps": {
                "cuda_mps_environment_variables": [],
                "pmon": pmon,
                "mps_status": "not_observed",
                "direct_compute_type_observed": True,
                "host_pid_observed_by_pmon": True,
                "continuous_exclusivity_proven": False,
            },
            "immediate_pre_cuda_sample": gpu_sample(101.0, 0),
            "context_live_sample": gpu_sample(102.6, 1),
            "interval_seconds": 0.1,
            "max_query_gap_seconds": 1.0,
            "minimum_backend_query_samples": 2,
            "interval_samples": [
                gpu_sample(101.0, 0),
                gpu_sample(102.6, 1),
                gpu_sample(102.8, 1),
                gpu_sample(103.0, 1),
                gpu_sample(103.8, 1),
                gpu_sample(104.7, 1),
            ],
            "post_cuda_samples": [
                gpu_sample(104.3, 1),
                gpu_sample(104.5, 1),
            ],
        }
        evidence = {
            "status": "ok",
            "device": "NVIDIA A100-SXM4-80GB",
            "device_uuid": gpu_uuid,
            "cuda_visible_devices": gpu_uuid,
            "physical_device_index": 0,
            "logical_cuda_device_index": 0,
            "host_pid": host_pid,
            "python_pid": container_pid,
            "pid_namespace_chain": [container_pid],
            "host_pid_namespace_visible": False,
            "pid_binding_method": (
                "sampled_temporal_association_after_idle_guard"
            ),
            "pmon_observation_mode": "direct_c_observed",
            "gpu_process_claim_scope": (
                "sampled_temporal_association_not_pid_ownership_or_"
                "continuous_exclusivity"
            ),
            "host_pid_ownership": (
                "unknown_sampled_temporal_association_only"
            ),
            "mps_status": "not_observed",
            "pre_run_gpu": copy.deepcopy(pre_run_gpu),
            "pre_run_gpu_sha256": "b" * 64,
            "post_cuda_sampled_at_unix_seconds": 104.5,
            "gpu_process_count": 1,
            "gpu_processes": [
                {
                    "host_pid": host_pid,
                    "process_name": "[Not Found]",
                    "used_memory_mib": 600,
                }
            ],
            "cuda_synchronized": True,
            "gpu_process_binding": copy.deepcopy(binding),
            "compute_capability": [8, 0],
            "torch": "2.6.0+cu124",
            "torch_cuda": "12.4",
            "torch_cxx11_abi": False,
            "dtype": "torch.bfloat16",
            "shape": [1, 15004, 24, 128],
            "grid": [31, 22, 22],
            "profile": "conservative",
            "decay_factor": 4.0,
            "prefix_sequence": 14976,
            "tail_sequence": 28,
            "tail_strategy": "dense_lse_merge_no_padding",
            "exact_backend_call_count": 2,
            "calls": 2,
            "plan_cache_entries": 1,
            "plan_cache_misses": 1,
            "plan_cache_hits": 1,
            "mask_audit": copy.deepcopy(
                RADIAL_PROFILE_AUDITS["conservative"]
            ),
            "finite": True,
            "runtime_dependencies_before_cuda": {
                "status": "ok",
                "aliases": 2,
                "mapped_files": 3,
                "inventory_sha256": "b" * 64,
            },
            "runtime_dependencies_after_cuda": {
                "status": "ok",
                "aliases": 2,
                "mapped_files": 3,
                "inventory_sha256": "b" * 64,
            },
            "output_abs_mean": 0.5,
            "output_abs_max": 4.0,
        }
        self.assertEqual(
            radial_microtest_evidence_errors(
                evidence,
                expected_gpu_uuid=gpu_uuid,
                expected_pre_run_gpu=pre_run_gpu,
                expected_pre_run_gpu_sha256="b" * 64,
                expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
                expected_python_executable=(
                    "/cache/liluchen/FastA2V/envs/ovi/bin/python"
                ),
            ),
            [],
        )

        all_idle_binding = copy.deepcopy(binding)
        all_idle_binding.update(
            {
                "pmon_observation_mode": (
                    "pmon_reported_all_idle_during_audited_window"
                ),
                "exact_backend_returned_at_unix_seconds": 123.0,
                "exact_backend_returned_at_monotonic_seconds": 123.0,
                "cuda_synchronize_started_at_unix_seconds": 123.1,
                "cuda_synchronize_started_at_monotonic_seconds": 123.1,
                "cuda_synchronized_at_unix_seconds": 123.2,
                "cuda_synchronized_at_monotonic_seconds": 123.2,
                "interval_samples": [
                    gpu_sample(101.0, 0),
                    gpu_sample(102.6, 1),
                    *[
                        gpu_sample(102.8 + 0.5 * index, 1)
                        for index in range(43)
                    ],
                    gpu_sample(124.7, 1),
                ],
                "post_cuda_samples": [
                    gpu_sample(124.3, 1),
                    gpu_sample(124.5, 1),
                ],
            }
        )
        idle_rows_text = "".join(
            (
                f"19700101 00:{second // 60:02d}:{second % 60:02d} "
                "0 - - - - - - -\n"
            )
            for second in range(102, 125)
        )
        all_idle_raw = (
            "#Date Time gpu pid type sm mem enc dec command\n"
            "#YYYYMMDD HH:MM:SS Idx # C/G % % % % name\n"
            + idle_rows_text
        )
        parsed_all_idle = parse_nvidia_smi_pmon_output(all_idle_raw)
        self.assertEqual(parsed_all_idle["errors"], [])
        all_idle_rows_by_line = {
            row["line_number"]: row for row in parsed_all_idle["rows"]
        }
        all_idle_line_times = [
            101.12,
            101.13,
            *[second + 0.1 for second in range(102, 125)],
        ]
        all_idle_line_records = [
            {
                "line_index": index,
                "raw_line": line,
                "received_at_unix_seconds": all_idle_line_times[index],
                "received_at_monotonic_seconds": all_idle_line_times[index],
                "parsed_row": all_idle_rows_by_line.get(index + 1),
            }
            for index, line in enumerate(
                all_idle_raw.splitlines(keepends=True)
            )
        ]
        all_idle_pmon = all_idle_binding["mps"]["pmon"]
        all_idle_pmon.update(
            {
                "status": "degraded",
                "collection_status": "ok",
                "observation_mode": (
                    "pmon_reported_all_idle_during_audited_window"
                ),
                "direct_compute_type_observed": False,
                "host_pid_observed": False,
                "raw_stdout": all_idle_raw,
                "raw_stdout_bytes": len(all_idle_raw.encode()),
                "raw_stdout_sha256": hashlib.sha256(
                    all_idle_raw.encode()
                ).hexdigest(),
                "header_columns": parsed_all_idle["header_columns"],
                "rows": parsed_all_idle["rows"],
                "parser_errors": [],
                "stdout_line_records": all_idle_line_records,
                "window_compute_ready_at_unix_seconds": None,
                "window_compute_ready_at_monotonic_seconds": None,
                "window_compute_line_number": None,
                "compute_observation_deadline_reached_at_unix_seconds": 122.8,
                "compute_observation_deadline_reached_at_monotonic_seconds": (
                    122.8
                ),
                "final_sync_covered_at_unix_seconds": 124.1,
                "final_sync_covered_at_monotonic_seconds": 124.1,
                "final_sync_covered_line_number": 25,
                "stop_requested_at_unix_seconds": 124.2,
                "stop_requested_at_monotonic_seconds": 124.2,
                "process_exited_at_unix_seconds": 124.25,
                "process_exited_at_monotonic_seconds": 124.25,
                "errors": [],
            }
        )
        all_idle_binding["mps"].update(
            {
                "mps_status": "unknown",
                "direct_compute_type_observed": False,
                "host_pid_observed_by_pmon": False,
            }
        )
        all_idle_evidence = copy.deepcopy(evidence)
        all_idle_evidence.update(
            {
                "pmon_observation_mode": (
                    "pmon_reported_all_idle_during_audited_window"
                ),
                "mps_status": "unknown",
                "post_cuda_sampled_at_unix_seconds": 124.5,
                "gpu_process_binding": all_idle_binding,
            }
        )
        self.assertEqual(
            radial_microtest_evidence_errors(
                all_idle_evidence,
                expected_gpu_uuid=gpu_uuid,
                expected_pre_run_gpu=pre_run_gpu,
                expected_pre_run_gpu_sha256="b" * 64,
                expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
                expected_python_executable=(
                    "/cache/liluchen/FastA2V/envs/ovi/bin/python"
                ),
            ),
            [],
        )

        microtest_bool_mutations = (
            (
                "physical_device_index",
                lambda item: item.__setitem__("physical_device_index", False),
            ),
            (
                "logical_cuda_device_index",
                lambda item: item.__setitem__(
                    "logical_cuda_device_index", False
                ),
            ),
            (
                "gpu_process_count",
                lambda item: item.__setitem__("gpu_process_count", True),
            ),
            ("finite", lambda item: item.__setitem__("finite", 1)),
            (
                "compute_capability",
                lambda item: item.__setitem__(
                    "compute_capability", [8, False]
                ),
            ),
            (
                "cuda_synchronized",
                lambda item: item.__setitem__("cuda_synchronized", 1),
            ),
            (
                "torch_cxx11_abi",
                lambda item: item.__setitem__("torch_cxx11_abi", 0),
            ),
            (
                "shape",
                lambda item: item.__setitem__(
                    "shape", [True, 15004, 24, 128]
                ),
            ),
            (
                "plan_cache_entries",
                lambda item: item.__setitem__("plan_cache_entries", True),
            ),
            (
                "plan_cache_misses",
                lambda item: item.__setitem__("plan_cache_misses", True),
            ),
            (
                "plan_cache_hits",
                lambda item: item.__setitem__("plan_cache_hits", True),
            ),
            (
                "output_abs_mean",
                lambda item: item.__setitem__("output_abs_mean", True),
            ),
            (
                "runtime_aliases",
                lambda item: (
                    item["runtime_dependencies_before_cuda"].__setitem__(
                        "aliases", True
                    ),
                    item["runtime_dependencies_after_cuda"].__setitem__(
                        "aliases", True
                    ),
                ),
            ),
        )
        for label, mutate in microtest_bool_mutations:
            with self.subTest(bool_forgery=label):
                mutated = copy.deepcopy(evidence)
                mutate(mutated)
                mutation_errors = radial_microtest_evidence_errors(
                    mutated,
                    expected_gpu_uuid=gpu_uuid,
                    expected_pre_run_gpu=pre_run_gpu,
                    expected_pre_run_gpu_sha256="b" * 64,
                    expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
                    expected_python_executable=(
                        "/cache/liluchen/FastA2V/envs/ovi/bin/python"
                    ),
                )
                self.assertTrue(mutation_errors, label)

        binding_bool_mutations = (
            (
                "current_cuda_device_index",
                binding,
                lambda item: item.__setitem__(
                    "current_cuda_device_index", False
                ),
            ),
            (
                "max_query_gap_seconds",
                binding,
                lambda item: item.__setitem__("max_query_gap_seconds", True),
            ),
            (
                "all_idle_backend_window_start_line_number",
                all_idle_binding,
                lambda item: item["mps"]["pmon"].__setitem__(
                    "backend_window_start_line_number", False
                ),
            ),
        )
        for label, source, mutate in binding_bool_mutations:
            with self.subTest(bool_forgery=label):
                mutated = copy.deepcopy(source)
                mutate(mutated)
                mutation_errors = radial_gpu_process_binding_errors(
                    mutated,
                    expected_pre_run_gpu=pre_run_gpu,
                    expected_pre_run_gpu_sha256="b" * 64,
                    expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
                    expected_python_executable=(
                        "/cache/liluchen/FastA2V/envs/ovi/bin/python"
                    ),
                )
                self.assertTrue(mutation_errors, label)

        unix_start_tamper = copy.deepcopy(all_idle_binding)
        unix_start_tamper["mps"]["pmon"][
            "backend_window_started_at_unix_seconds"
        ] = 102.75
        unix_start_errors = radial_gpu_process_binding_errors(
            unix_start_tamper,
            expected_pre_run_gpu=pre_run_gpu,
            expected_pre_run_gpu_sha256="b" * 64,
            expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
            expected_python_executable=(
                "/cache/liluchen/FastA2V/envs/ovi/bin/python"
            ),
        )
        self.assertTrue(
            any(
                "Unix backend-window start differs" in error
                for error in unix_start_errors
            ),
            unix_start_errors,
        )
        sparse_all_idle = copy.deepcopy(all_idle_binding)
        sparse_idle_raw = (
            "#Date Time gpu pid type sm mem enc dec command\n"
            "#YYYYMMDD HH:MM:SS Idx # C/G % % % % name\n"
            "19700101 00:01:42 0 - - - - - - -\n"
            "19700101 00:01:43 0 - - - - - - -\n"
            "19700101 00:02:04 0 - - - - - - -\n"
        )
        parsed_sparse_idle = parse_nvidia_smi_pmon_output(sparse_idle_raw)
        sparse_rows_by_line = {
            row["line_number"]: row for row in parsed_sparse_idle["rows"]
        }
        sparse_line_times = [101.12, 101.13, 102.1, 103.1, 124.1]
        sparse_line_records = [
            {
                "line_index": index,
                "raw_line": line,
                "received_at_unix_seconds": sparse_line_times[index],
                "received_at_monotonic_seconds": sparse_line_times[index],
                "parsed_row": sparse_rows_by_line.get(index + 1),
            }
            for index, line in enumerate(
                sparse_idle_raw.splitlines(keepends=True)
            )
        ]
        sparse_pmon = sparse_all_idle["mps"]["pmon"]
        sparse_pmon.update(
            {
                "raw_stdout": sparse_idle_raw,
                "raw_stdout_bytes": len(sparse_idle_raw.encode()),
                "raw_stdout_sha256": hashlib.sha256(
                    sparse_idle_raw.encode()
                ).hexdigest(),
                "header_columns": parsed_sparse_idle["header_columns"],
                "rows": parsed_sparse_idle["rows"],
                "parser_errors": [],
                "stdout_line_records": sparse_line_records,
                "final_sync_covered_line_number": 5,
            }
        )
        sparse_pmon_errors = radial_gpu_process_binding_errors(
            sparse_all_idle,
            expected_pre_run_gpu=pre_run_gpu,
            expected_pre_run_gpu_sha256="b" * 64,
            expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
            expected_python_executable=(
                "/cache/liluchen/FastA2V/envs/ovi/bin/python"
            ),
        )
        self.assertTrue(
            any("fixed cadence" in error for error in sparse_pmon_errors),
            sparse_pmon_errors,
        )

        duplicate_backend_receipt = copy.deepcopy(binding)
        duplicate_backend_receipt["interval_samples"][3] = copy.deepcopy(
            duplicate_backend_receipt["interval_samples"][2]
        )
        duplicate_backend_errors = radial_gpu_process_binding_errors(
            duplicate_backend_receipt,
            expected_pre_run_gpu=pre_run_gpu,
            expected_pre_run_gpu_sha256="b" * 64,
            expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
            expected_python_executable=(
                "/cache/liluchen/FastA2V/envs/ovi/bin/python"
            ),
        )
        self.assertTrue(
            any("duplicated" in error for error in duplicate_backend_errors),
            duplicate_backend_errors,
        )
        self.assertTrue(
            any(
                "two unique complete raw process-query receipts" in error
                for error in duplicate_backend_errors
            ),
            duplicate_backend_errors,
        )

        long_query_receipt = copy.deepcopy(all_idle_binding)
        long_sample = long_query_receipt["interval_samples"][2]
        for clock in ("unix", "monotonic"):
            finish_field = f"query_finished_at_{clock}_seconds"
            long_sample[finish_field] = 104.3
            long_sample["query_receipt"][finish_field] = 104.3
        del long_query_receipt["interval_samples"][3:5]
        long_query_errors = radial_gpu_process_binding_errors(
            long_query_receipt,
            expected_pre_run_gpu=pre_run_gpu,
            expected_pre_run_gpu_sha256="b" * 64,
            expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
            expected_python_executable=(
                "/cache/liluchen/FastA2V/envs/ovi/bin/python"
            ),
        )
        self.assertTrue(
            any("query duration" in error for error in long_query_errors),
            long_query_errors,
        )
        short_all_idle = copy.deepcopy(all_idle_binding)
        short_all_idle["mps"]["pmon"].update(
            {
                "compute_observation_deadline_reached_at_unix_seconds": 110.0,
                "compute_observation_deadline_reached_at_monotonic_seconds": (
                    110.0
                ),
            }
        )
        self.assertTrue(
            any(
                "ended before its timeout" in error
                for error in radial_gpu_process_binding_errors(
                    short_all_idle,
                    expected_pre_run_gpu=pre_run_gpu,
                    expected_pre_run_gpu_sha256="b" * 64,
                    expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
                    expected_python_executable=(
                        "/cache/liluchen/FastA2V/envs/ovi/bin/python"
                    ),
                )
            )
        )
        pre_backend_c = copy.deepcopy(binding)
        pre_backend_c.update(
            {
                "exact_backend_started_at_unix_seconds": 103.5,
                "exact_backend_started_at_monotonic_seconds": 103.5,
                "exact_backend_returned_at_unix_seconds": 104.2,
                "exact_backend_returned_at_monotonic_seconds": 104.2,
                "cuda_synchronize_started_at_unix_seconds": 104.3,
                "cuda_synchronize_started_at_monotonic_seconds": 104.3,
                "cuda_synchronized_at_unix_seconds": 104.4,
                "cuda_synchronized_at_monotonic_seconds": 104.4,
                "post_cuda_samples": [
                    gpu_sample(104.6, 1),
                    gpu_sample(104.8, 1),
                ],
            }
        )
        pre_backend_pmon = pre_backend_c["mps"]["pmon"]
        pre_backend_pmon.update(
            {
                "backend_window_started_at_unix_seconds": 103.5,
                "backend_window_started_at_monotonic_seconds": 103.5,
                "window_compute_ready_at_unix_seconds": 103.9,
                "window_compute_ready_at_monotonic_seconds": 103.9,
                "stop_requested_at_unix_seconds": 104.5,
                "stop_requested_at_monotonic_seconds": 104.5,
                "process_exited_at_unix_seconds": 104.6,
                "process_exited_at_monotonic_seconds": 104.6,
            }
        )
        pre_backend_pmon["stdout_line_records"][3].update(
            {
                "received_at_unix_seconds": 103.9,
                "received_at_monotonic_seconds": 103.9,
            }
        )
        pre_backend_errors = radial_gpu_process_binding_errors(
            pre_backend_c,
            expected_pre_run_gpu=pre_run_gpu,
            expected_pre_run_gpu_sha256="b" * 64,
            expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
            expected_python_executable=(
                "/cache/liluchen/FastA2V/envs/ovi/bin/python"
            ),
        )
        self.assertTrue(
            any(
                "source-DT sample is not strictly inside" in error
                for error in pre_backend_errors
            ),
            pre_backend_errors,
        )
        direct = copy.deepcopy(binding)
        direct["binding_method"] = "direct_nspid"
        direct["claim_scope"] = "snapshot_bound_not_continuous_exclusivity"
        direct["host_pid_ownership"] = "proven_by_nspid"
        direct["nspid"]["chain"] = [host_pid, container_pid]
        direct["host_pid_proc_visibility"] = {"status": "visible", "error": None}
        self.assertEqual(
            radial_gpu_process_binding_errors(
                direct,
                expected_pre_run_gpu=pre_run_gpu,
                expected_pre_run_gpu_sha256="b" * 64,
                expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
                expected_python_executable=(
                    "/cache/liluchen/FastA2V/envs/ovi/bin/python"
                ),
            ),
            [],
        )
        for field in tuple(binding):
            missing = copy.deepcopy(binding)
            missing.pop(field)
            self.assertTrue(
                radial_gpu_process_binding_errors(
                    missing,
                    expected_pre_run_gpu=pre_run_gpu,
                    expected_pre_run_gpu_sha256="b" * 64,
                    expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
                    expected_python_executable=(
                        "/cache/liluchen/FastA2V/envs/ovi/bin/python"
                    ),
                ),
                field,
            )
        for field in tuple(binding["mps"]["pmon"]):
            missing = copy.deepcopy(binding)
            missing["mps"]["pmon"].pop(field)
            self.assertTrue(
                radial_gpu_process_binding_errors(
                    missing,
                    expected_pre_run_gpu=pre_run_gpu,
                    expected_pre_run_gpu_sha256="b" * 64,
                    expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
                    expected_python_executable=(
                        "/cache/liluchen/FastA2V/envs/ovi/bin/python"
                    ),
                ),
                f"pmon.{field}",
            )
        for field in tuple(binding["context_live_sample"]):
            missing = copy.deepcopy(binding)
            missing["context_live_sample"].pop(field)
            self.assertTrue(
                radial_gpu_process_binding_errors(
                    missing,
                    expected_pre_run_gpu=pre_run_gpu,
                    expected_pre_run_gpu_sha256="b" * 64,
                    expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
                    expected_python_executable=(
                        "/cache/liluchen/FastA2V/envs/ovi/bin/python"
                    ),
                ),
                f"context_live_sample.{field}",
            )

        mutations = []
        wrong_hash = copy.deepcopy(binding)
        wrong_hash["pre_run_gpu_sha256"] = "c" * 64
        mutations.append(wrong_hash)
        stale = copy.deepcopy(binding)
        stale["microtest_started_at_unix_seconds"] = 800.0
        mutations.append(stale)
        short_gap = copy.deepcopy(binding)
        short_gap["post_cuda_samples"][1].update(
            {
                "sampled_at_unix_seconds": 104.05,
                "sampled_at_monotonic_seconds": 104.05,
                "query_started_at_unix_seconds": 104.05,
                "query_started_at_monotonic_seconds": 104.05,
                "query_finished_at_unix_seconds": 104.06,
                "query_finished_at_monotonic_seconds": 104.06,
            }
        )
        mutations.append(short_gap)
        pid_drift = copy.deepcopy(binding)
        pid_drift["interval_samples"][-1]["processes"][0]["host_pid"] += 1
        mutations.append(pid_drift)
        mps = copy.deepcopy(binding)
        mps["mps"]["cuda_mps_environment_variables"] = ["CUDA_MPS_PIPE_DIRECTORY"]
        mutations.append(mps)
        weak_memory = copy.deepcopy(binding)
        weak_memory["allocator_memory_bytes"] = RADIAL_QKV_STORAGE_BYTES - 1
        mutations.append(weak_memory)
        visible_proc = copy.deepcopy(binding)
        visible_proc["host_pid_proc_visibility"]["status"] = "visible"
        mutations.append(visible_proc)
        forged_pmon = copy.deepcopy(binding)
        forged_pmon["mps"]["pmon"]["rows"][0]["process_type"] = "M"
        mutations.append(forged_pmon)
        for mutated in mutations:
            self.assertTrue(
                radial_gpu_process_binding_errors(
                    mutated,
                    expected_pre_run_gpu=pre_run_gpu,
                    expected_pre_run_gpu_sha256="b" * 64,
                    expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
                    expected_python_executable=(
                        "/cache/liluchen/FastA2V/envs/ovi/bin/python"
                    ),
                )
            )

        evidence["finite"] = False
        evidence["output_abs_max"] = float("nan")
        evidence["pre_run_gpu"]["process_count"] = 1
        errors = radial_microtest_evidence_errors(
            evidence,
            expected_gpu_uuid="GPU-different",
            expected_pre_run_gpu=pre_run_gpu,
            expected_pre_run_gpu_sha256="b" * 64,
            expected_pre_run_gpu_path="/runs/test/pre_run_gpu.json",
            expected_python_executable=(
                "/cache/liluchen/FastA2V/envs/ovi/bin/python"
            ),
        )
        self.assertTrue(any("finite" in error for error in errors))
        self.assertTrue(any("device_uuid" in error for error in errors))
        self.assertTrue(any("pre-run" in error for error in errors))


class RadialBackendExecutionTests(unittest.TestCase):
    def make_backend(
        self,
        profile="aggressive",
        runtime_dependency_verifier=None,
    ):
        torch_module = FakeTorch()
        flashinfer = FakeFlashInfer()
        mask_calls = []
        rope_calls = []

        def mask_generator(*args, **kwargs):
            mask_calls.append((args, kwargs))
            return FakeMask(fixture_rows(profile))

        def rope_apply(tensor, grid, freqs):
            rope_calls.append((tensor.label, grid, freqs))
            return FakeTensor(
                f"rope({tensor.label})",
                tensor.shape,
                device=tensor.device,
                dtype=tensor.dtype,
            )

        backend = RadialVideoSelfAttentionBackend(
            torch_module=torch_module,
            flashinfer_module=flashinfer,
            mask_generator=mask_generator,
            get_indptr_from_mask=lambda mask, query: object(),
            get_indices_from_mask=lambda mask, query: object(),
            rope_apply_fn=rope_apply,
            profile=profile,
            runtime_dependency_verifier=runtime_dependency_verifier,
        )
        return backend, torch_module, flashinfer, mask_calls, rope_calls

    def test_first_cuda_dependency_audit_persists_across_metric_resets(self):
        evidence = {
            "status": "ok",
            "aliases": 26,
            "mapped_files": 31,
            "inventory_sha256": "a" * 64,
        }
        verifier_calls = []

        def verify_runtime():
            verifier_calls.append(True)
            return dict(evidence)

        backend, torch_module, _flashinfer, _mask_calls, _rope_calls = (
            self.make_backend(runtime_dependency_verifier=verify_runtime)
        )
        inputs = (
            RecordingAttention(),
            FakeTensor("hidden", (1, 15004, 3072)),
            FakeHostTensor([15004]),
            FakeHostTensor([[31, 22, 22]]),
            object(),
        )
        backend(*inputs)
        self.assertEqual(verifier_calls, [True])
        self.assertEqual(torch_module.cuda.synchronize_calls, ["cuda:0"])
        self.assertEqual(
            backend.metrics()["runtime_dependencies_after_first_cuda"],
            evidence,
        )

        backend.reset_metrics()
        backend(*inputs)
        self.assertEqual(verifier_calls, [True])
        self.assertEqual(torch_module.cuda.synchronize_calls, ["cuda:0"])
        self.assertEqual(
            backend.metrics()["runtime_dependencies_after_first_cuda"],
            evidence,
        )

    def test_exact_prefix_tail_protocol_reuses_ovi_components(self):
        backend, torch_module, flashinfer, mask_calls, rope_calls = self.make_backend()
        attention = RecordingAttention()
        hidden = FakeTensor("hidden", (1, 15004, 3072))
        result = backend(
            attention,
            hidden,
            FakeHostTensor([15004]),
            FakeHostTensor([[31, 22, 22]]),
            object(),
        )

        self.assertEqual(result.label, "projected")
        self.assertEqual(attention.projection_inputs[0].shape, (1, 15004, 3072))
        self.assertEqual([item[0] for item in rope_calls], ["q", "k"])
        self.assertEqual(len(mask_calls), 1)
        args, kwargs = mask_calls[0]
        self.assertEqual(args[1:], (15004, 15004, 31))
        self.assertEqual(
            kwargs,
            {
                "block_size": 128,
                "sparse_type": "radial",
                "decay_factor": 1.0,
                "model_type": "wan",
            },
        )
        self.assertEqual(len(flashinfer.wrappers), 1)
        wrapper = flashinfer.wrappers[0]
        self.assertEqual(wrapper.backend, "fa2")
        self.assertEqual(wrapper.plan_kwargs["M"], 14976)
        self.assertEqual(wrapper.plan_kwargs["N"], 14976)
        self.assertEqual(wrapper.plan_kwargs["R"], 128)
        self.assertEqual(wrapper.plan_kwargs["C"], 128)
        self.assertNotIn("o_data_type", wrapper.plan_kwargs)
        self.assertEqual(wrapper.run_calls[0][0].shape, (14976, 24, 128))
        self.assertEqual(wrapper.run_calls[0][1].shape, (14976, 24, 128))
        self.assertTrue(wrapper.run_calls[0][3])
        self.assertEqual(len(flashinfer.dense_calls), 2)
        prefix_tail = flashinfer.dense_calls[0]
        self.assertEqual(prefix_tail[0].shape, (14976, 24, 128))
        self.assertEqual(prefix_tail[1].shape, (28, 24, 128))
        self.assertTrue(prefix_tail[4])
        tail_all = flashinfer.dense_calls[1]
        self.assertEqual(tail_all[0].shape, (28, 24, 128))
        self.assertEqual(tail_all[1].shape, (15004, 24, 128))
        self.assertFalse(tail_all[4])
        self.assertEqual(len(flashinfer.merge_calls), 1)
        self.assertEqual(
            torch_module.empty_calls,
            [(128 * 1024 * 1024, "cuda:0", "torch.uint8")],
        )

    def test_plan_cache_is_keyed_retained_and_generation_metrics_reset(self):
        backend, _torch, flashinfer, mask_calls, _rope = self.make_backend()
        inputs = (
            RecordingAttention(),
            FakeTensor("hidden", (1, 15004, 3072)),
            FakeHostTensor([15004]),
            FakeHostTensor([[31, 22, 22]]),
            object(),
        )
        backend(*inputs)
        backend(*inputs)
        self.assertEqual(len(mask_calls), 1)
        self.assertEqual(len(flashinfer.wrappers), 1)
        self.assertEqual(backend.metrics()["plan_cache_misses"], 1)
        self.assertEqual(backend.metrics()["plan_cache_hits"], 1)
        exposed_metrics = backend.metrics()
        exposed_metrics["last_mask_audit"]["empty_rows"].append(116)
        self.assertEqual(
            backend.metrics()["last_mask_audit"]["empty_rows"],
            [22, 56, 90],
        )

        backend.reset_metrics()
        backend(*inputs)
        metrics = backend.metrics()
        self.assertEqual(len(mask_calls), 1)
        self.assertEqual(metrics["calls"], 1)
        self.assertEqual(metrics["plan_cache_entries"], 1)
        self.assertEqual(metrics["plan_cache_misses"], 0)
        self.assertEqual(metrics["plan_cache_hits"], 1)

    def test_unsupported_shape_grid_or_sequence_parallel_fails_fast(self):
        backend, _torch, flashinfer, mask_calls, _rope = self.make_backend()
        hidden = FakeTensor("hidden", (1, 15004, 3072))
        with self.assertRaisesRegex(RadialAttentionInputError, "requires q/k/v shape"):
            backend(
                RecordingAttention(shape=(1, 15003, 24, 128)),
                hidden,
                FakeHostTensor([15003]),
                FakeHostTensor([[31, 22, 22]]),
                object(),
            )
        with self.assertRaisesRegex(RadialAttentionInputError, "grid"):
            backend(
                RecordingAttention(),
                hidden,
                FakeHostTensor([15004]),
                FakeHostTensor([[31, 21, 22]]),
                object(),
            )
        with self.assertRaisesRegex(RadialAttentionInputError, "sp_size=1"):
            backend(
                RecordingAttention(use_sp=True),
                hidden,
                FakeHostTensor([15004]),
                FakeHostTensor([[31, 22, 22]]),
                object(),
            )
        self.assertEqual(mask_calls, [])
        self.assertEqual(flashinfer.wrappers, [])

    def test_same_shape_cannot_hide_later_sequence_or_grid_drift(self):
        backend, _torch, _flashinfer, mask_calls, _rope = self.make_backend()
        attention = RecordingAttention()
        hidden = FakeTensor("hidden", (1, 15004, 3072))
        backend(
            attention,
            hidden,
            FakeHostTensor([15004]),
            FakeHostTensor([[31, 22, 22]]),
            object(),
        )
        with self.assertRaisesRegex(RadialAttentionInputError, "seq_lens"):
            backend(
                attention,
                hidden,
                FakeHostTensor([15003]),
                FakeHostTensor([[31, 22, 22]]),
                object(),
            )
        with self.assertRaisesRegex(RadialAttentionInputError, "grid"):
            backend(
                attention,
                hidden,
                FakeHostTensor([15004]),
                FakeHostTensor([[1, 22, 682]]),
                object(),
            )
        self.assertEqual(len(mask_calls), 1)


if __name__ == "__main__":
    unittest.main()
