#!/usr/bin/env bash
set -euo pipefail

# One-shot CPU-only quality environment bootstrap.  The generated hashes are
# candidates, not trust roots: an independent reviewer must promote the full
# environment lock, seven direct-wheel hashes, and two weight hashes together
# in configs/quality_protocol.json before any score can be produced.
FASTA2V_CACHE_ROOT="${FASTA2V_CACHE_ROOT:-/cache/liluchen/FastA2V}"
EVAL_ENV="${FASTA2V_EVAL_ENV:-${FASTA2V_CACHE_ROOT}/envs/eval}"
EVAL_CHECKPOINT_ROOT="${FASTA2V_EVAL_CHECKPOINT_ROOT:-${FASTA2V_CACHE_ROOT}/checkpoints/eval}"
TORCH_HOME="${EVAL_CHECKPOINT_ROOT}/torch"
RECEIPT="${EVAL_CHECKPOINT_ROOT}/lpips_alex_v0.1_receipt.json"
LOCK_CANDIDATE="${EVAL_CHECKPOINT_ROOT}/quality_dependency_lock_candidate.json"
BOOTSTRAP_PIP_REPORT="${EVAL_CHECKPOINT_ROOT}/quality-bootstrap-pip-report.json"
CORE_PIP_REPORT="${EVAL_CHECKPOINT_ROOT}/quality-core-pip-report.json"
LPIPS_PIP_REPORT="${EVAL_CHECKPOINT_ROOT}/quality-lpips-pip-report.json"
WHEELHOUSE="${EVAL_CHECKPOINT_ROOT}/wheels"
PIP_CACHE_DIR="${FASTA2V_CACHE_ROOT}/cache/pip-eval"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.11}"
ALEXNET_URL="https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
ALEXNET_PATH="${TORCH_HOME}/hub/checkpoints/alexnet-owt-7be5be79.pth"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTOCOL_CONFIG="${SCRIPT_DIR}/../configs/quality_protocol.json"
QUALITY_URL_POLICY="${SCRIPT_DIR}/quality_archive_urls.py"
QUALITY_INSTALL_MODE="${QUALITY_INSTALL_MODE:-bootstrap}"
PINNED_PIP_REPORT="${EVAL_CHECKPOINT_ROOT}/quality-pinned-pip-report.json"
PINNED_REQUIREMENTS="${EVAL_CHECKPOINT_ROOT}/quality-pinned-requirements.txt"

case "${QUALITY_INSTALL_MODE}" in
  bootstrap|pinned) ;;
  *)
    echo "QUALITY_INSTALL_MODE must be bootstrap or pinned" >&2
    exit 2
    ;;
esac

if [[ "${EVAL_ENV}" != "/cache/liluchen/FastA2V/envs/eval" ]]; then
  echo "FASTA2V_EVAL_ENV must remain /cache/liluchen/FastA2V/envs/eval" >&2
  exit 2
fi
if [[ "${EVAL_CHECKPOINT_ROOT}" != "/cache/liluchen/FastA2V/checkpoints/eval" ]]; then
  echo "FASTA2V_EVAL_CHECKPOINT_ROOT must remain /cache/liluchen/FastA2V/checkpoints/eval" >&2
  exit 2
fi
command -v "${PYTHON_BOOTSTRAP}" >/dev/null
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
[[ -f "${PROTOCOL_CONFIG}" ]]
[[ -f "${QUALITY_URL_POLICY}" ]]
BOOTSTRAP_PYTHON_MINOR="$("${PYTHON_BOOTSTRAP}" -I -S -B -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${BOOTSTRAP_PYTHON_MINOR}" != "3.11" ]]; then
  echo "PYTHON_BOOTSTRAP must be Python 3.11, found ${BOOTSTRAP_PYTHON_MINOR}" >&2
  exit 2
fi
PROTOCOL_LOCK_STATUS="$(EVAL_PROTOCOL_CONFIG="${PROTOCOL_CONFIG}" "${PYTHON_BOOTSTRAP}" -I -S -B -c 'import json, os; print(json.load(open(os.environ["EVAL_PROTOCOL_CONFIG"], encoding="utf-8"))["lpips"]["trusted_lock_status"])')"
if [[ "${QUALITY_INSTALL_MODE}" == "bootstrap" && "${PROTOCOL_LOCK_STATUS}" != "bootstrap_unpinned" ]]; then
  echo "bootstrap mode requires an unpinned checked-in protocol" >&2
  exit 2
fi
if [[ "${QUALITY_INSTALL_MODE}" == "pinned" && "${PROTOCOL_LOCK_STATUS}" != "pinned" ]]; then
  echo "pinned mode requires a fully promoted checked-in protocol lock" >&2
  exit 2
fi

# Reusing an environment can retain executable .pth/sitecustomize/pyc files.
# Refuse it instead of deleting user state or layering over an unknown tree.
if [[ -e "${EVAL_ENV}" || -L "${EVAL_ENV}" ]]; then
  echo "fixed eval environment already exists; move it aside and review it before a fresh bootstrap: ${EVAL_ENV}" >&2
  exit 2
fi
for output in \
  "${RECEIPT}" \
  "${LOCK_CANDIDATE}" \
  "${BOOTSTRAP_PIP_REPORT}" \
  "${CORE_PIP_REPORT}" \
  "${LPIPS_PIP_REPORT}" \
  "${PINNED_PIP_REPORT}" \
  "${PINNED_REQUIREMENTS}" \
  "${WHEELHOUSE}" \
  "${ALEXNET_PATH}"
do
  if [[ -e "${output}" || -L "${output}" ]]; then
    echo "quality bootstrap output already exists; refusing to reuse it: ${output}" >&2
    exit 2
  fi
done

mkdir -p "${FASTA2V_CACHE_ROOT}/envs" "${EVAL_CHECKPOINT_ROOT}" "${PIP_CACHE_DIR}"
"${PYTHON_BOOTSTRAP}" -I -B -m venv "${EVAL_ENV}"
EVAL_PYTHON_MINOR="$("${EVAL_ENV}/bin/python" -I -S -B -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${EVAL_PYTHON_MINOR}" != "3.11" ]]; then
  echo "fixed eval environment must use Python 3.11, found ${EVAL_PYTHON_MINOR}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=""
export PIP_CACHE_DIR
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TORCH_HOME
unset PYTHONHOME
unset PYTHONPATH

if [[ "${QUALITY_INSTALL_MODE}" == "bootstrap" ]]; then
  "${EVAL_ENV}/bin/python" -I -B -m pip install \
    --force-reinstall \
    --no-compile \
    --only-binary=:all: \
    --report "${BOOTSTRAP_PIP_REPORT}" \
    --index-url "https://pypi.org/simple" \
    "pip==25.1.1" \
    "setuptools==75.8.0"
  "${EVAL_ENV}/bin/python" -I -B -m pip install \
    --force-reinstall \
    --no-compile \
    --only-binary=:all: \
    --report "${CORE_PIP_REPORT}" \
    --index-url "https://download.pytorch.org/whl/cpu" \
    --extra-index-url "https://pypi.org/simple" \
    "torch==2.6.0+cpu" \
    "torchvision==0.21.0+cpu" \
    "numpy==1.26.4" \
    "scipy==1.13.1" \
    "tqdm==4.67.1" \
    "pillow==11.1.0"
  "${EVAL_ENV}/bin/python" -I -B -m pip install \
    --force-reinstall \
    --no-compile \
    --only-binary=:all: \
    --report "${LPIPS_PIP_REPORT}" \
    --index-url "https://pypi.org/simple" \
    --no-deps \
    "lpips==0.1.4"
else
  EVAL_PROTOCOL_CONFIG="${PROTOCOL_CONFIG}" \
  EVAL_QUALITY_URL_POLICY="${QUALITY_URL_POLICY}" \
  EVAL_WHEELHOUSE="${WHEELHOUSE}" \
  EVAL_PINNED_REQUIREMENTS="${PINNED_REQUIREMENTS}" \
  "${PYTHON_BOOTSTRAP}" -I -S -B - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse
import urllib.request


def load_url_policy(path):
    path = Path(path).resolve()
    namespace = {
        "__name__": "_fasta2v_quality_archive_urls",
        "__file__": str(path),
    }
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    return namespace


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


protocol = json.loads(Path(os.environ["EVAL_PROTOCOL_CONFIG"]).read_text(encoding="utf-8"))["lpips"]
url_policy = load_url_policy(os.environ["EVAL_QUALITY_URL_POLICY"])
packages = protocol.get("trusted_environment_packages")
if protocol.get("trusted_lock_status") != "pinned" or not isinstance(packages, list):
    raise SystemExit("checked-in full dependency lock is not pinned")
wheelhouse = Path(os.environ["EVAL_WHEELHOUSE"])
wheelhouse.mkdir(parents=True, exist_ok=False)
requirements = []
for package in packages:
    url = package["archive_url"]
    expected_hash = package["archive_sha256"]
    try:
        url_policy["validate_dependency_archive_url"](
            url,
            expected_source_index=package["source_index"],
        )
    except ValueError as exc:
        raise SystemExit(
            f"pinned dependency URL violates the fixed source policy: "
            f"{package['distribution']}: {exc}"
        ) from exc
    filename = Path(unquote(urlparse(url).path)).name
    if not filename.endswith(".whl"):
        raise SystemExit(f"pinned dependency is not a wheel: {filename}")
    wheel = wheelhouse / filename
    temporary = wheel.with_suffix(wheel.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    if sha256(temporary) != expected_hash:
        raise SystemExit(f"pinned archive hash mismatch: {package['distribution']}")
    temporary.replace(wheel)
    requirements.append(
        f"{package['distribution']}=={package['version']} --hash=sha256:{expected_hash}"
    )
Path(os.environ["EVAL_PINNED_REQUIREMENTS"]).write_text(
    "\n".join(requirements) + "\n",
    encoding="utf-8",
)
PY
  "${EVAL_ENV}/bin/python" -I -B -m pip install \
    --force-reinstall \
    --no-compile \
    --no-index \
    --find-links "${WHEELHOUSE}" \
    --no-deps \
    --require-hashes \
    --report "${PINNED_PIP_REPORT}" \
    --requirement "${PINNED_REQUIREMENTS}"
fi

mkdir -p "$(dirname "${ALEXNET_PATH}")"
EVAL_ALEXNET_URL="${ALEXNET_URL}" \
EVAL_ALEXNET_PATH="${ALEXNET_PATH}" \
EVAL_INSTALL_MODE="${QUALITY_INSTALL_MODE}" \
EVAL_PROTOCOL_CONFIG="${PROTOCOL_CONFIG}" \
"${EVAL_ENV}/bin/python" -I -S -B - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import urllib.request

url = os.environ["EVAL_ALEXNET_URL"]
path = Path(os.environ["EVAL_ALEXNET_PATH"])
temporary = path.with_suffix(path.suffix + ".part")
urllib.request.urlretrieve(url, temporary)
digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
if not digest.startswith("7be5be79"):
    raise SystemExit(
        f"AlexNet weight SHA256 {digest} does not match official prefix 7be5be79"
    )
if os.environ["EVAL_INSTALL_MODE"] == "pinned":
    protocol = json.loads(
        Path(os.environ["EVAL_PROTOCOL_CONFIG"]).read_text(encoding="utf-8")
    )["lpips"]
    expected = next(
        item["trusted_sha256"]
        for item in protocol["weights"]
        if item["weight_id"] == "torchvision_alexnet_owt"
    )
    if digest != expected:
        raise SystemExit(f"AlexNet weight SHA256 {digest} differs from pinned trust root")
temporary.replace(path)
PY

EVAL_ENV="${EVAL_ENV}" \
EVAL_RECEIPT="${RECEIPT}" \
EVAL_LOCK_CANDIDATE="${LOCK_CANDIDATE}" \
EVAL_BOOTSTRAP_PIP_REPORT="${BOOTSTRAP_PIP_REPORT}" \
EVAL_CORE_PIP_REPORT="${CORE_PIP_REPORT}" \
EVAL_LPIPS_PIP_REPORT="${LPIPS_PIP_REPORT}" \
EVAL_PINNED_PIP_REPORT="${PINNED_PIP_REPORT}" \
EVAL_WHEELHOUSE="${WHEELHOUSE}" \
EVAL_ALEXNET_PATH="${ALEXNET_PATH}" \
EVAL_INSTALL_MODE="${QUALITY_INSTALL_MODE}" \
EVAL_PROTOCOL_CONFIG="${PROTOCOL_CONFIG}" \
EVAL_QUALITY_URL_POLICY="${QUALITY_URL_POLICY}" \
"${EVAL_ENV}/bin/python" -I -B - <<'PY'
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from urllib.parse import unquote, urlparse
import urllib.request

from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive_json(path, payload):
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())


def load_url_policy(path):
    path = Path(path).resolve()
    namespace = {
        "__name__": "_fasta2v_quality_archive_urls",
        "__file__": str(path),
    }
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    return namespace


environment_root = str(Path(os.environ["EVAL_ENV"]).resolve())
site_packages = (
    Path(environment_root) / "lib" / "python3.11" / "site-packages"
).resolve()
receipt_path = Path(os.environ["EVAL_RECEIPT"])
lock_candidate_path = Path(os.environ["EVAL_LOCK_CANDIDATE"])
alexnet_path = Path(os.environ["EVAL_ALEXNET_PATH"]).resolve()
wheelhouse = Path(os.environ["EVAL_WHEELHOUSE"]).resolve()
install_mode = os.environ["EVAL_INSTALL_MODE"]
protocol_config = json.loads(
    Path(os.environ["EVAL_PROTOCOL_CONFIG"]).read_text(encoding="utf-8")
)["lpips"]
url_policy = load_url_policy(os.environ["EVAL_QUALITY_URL_POLICY"])
if install_mode == "bootstrap":
    wheelhouse.mkdir(parents=True, exist_ok=False)
elif not wheelhouse.is_dir():
    raise SystemExit(f"pinned wheelhouse is missing: {wheelhouse}")
expected_direct_packages = {
    "torch": ("2.6.0+cpu", "torch"),
    "torchvision": ("0.21.0+cpu", "torchvision"),
    "lpips": ("0.1.4", "lpips"),
    "numpy": ("1.26.4", "numpy"),
    "scipy": ("1.13.1", "scipy"),
    "tqdm": ("4.67.1", "tqdm"),
    "pillow": ("11.1.0", "PIL"),
}
report_paths = (
    [Path(os.environ["EVAL_PINNED_PIP_REPORT"])]
    if install_mode == "pinned"
    else [
        Path(os.environ["EVAL_BOOTSTRAP_PIP_REPORT"]),
        Path(os.environ["EVAL_CORE_PIP_REPORT"]),
        Path(os.environ["EVAL_LPIPS_PIP_REPORT"]),
    ]
)
archive_receipts = {}
if install_mode == "pinned":
    for locked in protocol_config["trusted_environment_packages"]:
        distribution = locked["distribution"]
        try:
            url_policy["validate_dependency_archive_url"](
                locked["archive_url"],
                expected_source_index=locked["source_index"],
            )
        except ValueError as exc:
            raise SystemExit(
                f"pinned dependency URL violates the fixed source policy: "
                f"{distribution}: {exc}"
            ) from exc
        filename = Path(unquote(urlparse(locked["archive_url"]).path)).name
        wheel_path = wheelhouse / filename
        if not wheel_path.is_file() or sha256(wheel_path) != locked["archive_sha256"]:
            raise SystemExit(f"retained pinned wheel drifted: {distribution}")
        archive_receipts[distribution] = {
            **locked,
            "archive_path": str(wheel_path.resolve()),
        }
else:
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for install in report.get("install", []):
            metadata = install.get("metadata", {})
            distribution = str(canonicalize_name(str(metadata.get("name", ""))))
            version = str(metadata.get("version", ""))
            download = install.get("download_info", {})
            archive_url = download.get("url")
            archive_info = download.get("archive_info", {})
            archive_hash = archive_info.get("hashes", {}).get("sha256")
            if archive_hash is None:
                legacy_hash = archive_info.get("hash", "")
                if legacy_hash.startswith("sha256="):
                    archive_hash = legacy_hash.split("=", 1)[1]
            if not distribution or not version or not isinstance(archive_url, str):
                raise SystemExit(f"incomplete pip report dependency in {report_path}")
            if re.fullmatch(r"[0-9a-f]{64}", str(archive_hash)) is None:
                raise SystemExit(f"pip report omitted full SHA256 for {distribution}")
            try:
                source_index = url_policy[
                    "classify_dependency_archive_url"
                ](archive_url)
            except ValueError as exc:
                raise SystemExit(
                    f"unapproved archive source for {distribution}: "
                    f"{archive_url}: {exc}"
                ) from exc
            if distribution in archive_receipts:
                raise SystemExit(f"dependency appeared in multiple pip reports: {distribution}")
            filename = Path(unquote(urlparse(archive_url).path)).name
            if not filename.endswith(".whl"):
                raise SystemExit(f"dependency is not a wheel: {distribution}: {filename}")
            wheel_name, wheel_version, _build, _tags = parse_wheel_filename(filename)
            if str(canonicalize_name(str(wheel_name))) != distribution:
                raise SystemExit(f"wheel filename name differs for {distribution}: {filename}")
            if str(wheel_version) != version:
                raise SystemExit(f"wheel filename version differs for {distribution}: {filename}")
            wheel_path = wheelhouse / filename
            temporary = wheel_path.with_suffix(wheel_path.suffix + ".part")
            urllib.request.urlretrieve(archive_url, temporary)
            if sha256(temporary) != archive_hash:
                raise SystemExit(f"downloaded wheel hash differs from pip report: {distribution}")
            temporary.replace(wheel_path)
            archive_receipts[distribution] = {
                "distribution": distribution,
                "version": version,
                "source_index": source_index,
                "archive_url": archive_url,
                "archive_sha256": archive_hash,
                "archive_path": str(wheel_path.resolve()),
            }

installed = {}
for distribution_metadata in importlib.metadata.distributions(path=[str(site_packages)]):
    distribution = str(canonicalize_name(distribution_metadata.metadata["Name"]))
    if distribution in installed:
        raise SystemExit(f"duplicate installed distribution: {distribution}")
    installed[distribution] = distribution_metadata
if set(installed) != set(archive_receipts):
    missing = sorted(set(archive_receipts) - set(installed))
    extra = sorted(set(installed) - set(archive_receipts))
    raise SystemExit(f"installed distribution set differs from pip reports; missing={missing}, extra={extra}")

packages = []
modules = {}
for distribution in sorted(archive_receipts):
    distribution_metadata = installed[distribution]
    archive = archive_receipts[distribution]
    if distribution_metadata.version != archive["version"]:
        raise SystemExit(f"installed version differs from report for {distribution}")
    record_path = Path(distribution_metadata._path) / "RECORD"
    if not record_path.is_file() or record_path.parent.parent.resolve() != site_packages:
        raise SystemExit(f"fixed wheel RECORD is missing for {distribution}: {record_path}")
    package = {
        **archive,
        "record_path": str(record_path.resolve()),
        "record_sha256": sha256(record_path),
    }
    direct = expected_direct_packages.get(distribution)
    if direct is not None:
        expected_version, module_name = direct
        if distribution_metadata.version != expected_version:
            raise SystemExit(
                f"{distribution} version {distribution_metadata.version!r} != fixed {expected_version!r}"
            )
        module = importlib.import_module(module_name)
        modules[module_name] = module
        module_path = str(Path(module.__file__).resolve())
        if not module_path.startswith(str(site_packages) + "/"):
            raise SystemExit(f"{distribution} escaped fixed eval env: {module_path}")
        package.update(
            {
                "module": module_name,
                "module_path": module_path,
                "module_sha256": sha256(Path(module_path)),
            }
        )
    packages.append(package)

pyc_files = [path for path in site_packages.rglob("*.pyc") if path.is_file()]
symlinks = [path for path in site_packages.rglob("*") if path.is_symlink()]
if pyc_files:
    raise SystemExit(f"--no-compile invariant failed; found bytecode: {pyc_files[:3]}")
if symlinks:
    raise SystemExit(f"site-packages symlinks are forbidden: {symlinks[:3]}")

torch = modules["torch"]
if torch.cuda.is_available():
    raise SystemExit("fixed quality environment must remain CPU-only")
lpips_linear_path = (
    Path(modules["lpips"].__file__).resolve().parent
    / "weights"
    / "v0.1"
    / "alex.pth"
)
for weight in (lpips_linear_path, alexnet_path):
    if not weight.is_file() or weight.stat().st_size <= 0:
        raise SystemExit(f"required LPIPS weight is missing or empty: {weight}")

lock_payload = [
    {
        key: package[key]
        for key in (
            "distribution",
            "version",
            "source_index",
            "archive_url",
            "archive_sha256",
        )
    }
    for package in packages
]
environment_lock_sha256 = hashlib.sha256(
    json.dumps(
        lock_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
receipt = {
    "schema_version": 2,
    "created_by": "scripts/install_ovi_quality_env.sh",
    "environment_root": environment_root,
    "python_executable": str(Path(sys.executable).absolute()),
    "sys_prefix": environment_root,
    "python_version": platform.python_version(),
    "runtime_contract": {
        "python_arguments": ["-I", "-S", "-B"],
        "python_minor": "3.11",
        "site_packages": str(site_packages),
    },
    "environment_lock_sha256": environment_lock_sha256,
    "installer_reports": [
        {"path": str(path.resolve()), "sha256": sha256(path)}
        for path in report_paths
    ],
    "packages": packages,
    "weights": [
        {
            "weight_id": "lpips_alex_v0.1_linear",
            "path": str(lpips_linear_path),
            "bytes": lpips_linear_path.stat().st_size,
            "sha256": sha256(lpips_linear_path),
            "source_type": "installed_package",
            "source": "https://pypi.org/project/lpips/0.1.4/",
            "source_distribution": "lpips",
            "source_version": "0.1.4",
        },
        {
            "weight_id": "torchvision_alexnet_owt",
            "path": str(alexnet_path),
            "bytes": alexnet_path.stat().st_size,
            "sha256": sha256(alexnet_path),
            "source_type": "url",
            "source": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        },
    ],
}
if install_mode == "pinned":
    if lock_payload != protocol_config["trusted_environment_packages"]:
        raise SystemExit("reproduced dependency payload differs from checked-in full lock")
    if environment_lock_sha256 != protocol_config["trusted_environment_lock_sha256"]:
        raise SystemExit("reproduced dependency lock SHA256 differs from checked-in trust root")
    direct_hashes = {
        item["distribution"]: item["trusted_archive_sha256"]
        for item in protocol_config["packages"]
    }
    for package in packages:
        if (
            package["distribution"] in direct_hashes
            and package["archive_sha256"]
            != direct_hashes[package["distribution"]]
        ):
            raise SystemExit(f"direct archive trust mismatch: {package['distribution']}")
    trusted_weights = {
        item["weight_id"]: item["trusted_sha256"]
        for item in protocol_config["weights"]
    }
    for weight in receipt["weights"]:
        if weight["sha256"] != trusted_weights[weight["weight_id"]]:
            raise SystemExit(f"weight trust mismatch: {weight['weight_id']}")
write_exclusive_json(receipt_path, receipt)
lock_candidate = {
    "schema_version": 2,
    "status": (
        "pinned_environment_reproduced"
        if install_mode == "pinned"
        else "candidate_only_requires_independent_review_and_tracked_promotion"
    ),
    "receipt_path": str(receipt_path),
    "receipt_sha256": sha256(receipt_path),
    "trusted_environment_lock_sha256": environment_lock_sha256,
    "trusted_environment_packages": lock_payload,
    "package_trusted_archive_sha256": {
        item["distribution"]: item["archive_sha256"] for item in packages
    },
    "package_archive_url": {
        item["distribution"]: item["archive_url"] for item in packages
    },
    "weight_trusted_sha256": {
        item["weight_id"]: item["sha256"] for item in receipt["weights"]
    },
    "weight_source": {
        item["weight_id"]: item["source"] for item in receipt["weights"]
    },
}
write_exclusive_json(lock_candidate_path, lock_candidate)
print(receipt_path)
print(lock_candidate_path)
PY

echo "Quality environment receipt: ${RECEIPT}"
echo "Candidate hashes require independent review and a tracked protocol update: ${LOCK_CANDIDATE}"
