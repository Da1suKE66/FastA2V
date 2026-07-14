#!/usr/bin/env python3
"""Canonical dependency-archive URL policy for the fixed quality environment."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


PYTORCH_SOURCE_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_SOURCE_INDEX = "https://pypi.org/simple"
PYTORCH_ARCHIVE_HOSTS = frozenset(
    {
        "download.pytorch.org",
        "download-r2.pytorch.org",
    }
)
PYPI_ARCHIVE_HOSTS = frozenset({"files.pythonhosted.org"})
_MALFORMED_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_ENCODED_PATH_SEPARATOR = re.compile(r"%(?:2f|5c)", re.IGNORECASE)


def _canonical_https_url_parts(url: object):
    if not isinstance(url, str) or not url:
        raise ValueError("archive URL must be a non-empty string")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in url):
        raise ValueError("archive URL contains a control character")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
        username = parsed.username
        password = parsed.password
    except ValueError as exc:
        raise ValueError(f"archive URL authority is invalid: {exc}") from exc
    if parsed.scheme != "https":
        raise ValueError("archive URL scheme must be exactly https")
    if username is not None or password is not None:
        raise ValueError("archive URL must not contain credentials")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("archive URL hostname is missing")
    if port not in (None, 443):
        raise ValueError("archive URL port must be the HTTPS default or 443")
    if parsed.netloc.lower() not in {hostname, f"{hostname}:443"}:
        raise ValueError("archive URL authority is not canonical")
    if parsed.fragment or "#" in url:
        raise ValueError("archive URL must not contain a fragment")
    if parsed.query or "?" in url:
        raise ValueError("archive URL must not contain a query")
    if not parsed.path.startswith("/"):
        raise ValueError("archive URL path must be absolute")
    if "\\" in parsed.path or _MALFORMED_PERCENT_ESCAPE.search(parsed.path):
        raise ValueError("archive URL path is not canonical")
    if _ENCODED_PATH_SEPARATOR.search(parsed.path):
        raise ValueError("archive URL path must not encode a path separator")
    try:
        decoded_path = unquote(parsed.path, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("archive URL path is not valid UTF-8") from exc
    if any(
        ord(character) < 0x20 or ord(character) == 0x7F
        for character in decoded_path
    ):
        raise ValueError("archive URL path decodes to a control character")
    if "\\" in decoded_path or "?" in decoded_path or "#" in decoded_path:
        raise ValueError("archive URL path decodes to a reserved delimiter")
    segments = decoded_path.split("/")
    if any(segment in {"", ".", ".."} for segment in segments[1:]):
        raise ValueError("archive URL path contains an empty or dot segment")
    filename = segments[-1]
    if not filename.endswith(".whl"):
        raise ValueError("dependency archive URL must name a wheel")
    return parsed, hostname, decoded_path


def classify_dependency_archive_url(url: object) -> str:
    """Return the canonical resolver index for one accepted archive URL."""

    _parsed, hostname, decoded_path = _canonical_https_url_parts(url)
    if hostname in PYTORCH_ARCHIVE_HOSTS:
        if not decoded_path.startswith("/whl/cpu/"):
            raise ValueError("PyTorch archive URL must remain below /whl/cpu/")
        return PYTORCH_SOURCE_INDEX
    if hostname in PYPI_ARCHIVE_HOSTS:
        if not decoded_path.startswith("/packages/"):
            raise ValueError("PyPI archive URL must remain below /packages/")
        return PYPI_SOURCE_INDEX
    raise ValueError(f"archive URL hostname is not approved: {hostname!r}")


def validate_dependency_archive_url(
    url: object,
    *,
    expected_source_index: object,
) -> str:
    """Validate an archive URL and bind it to its canonical resolver index."""

    if expected_source_index not in {PYTORCH_SOURCE_INDEX, PYPI_SOURCE_INDEX}:
        raise ValueError(
            f"dependency source index is not approved: {expected_source_index!r}"
        )
    observed_source_index = classify_dependency_archive_url(url)
    if observed_source_index != expected_source_index:
        raise ValueError(
            "dependency archive URL does not match its canonical source index: "
            f"{observed_source_index!r} != {expected_source_index!r}"
        )
    return observed_source_index


def validate_exact_wheelhouse(
    wheelhouse: object,
    expected_paths: object,
) -> frozenset[Path]:
    """Require an exact flat set of regular, non-symlink wheel files."""

    wheelhouse_path = Path(wheelhouse)
    if wheelhouse_path.is_symlink() or not wheelhouse_path.is_dir():
        raise ValueError("wheelhouse must be a non-symlink directory")
    wheelhouse_path = wheelhouse_path.resolve()
    try:
        expected_sequence = tuple(expected_paths)
    except TypeError as exc:
        raise ValueError("expected wheel paths must be iterable") from exc
    expected = set()
    for raw_path in expected_sequence:
        path = Path(raw_path).resolve()
        if path.parent != wheelhouse_path:
            raise ValueError(f"expected wheel escaped the flat wheelhouse: {path}")
        expected.add(path)
    if len(expected) != len(expected_sequence):
        raise ValueError("expected wheel paths contain a duplicate filename")

    observed = set()
    rejected = []
    for path in wheelhouse_path.iterdir():
        if path.is_symlink() or not path.is_file():
            rejected.append(path.name)
            continue
        observed.add(path.resolve())
    if rejected:
        raise ValueError(
            "wheelhouse contains non-regular or symlink top-level entries: "
            f"{sorted(rejected)}"
        )
    if observed != expected:
        raise ValueError(
            "wheelhouse file set differs; "
            f"missing={sorted(str(path) for path in expected - observed)}, "
            f"extra={sorted(str(path) for path in observed - expected)}"
        )
    return frozenset(observed)
