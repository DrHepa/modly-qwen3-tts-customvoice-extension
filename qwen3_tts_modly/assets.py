"""Pinned snapshot download, integrity, and readiness support."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Callable, Iterable
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

from .constants import (
    ASSETS,
    EXTENSION_ID,
    EXTENSION_VERSION,
    JSON_ASSET_PATHS,
    MODEL_REPO,
    MODEL_REVISION,
    NODE_ID,
    READY_MARKER_FILENAME,
    READY_SCHEMA,
    AssetSpec,
)
from .paths import PathContractError, canonical, safe_asset_file


LogFunction = Callable[[str], None]
OpenFunction = Callable[..., object]
MAX_MARKER_BYTES = 64 * 1024
CHUNK_SIZE = 1024 * 1024
WINDOWS_REPARSE_ATTRIBUTE = 0x400


class AssetError(RuntimeError):
    """A stable-code model snapshot provisioning failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.public_message = (
            "model snapshot provisioning failed; check network access and storage, then run Repair"
        )
        super().__init__(f"{code}: {message}")


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(path: Path, spec: AssetSpec) -> tuple[bool, str]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False, "file is missing"
    except OSError as exc:
        return False, f"file metadata could not be read ({exc})"
    attributes = getattr(info, "st_file_attributes", 0)
    if stat.S_ISLNK(info.st_mode) or attributes & WINDOWS_REPARSE_ATTRIBUTE:
        return False, "file is a symlink or reparse alias"
    if not stat.S_ISREG(info.st_mode):
        return False, "path is not a regular file"
    if info.st_size != spec.size:
        return False, f"size is {info.st_size}; expected {spec.size}"
    try:
        digest = sha256_file(path)
    except OSError as exc:
        return False, f"file could not be hashed ({exc})"
    if digest != spec.sha256:
        return False, "SHA-256 does not match the pinned snapshot"
    if spec.relative_path in JSON_ASSET_PATHS:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                return False, "JSON asset does not contain an object"
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return False, f"JSON asset is invalid ({exc})"
    return True, "valid"


def inventory_payload() -> list[dict[str, object]]:
    return [
        {"path": spec.relative_path, "size": spec.size, "sha256": spec.sha256}
        for spec in ASSETS
    ]


def ready_payload() -> dict[str, object]:
    return {
        "schema": READY_SCHEMA,
        "extensionId": EXTENSION_ID,
        "extensionVersion": EXTENSION_VERSION,
        "nodeId": NODE_ID,
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "inventory": inventory_payload(),
        },
    }


def _atomic_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if len(encoded) > MAX_MARKER_BYTES:
        raise AssetError("ASSET_MARKER_TOO_LARGE", "the readiness marker exceeds its size limit")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AssetError("ASSET_MARKER_WRITE_FAILED", "the readiness marker could not be written") from exc


def _read_ready_marker(model_dir: Path) -> tuple[bool, str]:
    marker = model_dir / READY_MARKER_FILENAME
    try:
        info = marker.lstat()
    except FileNotFoundError:
        return False, "readiness marker is missing"
    except OSError as exc:
        return False, f"readiness marker metadata failed ({exc})"
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return False, "readiness marker is not a regular file"
    if info.st_size > MAX_MARKER_BYTES:
        return False, "readiness marker is too large"
    try:
        parsed = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"readiness marker is invalid ({exc})"
    if parsed != ready_payload():
        return False, "readiness marker does not match this extension revision"
    return True, "valid"


def _inventory_entries(model_dir: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories_seen: set[str] = set()
    for root, directories, filenames in os.walk(model_dir, followlinks=False):
        root_path = Path(root)
        for name in list(directories):
            directory = root_path / name
            info = directory.lstat()
            attributes = getattr(info, "st_file_attributes", 0)
            if stat.S_ISLNK(info.st_mode) or attributes & WINDOWS_REPARSE_ATTRIBUTE:
                raise AssetError(
                    "ASSET_INVENTORY_ALIAS", "the model snapshot contains a directory alias"
                )
            directories_seen.add(directory.relative_to(model_dir).as_posix())
        for name in filenames:
            path = root_path / name
            info = path.lstat()
            attributes = getattr(info, "st_file_attributes", 0)
            if stat.S_ISLNK(info.st_mode) or attributes & WINDOWS_REPARSE_ATTRIBUTE:
                raise AssetError("ASSET_INVENTORY_ALIAS", "the model snapshot contains a file alias")
            files.add(path.relative_to(model_dir).as_posix())
    return files, directories_seen


def verify_snapshot(model_dir: Path, *, require_ready: bool = True) -> list[str]:
    """Return bounded verification failures for the exact pinned inventory."""

    failures: list[str] = []
    expected = {spec.relative_path for spec in ASSETS}
    expected_directories = {
        parent.as_posix()
        for spec in ASSETS
        for parent in Path(spec.relative_path).parents
        if parent != Path(".")
    }
    try:
        actual, actual_directories = _inventory_entries(model_dir)
    except (OSError, AssetError) as exc:
        return [str(exc)]
    allowed = expected | ({READY_MARKER_FILENAME} if require_ready else set())
    unexpected = sorted(actual - allowed)
    unexpected_directories = sorted(actual_directories - expected_directories)
    missing = sorted(expected - actual)
    if unexpected:
        failures.append("unexpected files: " + ", ".join(unexpected[:8]))
    if unexpected_directories:
        failures.append("unexpected directories: " + ", ".join(unexpected_directories[:8]))
    if missing:
        failures.append("missing files: " + ", ".join(missing[:8]))
    for spec in ASSETS:
        try:
            path = safe_asset_file(model_dir, spec.relative_path, create_parent=False)
        except PathContractError as exc:
            failures.append(f"{spec.relative_path}: {exc}")
            continue
        valid, reason = verify_asset(path, spec)
        if not valid:
            failures.append(f"{spec.relative_path}: {reason}")
    if require_ready:
        ready, reason = _read_ready_marker(model_dir)
        if not ready:
            failures.append(reason)
    return failures[:32]


def _is_alias(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def _validate_owned_root(model_dir: Path) -> Path:
    """Return the canonical owned root after rejecting aliases and non-directories."""

    try:
        info = model_dir.lstat()
    except OSError as exc:
        raise AssetError("ASSET_OWNED_ROOT_INVALID", "the owned model root is unavailable") from exc
    if _is_alias(info) or not stat.S_ISDIR(info.st_mode):
        raise AssetError(
            "ASSET_OWNED_ROOT_INVALID", "the owned model root must be a regular directory"
        )
    try:
        return model_dir.resolve(strict=True)
    except OSError as exc:
        raise AssetError("ASSET_OWNED_ROOT_INVALID", "the owned model root cannot be verified") from exc


def _validate_removal_parent(path: Path, model_dir: Path, canonical_root: Path) -> None:
    """Verify only the parent; resolving the entry itself could follow an alias."""

    try:
        path.relative_to(model_dir)
        parent = path.parent.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise AssetError(
            "ASSET_REPAIR_CONTAINMENT", "an extension-owned repair path is unsafe"
        ) from exc
    if parent != canonical_root and canonical_root not in parent.parents:
        raise AssetError(
            "ASSET_REPAIR_CONTAINMENT", "an extension-owned repair path escapes its root"
        )


def _remove_entry_no_follow(path: Path, model_dir: Path, canonical_root: Path) -> None:
    """Remove one owned entry recursively without ever following aliases."""

    _validate_removal_parent(path, model_dir, canonical_root)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise AssetError("ASSET_REPAIR_FAILED", "an obsolete owned entry cannot be inspected") from exc

    if _is_alias(info):
        try:
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                path.rmdir()
            else:
                path.unlink()
        except IsADirectoryError:
            # Windows directory junctions can report as links but require rmdir.
            try:
                path.rmdir()
            except OSError as exc:
                raise AssetError(
                    "ASSET_REPAIR_FAILED", "an obsolete owned alias cannot be removed"
                ) from exc
        except OSError as exc:
            raise AssetError("ASSET_REPAIR_FAILED", "an obsolete owned alias cannot be removed") from exc
        return

    if stat.S_ISDIR(info.st_mode):
        try:
            entries = list(os.scandir(path))
        except OSError as exc:
            raise AssetError(
                "ASSET_REPAIR_FAILED", "an obsolete owned directory cannot be inspected"
            ) from exc
        for entry in entries:
            _remove_entry_no_follow(Path(entry.path), model_dir, canonical_root)
        try:
            path.rmdir()
        except OSError as exc:
            raise AssetError(
                "ASSET_REPAIR_FAILED", "an obsolete owned directory cannot be removed"
            ) from exc
        return

    try:
        path.unlink()
    except OSError as exc:
        raise AssetError("ASSET_REPAIR_FAILED", "an obsolete owned file cannot be removed") from exc


def reconcile_snapshot_layout(model_dir: Path, *, log: LogFunction = print) -> None:
    """Remove only obsolete entries from the exact extension-owned snapshot root.

    Valid/corrupt expected files and useful regular ``.part`` files remain in place,
    so Repair never duplicates the multi-gigabyte snapshot in a staging sibling.
    """

    canonical_root = _validate_owned_root(model_dir)
    expected_files = {spec.relative_path for spec in ASSETS}
    expected_parts = {f"{spec.relative_path}.part" for spec in ASSETS}
    expected_directories = {
        parent.as_posix()
        for spec in ASSETS
        for parent in Path(spec.relative_path).parents
        if parent != Path(".")
    }
    removed_entries = 0

    def visit(directory: Path) -> None:
        nonlocal removed_entries
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise AssetError(
                "ASSET_REPAIR_FAILED", "the extension-owned model layout cannot be inspected"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                relative = path.relative_to(model_dir).as_posix()
                info = path.lstat()
            except (OSError, ValueError) as exc:
                raise AssetError(
                    "ASSET_REPAIR_CONTAINMENT", "an extension-owned repair entry is unsafe"
                ) from exc
            alias = _is_alias(info)
            if relative in expected_directories:
                if not alias and stat.S_ISDIR(info.st_mode):
                    visit(path)
                    continue
                _remove_entry_no_follow(path, model_dir, canonical_root)
                removed_entries += 1
                continue
            if relative in expected_files or relative in expected_parts:
                if not alias and stat.S_ISREG(info.st_mode):
                    continue
                _remove_entry_no_follow(path, model_dir, canonical_root)
                removed_entries += 1
                continue
            _remove_entry_no_follow(path, model_dir, canonical_root)
            removed_entries += 1

    visit(model_dir)
    if removed_entries:
        log(f"Removed {removed_entries} obsolete extension-owned snapshot entries")


def _response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    getter = getattr(response, "getcode", None)
    if callable(getter):
        result = getter()
        if isinstance(result, int):
            return result
    raise AssetError("ASSET_HTTP_STATUS_MISSING", "the download response has no HTTP status")


def _header(response: object, name: str) -> str:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    return str(getter(name, "")) if callable(getter) else ""


def _validate_response(
    response: object,
    spec: AssetSpec,
    existing_size: int,
) -> tuple[str, int]:
    status = _response_status(response)
    if existing_size:
        if status == 200:
            mode, expected_body = "wb", spec.size
        elif status == 206:
            content_range = _header(response, "Content-Range")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range.strip())
            if not match:
                raise AssetError("ASSET_RANGE_INVALID", "resume response has an invalid Content-Range")
            start, end, total = (int(value) for value in match.groups())
            if start != existing_size or total != spec.size or end < start or end >= total:
                raise AssetError("ASSET_RANGE_INVALID", "resume response does not match the requested range")
            mode, expected_body = "ab", spec.size - existing_size
        else:
            raise AssetError("ASSET_HTTP_STATUS", f"download returned HTTP status {status}")
    else:
        if status != 200:
            raise AssetError("ASSET_HTTP_STATUS", f"full download returned HTTP status {status}")
        mode, expected_body = "wb", spec.size

    content_length = _header(response, "Content-Length").strip()
    if content_length:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise AssetError("ASSET_LENGTH_INVALID", "download returned an invalid Content-Length") from exc
        if declared != expected_body:
            raise AssetError("ASSET_LENGTH_INVALID", "download body length does not match the pinned size")
    return mode, 0 if mode == "wb" else existing_size


def _stream_download(
    model_dir: Path,
    spec: AssetSpec,
    part_path: Path,
    *,
    opener: OpenFunction,
    log: LogFunction,
    timeout: float,
) -> None:
    existing_size = part_path.stat().st_size if part_path.is_file() else 0
    if existing_size >= spec.size:
        part_path.unlink(missing_ok=True)
        existing_size = 0
    headers = {"User-Agent": "Modly-Qwen3-TTS-CustomVoice/0.1.0"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
        log(f"Resuming {spec.relative_path} at {existing_size} bytes")
    request = Request(spec.resolve_url, headers=headers)
    with opener(request, timeout=timeout) as response:
        mode, downloaded = _validate_response(response, spec, existing_size)
        if existing_size and mode == "wb":
            log(f"Restarting {spec.relative_path}; server ignored Range")
        last_bucket = -1
        with part_path.open(mode) as handle:
            while True:
                block = response.read(CHUNK_SIZE)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                if downloaded > spec.size:
                    raise AssetError(
                        "ASSET_SIZE_EXCEEDED", f"{spec.relative_path} exceeded its pinned size"
                    )
                bucket = int(downloaded * 10 / spec.size)
                if bucket != last_bucket:
                    log(f"Downloading {spec.relative_path}: {min(100, int(downloaded * 100 / spec.size))}%")
                    last_bucket = bucket
            handle.flush()
            os.fsync(handle.fileno())
    if downloaded != spec.size:
        raise AssetError(
            "ASSET_SIZE_INCOMPLETE",
            f"{spec.relative_path} stopped at {downloaded} of {spec.size} bytes",
        )


def ensure_asset(
    model_dir: Path,
    spec: AssetSpec,
    *,
    opener: OpenFunction = urlopen,
    log: LogFunction = print,
    retries: int = 3,
    timeout: float = 90.0,
    retry_delay: float = 0.5,
) -> Path:
    if retries < 1:
        raise ValueError("retries must be at least one")
    destination = safe_asset_file(model_dir, spec.relative_path, create_parent=True)
    part = destination.with_name(f"{destination.name}.part")
    valid, _ = verify_asset(destination, spec)
    if valid:
        if part.exists() or part.is_symlink():
            try:
                info = part.lstat()
            except OSError as exc:
                raise AssetError("ASSET_PART_INVALID", "partial file metadata could not be read") from exc
            attributes = getattr(info, "st_file_attributes", 0)
            if (
                stat.S_ISLNK(info.st_mode)
                or attributes & WINDOWS_REPARSE_ATTRIBUTE
                or not stat.S_ISREG(info.st_mode)
            ):
                raise AssetError("ASSET_PART_INVALID", "partial asset is not a regular file")
            part.unlink()
        log(f"Verified {spec.relative_path}; skipped download")
        return destination

    if part.exists() or part.is_symlink():
        try:
            info = part.lstat()
        except OSError as exc:
            raise AssetError("ASSET_PART_INVALID", "partial file metadata could not be read") from exc
        attributes = getattr(info, "st_file_attributes", 0)
        if stat.S_ISLNK(info.st_mode) or attributes & WINDOWS_REPARSE_ATTRIBUTE or not stat.S_ISREG(info.st_mode):
            raise AssetError("ASSET_PART_INVALID", "partial asset is not a regular file")
        part_valid, _ = verify_asset(part, spec)
        if part_valid:
            os.replace(part, destination)
            log(f"Recovered completed {spec.relative_path} partial")
            return destination
        if info.st_size >= spec.size:
            part.unlink()

    last_error: BaseException | None = None
    range_restart_used = False
    for attempt in range(1, retries + 1):
        try:
            try:
                _stream_download(
                    model_dir,
                    spec,
                    part,
                    opener=opener,
                    log=log,
                    timeout=timeout,
                )
            except HTTPError as exc:
                partial_size = part.stat().st_size if part.is_file() else 0
                if exc.code != 416 or partial_size <= 0 or range_restart_used:
                    raise
                range_restart_used = True
                part.unlink(missing_ok=True)
                log(f"Restarting {spec.relative_path} after rejected Range")
                _stream_download(
                    model_dir,
                    spec,
                    part,
                    opener=opener,
                    log=log,
                    timeout=timeout,
                )
            valid, reason = verify_asset(part, spec)
            if not valid:
                if part.is_file() and part.stat().st_size >= spec.size:
                    part.unlink()
                raise AssetError("ASSET_VERIFY_FAILED", f"{spec.relative_path}: {reason}")
            destination = safe_asset_file(model_dir, spec.relative_path, create_parent=False)
            os.replace(part, destination)
            final_valid, final_reason = verify_asset(destination, spec)
            if not final_valid:
                raise AssetError(
                    "ASSET_PROMOTION_FAILED", f"{spec.relative_path}: {final_reason}"
                )
            log(f"Installed {spec.relative_path}")
            return destination
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            last_error = exc
            log(f"Attempt {attempt}/{retries} failed for {spec.relative_path}: {type(exc).__name__}")
            if attempt < retries:
                time.sleep(retry_delay)
    raise AssetError(
        "ASSET_DOWNLOAD_FAILED",
        f"could not download and verify {spec.relative_path}; check network and free space, then run Repair",
    ) from last_error


def ensure_snapshot(
    model_dir: Path,
    *,
    opener: OpenFunction = urlopen,
    log: LogFunction = print,
) -> Path:
    """Provision every pinned asset and write readiness only after exact verification."""

    _validate_owned_root(model_dir)
    current_failures = verify_snapshot(model_dir, require_ready=True)
    if not current_failures:
        log("Pinned model snapshot is already ready")
        return canonical(model_dir)

    marker = model_dir / READY_MARKER_FILENAME
    reconcile_snapshot_layout(model_dir, log=log)

    for spec in ASSETS:
        ensure_asset(model_dir, spec, opener=opener, log=log)

    # Remove interrupted marker temporaries and obsolete layouts that appeared
    # during a previous Repair before checking the exact inventory.
    reconcile_snapshot_layout(model_dir, log=log)
    failures = verify_snapshot(model_dir, require_ready=False)
    if failures:
        raise AssetError(
            "ASSET_INVENTORY_INVALID",
            "pinned model inventory is not exact: " + "; ".join(failures[:8]),
        )
    _atomic_json(marker, ready_payload())
    final_failures = verify_snapshot(model_dir, require_ready=True)
    if final_failures:
        marker.unlink(missing_ok=True)
        raise AssetError(
            "ASSET_READINESS_INVALID",
            "model readiness verification failed: " + "; ".join(final_failures[:8]),
        )
    log("Pinned model snapshot verified and marked ready")
    return canonical(model_dir)
