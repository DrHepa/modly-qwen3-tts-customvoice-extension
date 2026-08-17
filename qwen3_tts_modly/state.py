"""Pathless setup state and fail-closed runtime snapshot validation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata as importlib_metadata
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import stat
import sys
from typing import Mapping
import uuid

from .assets import inventory_payload, verify_snapshot
from .constants import (
    ENTRY_FILENAME,
    EXTENSION_ID,
    EXTENSION_VERSION,
    MODEL_REPO,
    MODEL_REVISION,
    NODE_ID,
    STATE_FILENAME,
    STATE_SCHEMA,
)
from .paths import PathContractError, canonical, normalize_architecture, normalize_platform_name, owned_model_directory
from .setup_support import PlatformFlavor, SUPPORTED_PYTHON_MINORS, expected_distributions


MAX_STATE_BYTES = 64 * 1024
WINDOWS_REPARSE_ATTRIBUTE = 0x400


class StateError(RuntimeError):
    """A stable-code stale, missing, or unsafe setup-state failure."""

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


@dataclass(frozen=True)
class RuntimeState:
    model_dir: Path
    flavor: PlatformFlavor
    payload: Mapping[str, object]


def state_path(extension_dir: Path) -> Path:
    return extension_dir / "venv" / STATE_FILENAME


def _link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def _unsafe_file(path: Path) -> bool:
    info = path.lstat()
    return (
        _link_or_reparse(info)
        or not stat.S_ISREG(info.st_mode)
    )


def _validated_state_venv(extension_dir: Path, *, missing_ok: bool) -> Path | None:
    venv = extension_dir / "venv"
    try:
        info = venv.lstat()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise StateError("STATE_VENV_MISSING", "the extension venv is unavailable")
    except OSError as exc:
        raise StateError("STATE_VENV_UNSAFE", "the extension venv could not be inspected") from exc
    if _link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise StateError("STATE_VENV_UNSAFE", "the extension venv must be a regular directory")
    try:
        return venv.resolve(strict=True)
    except OSError as exc:
        raise StateError("STATE_VENV_UNSAFE", "the extension venv could not be verified") from exc


def invalidate_setup_state(extension_dir: Path) -> None:
    """Remove only generated readiness state before a Repair attempt."""

    venv_root = _validated_state_venv(extension_dir, missing_ok=True)
    if venv_root is None:
        return
    path = state_path(extension_dir)
    try:
        if path.parent.resolve(strict=True) != venv_root:
            raise StateError("STATE_PATH_UNSAFE", "existing setup state parent is unsafe")
        info = path.lstat()
    except FileNotFoundError:
        return
    except StateError:
        raise
    except OSError as exc:
        raise StateError("STATE_PATH_UNSAFE", "existing setup state path is unsafe") from exc
    try:
        if _link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
            raise StateError("STATE_PATH_UNSAFE", "existing setup state path is unsafe")
        path.unlink()
    except StateError:
        raise
    except OSError as exc:
        raise StateError(
            "STATE_INVALIDATE_FAILED", "existing setup state could not be invalidated"
        ) from exc


def build_state_payload(
    flavor: PlatformFlavor,
    health_record: Mapping[str, object],
) -> dict[str, object]:
    python_minor = str(health_record.get("pythonMinor") or "")
    cuda_bf16 = health_record.get("cudaBf16")
    if python_minor not in {"3.11", "3.12"} or not isinstance(cuda_bf16, bool):
        raise StateError("STATE_HEALTH_INVALID", "health-check state is incomplete")
    return {
        "schema": STATE_SCHEMA,
        "extension": {
            "id": EXTENSION_ID,
            "version": EXTENSION_VERSION,
            "nodeId": NODE_ID,
            "entry": ENTRY_FILENAME,
        },
        "platform": {
            "system": flavor.system,
            "arch": flavor.arch,
            "accelerator": flavor.accelerator,
            "torchVariant": flavor.torch_variant,
            "expectedCuda": flavor.expected_cuda,
            "pythonMinor": python_minor,
            "setupCudaBf16": cuda_bf16,
            "attention": "sdpa",
        },
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "inventory": inventory_payload(),
        },
    }


def _contains_absolute_or_secret(value: object, *, key: str = "") -> bool:
    lowered = key.casefold()
    if any(fragment in lowered for fragment in ("token", "secret", "password", "authorization")):
        return True
    if isinstance(value, dict):
        return any(_contains_absolute_or_secret(item, key=str(name)) for name, item in value.items())
    if isinstance(value, list):
        return any(_contains_absolute_or_secret(item) for item in value)
    if isinstance(value, str):
        return value.startswith("/") or PureWindowsPath(value).is_absolute()
    return False


def write_setup_state(
    extension_dir: Path,
    flavor: PlatformFlavor,
    health_record: Mapping[str, object],
) -> Path:
    payload = build_state_payload(flavor, health_record)
    if _contains_absolute_or_secret(payload):
        raise StateError("STATE_PRIVACY_INVALID", "setup state contains a forbidden field")
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > MAX_STATE_BYTES:
        raise StateError("STATE_TOO_LARGE", "the setup state exceeds its size limit")
    venv = extension_dir / "venv"
    try:
        info = venv.lstat()
    except OSError as exc:
        raise StateError("STATE_VENV_MISSING", "the extension venv is unavailable") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or bool(getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise StateError("STATE_VENV_UNSAFE", "the extension venv is not a regular directory")

    destination = state_path(extension_dir)
    if destination.exists() or destination.is_symlink():
        try:
            if _unsafe_file(destination):
                raise StateError("STATE_PATH_UNSAFE", "setup state path is unsafe")
        except OSError as exc:
            raise StateError("STATE_PATH_UNSAFE", "setup state path is unsafe") from exc
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise StateError("STATE_WRITE_FAILED", "verified setup state could not be written") from exc
    return destination


def _read_state(extension_dir: Path) -> dict[str, object]:
    venv = extension_dir / "venv"
    try:
        venv_info = venv.lstat()
    except OSError as exc:
        raise StateError("STATE_VENV_MISSING", "extension venv is unavailable; run Repair") from exc
    if (
        stat.S_ISLNK(venv_info.st_mode)
        or bool(getattr(venv_info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE)
        or not stat.S_ISDIR(venv_info.st_mode)
    ):
        raise StateError("STATE_VENV_UNSAFE", "extension venv is unsafe; run Repair")
    path = state_path(extension_dir)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise StateError("STATE_MISSING", "setup state is missing; run Repair") from exc
    except OSError as exc:
        raise StateError("STATE_READ_FAILED", "setup state could not be read; run Repair") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or bool(getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE)
        or not stat.S_ISREG(info.st_mode)
    ):
        raise StateError("STATE_PATH_UNSAFE", "setup state is unsafe; run Repair")
    if info.st_size > MAX_STATE_BYTES:
        raise StateError("STATE_TOO_LARGE", "setup state is too large; run Repair")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateError("STATE_INVALID", "setup state is unreadable; run Repair") from exc
    if not isinstance(parsed, dict) or _contains_absolute_or_secret(parsed):
        raise StateError("STATE_PRIVACY_INVALID", "setup state contains forbidden data; run Repair")
    return parsed


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise StateError("STATE_INVALID", "setup state structure is invalid; run Repair")
    return value


def _validate_manifest(code_root: Path) -> None:
    try:
        manifest = json.loads((code_root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateError("STATE_EXTENSION_DRIFT", "manifest identity cannot be verified; run Repair") from exc
    nodes = manifest.get("nodes") if isinstance(manifest, dict) else None
    valid = (
        isinstance(manifest, dict)
        and manifest.get("id") == EXTENSION_ID
        and manifest.get("version") == EXTENSION_VERSION
        and manifest.get("type") == "process"
        and manifest.get("entry") == ENTRY_FILENAME
        and isinstance(nodes, list)
        and len(nodes) == 1
        and isinstance(nodes[0], dict)
        and nodes[0].get("id") == NODE_ID
    )
    if not valid:
        raise StateError("STATE_EXTENSION_DRIFT", "manifest identity changed; run Repair")


def _validate_platform(value: object) -> PlatformFlavor:
    record = _mapping(value)
    if set(record) != {
        "system",
        "arch",
        "accelerator",
        "torchVariant",
        "expectedCuda",
        "pythonMinor",
        "setupCudaBf16",
        "attention",
    }:
        raise StateError("STATE_PLATFORM_INVALID", "platform state is invalid; run Repair")
    python_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    system = normalize_platform_name(sys.platform)
    arch = normalize_architecture(platform.machine())
    if tuple(sys.version_info[:2]) not in SUPPORTED_PYTHON_MINORS:
        raise StateError("STATE_PYTHON_UNSUPPORTED", "runtime Python must be 3.11 or 3.12; run Repair")
    if record.get("system") != system or record.get("arch") != arch:
        raise StateError("STATE_PLATFORM_DRIFT", "runtime platform changed; run Repair")
    if record.get("pythonMinor") != python_minor or record.get("attention") != "sdpa":
        raise StateError("STATE_PLATFORM_DRIFT", "runtime environment changed; run Repair")
    try:
        flavor = PlatformFlavor(
            system=str(record["system"]),
            arch=str(record["arch"]),
            accelerator=str(record["accelerator"]),
            torch_variant=str(record["torchVariant"]),
            expected_cuda=(None if record.get("expectedCuda") is None else str(record["expectedCuda"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StateError("STATE_PLATFORM_INVALID", "platform state is invalid; run Repair") from exc
    if flavor.accelerator == "cpu" and flavor.expected_cuda is None:
        expected_variant = "cpu"
    elif flavor.accelerator == "cuda" and flavor.expected_cuda == "12.8":
        expected_variant = "cu128"
    elif flavor.accelerator == "cuda" and flavor.expected_cuda == "13.0":
        expected_variant = "cu130"
    else:
        expected_variant = ""
    if (
        not expected_variant
        or flavor.torch_variant != expected_variant
        or flavor.system not in {"linux", "win32"}
        or flavor.arch not in {"x64", "arm64"}
        or (flavor.system == "win32" and flavor.arch != "x64")
    ):
        raise StateError("STATE_PLATFORM_INVALID", "platform state is invalid; run Repair")
    if not isinstance(record.get("setupCudaBf16"), bool):
        raise StateError("STATE_PLATFORM_INVALID", "platform state is invalid; run Repair")
    return flavor


def _validate_distributions(flavor: PlatformFlavor) -> None:
    try:
        actual = {
            name: importlib_metadata.version(name)
            for name in expected_distributions(flavor)
        }
    except importlib_metadata.PackageNotFoundError as exc:
        raise StateError(
            "STATE_DEPENDENCY_DRIFT", "a selected dependency pin is missing; run Repair"
        ) from exc
    if actual != expected_distributions(flavor):
        raise StateError(
            "STATE_DEPENDENCY_DRIFT", "selected dependency pin versions changed; run Repair"
        )


def validate_runtime_state(
    code_root: Path,
    models_root: Path,
) -> RuntimeState:
    """Re-derive the owned snapshot from the resolved models root and verify it exactly."""

    if code_root.name != EXTENSION_ID:
        raise StateError("STATE_EXTENSION_DRIFT", "extension directory identity changed; run Repair")
    _validate_manifest(code_root)
    state = _read_state(code_root)
    if set(state) != {"schema", "extension", "platform", "model"}:
        raise StateError("STATE_INVALID", "setup state structure changed; run Repair")
    if state.get("schema") != STATE_SCHEMA:
        raise StateError("STATE_SCHEMA_MISMATCH", "setup state schema changed; run Repair")
    if _mapping(state.get("extension")) != {
        "id": EXTENSION_ID,
        "version": EXTENSION_VERSION,
        "nodeId": NODE_ID,
        "entry": ENTRY_FILENAME,
    }:
        raise StateError("STATE_EXTENSION_DRIFT", "setup state identity changed; run Repair")
    if _mapping(state.get("model")) != {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "inventory": inventory_payload(),
    }:
        raise StateError("STATE_MODEL_DRIFT", "setup state model identity changed; run Repair")
    flavor = _validate_platform(state.get("platform"))
    _validate_distributions(flavor)
    try:
        model_dir = owned_model_directory(models_root, create=False)
    except PathContractError as exc:
        raise StateError("STATE_MODEL_PATH_INVALID", "resolved model storage is unsafe; run Repair") from exc
    if verify_snapshot(model_dir, require_ready=True):
        raise StateError("STATE_ASSET_INVENTORY_INVALID", "model files are missing or invalid; run Repair")
    return RuntimeState(model_dir=canonical(model_dir), flavor=flavor, payload=state)
