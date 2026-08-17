"""Explicit Modly storage paths and owned model-directory safety."""

from __future__ import annotations

from dataclasses import dataclass
import json
import ntpath
import os
from pathlib import Path, PurePath
import posixpath
import re
import stat
import sys
from typing import Iterable, Mapping, Sequence

from .constants import ENTRY_FILENAME, EXTENSION_ID, NODE_ID


WINDOWS_REPARSE_ATTRIBUTE = 0x400


class PathContractError(RuntimeError):
    """A stable-code failure at an explicit filesystem boundary."""

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


@dataclass(frozen=True)
class ExtensionBinding:
    extension_dir: Path
    bootstrap_python: Path


SETUP_MODELS_PAYLOAD_KEYS = ("models_dir", "modelsDir")
RUNTIME_MODELS_PAYLOAD_KEYS = ("modelsDir",)
MODELS_ENVIRONMENT_KEYS = ("MODLY_MODELS_DIR", "MODELS_DIR")


def normalize_platform_name(value: object) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized.startswith("linux"):
        return "linux"
    if normalized in {"win32", "windows"}:
        return "win32"
    return normalized


def normalize_architecture(value: object) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized in {"x64", "amd64", "x86_64"}:
        return "x64"
    if normalized in {"arm64", "aarch64"}:
        return "arm64"
    return normalized


def normalize_configured_directory_path(value: object, label: str, platform_name: object) -> str:
    """Mirror the host's fully-qualified Linux/Windows directory validation."""

    if not isinstance(value, str) or not value.strip():
        raise PathContractError("PATH_ABSOLUTE_REQUIRED", f"{label} must be a non-empty absolute path")
    if "\0" in value:
        raise PathContractError("PATH_NULL_BYTE", f"{label} must not contain null bytes")
    platform_value = normalize_platform_name(platform_name)
    if platform_value == "win32":
        fully_qualified = bool(
            re.match(r"^(?:[A-Za-z]:[\\/]|[\\/]{2}[^\\/]+[\\/][^\\/]+)", value)
        )
        if not fully_qualified:
            raise PathContractError(
                "PATH_ABSOLUTE_REQUIRED", f"{label} must be a fully-qualified Windows path"
            )
        return ntpath.normpath(value)
    if platform_value == "linux":
        if not posixpath.isabs(value):
            raise PathContractError("PATH_ABSOLUTE_REQUIRED", f"{label} must be an absolute Linux path")
        return posixpath.normpath(value)
    raise PathContractError("PATH_PLATFORM_UNSUPPORTED", "configured paths require Windows or Linux")


def current_platform_name() -> str:
    return normalize_platform_name(sys.platform)


def native_directory_path(
    value: object,
    label: str,
    platform_name: object,
    *,
    must_exist: bool,
    create: bool = False,
) -> Path:
    platform_value = normalize_platform_name(platform_name)
    normalized = normalize_configured_directory_path(value, label, platform_value)
    if platform_value != current_platform_name():
        raise PathContractError(
            "PATH_PLATFORM_MISMATCH", "configured path flavor does not match the running platform"
        )
    path = Path(normalized)
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PathContractError("PATH_CREATE_FAILED", f"{label} could not be created") from exc
    if must_exist and not path.is_dir():
        raise PathContractError("PATH_DIRECTORY_MISSING", f"{label} must identify an existing directory")
    return path


def verify_extension_identity(extension_dir: Path, code_root: Path) -> None:
    """Verify the manifest through a lexical install path without resolving that path away."""

    try:
        extension_root = extension_dir.resolve(strict=True)
        bound_code_root = code_root.resolve(strict=True)
        if not bound_code_root.is_dir() or extension_root != bound_code_root:
            raise PathContractError(
                "PATH_EXTENSION_BINDING_MISMATCH", "ext_dir does not identify this extension"
            )
    except OSError as exc:
        raise PathContractError(
            "PATH_EXTENSION_BINDING_MISMATCH", "the extension directory could not be verified"
        ) from exc

    manifest_path = extension_dir / "manifest.json"
    try:
        manifest_info = manifest_path.lstat()
    except OSError as exc:
        raise PathContractError(
            "PATH_MANIFEST_UNSAFE", "the extension manifest must be a regular local file"
        ) from exc
    if (
        stat.S_ISLNK(manifest_info.st_mode)
        or bool(getattr(manifest_info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE)
        or not stat.S_ISREG(manifest_info.st_mode)
    ):
        raise PathContractError(
            "PATH_MANIFEST_UNSAFE", "the extension manifest must be a regular local file"
        )
    try:
        if manifest_path.resolve(strict=True).parent != bound_code_root:
            raise PathContractError(
                "PATH_MANIFEST_UNSAFE", "the extension manifest is outside the bound extension root"
            )
        if manifest_info.st_size <= 0 or manifest_info.st_size > 1024 * 1024:
            raise PathContractError(
                "PATH_MANIFEST_INVALID", "the extension manifest has an invalid size"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except PathContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PathContractError(
            "PATH_MANIFEST_INVALID", "the extension manifest could not be verified"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("id") != EXTENSION_ID:
        raise PathContractError(
            "PATH_MANIFEST_ID_MISMATCH", "the extension manifest id does not match this extension"
        )
    nodes = manifest.get("nodes")
    if not (
        manifest.get("type") == "process"
        and manifest.get("entry") == ENTRY_FILENAME
        and isinstance(nodes, list)
        and len(nodes) == 1
        and isinstance(nodes[0], dict)
        and nodes[0].get("id") == NODE_ID
    ):
        raise PathContractError(
            "PATH_MANIFEST_CONTRACT_MISMATCH", "the extension manifest contract is invalid"
        )


def bind_extension(context: Mapping[str, object], code_root: Path) -> ExtensionBinding:
    platform_name = normalize_platform_name(context.get("platform") or current_platform_name())
    extension_dir = native_directory_path(
        context.get("ext_dir"), "ext_dir", platform_name, must_exist=True
    )
    verify_extension_identity(extension_dir, code_root)

    python_value = context.get("python_exe")
    if not isinstance(python_value, str) or not python_value.strip():
        raise PathContractError("PATH_PYTHON_REQUIRED", "python_exe is required")
    normalized_python = normalize_configured_directory_path(
        str(Path(python_value).parent), "python_exe parent", platform_name
    )
    python = Path(normalized_python) / Path(python_value).name
    if not python.is_file():
        raise PathContractError("PATH_PYTHON_MISSING", "python_exe must identify an existing file")
    return ExtensionBinding(extension_dir=extension_dir, bootstrap_python=python)


def _reject_traversal(value: object, label: str, platform_name: str) -> None:
    if not isinstance(value, str):
        return
    separators = r"[\\/]" if platform_name == "win32" else "/"
    if any(part in {".", ".."} for part in re.split(separators, value)):
        raise PathContractError(
            "PATH_TRAVERSAL_REJECTED", f"{label} must not contain traversal segments"
        )


def _configured_models_root(
    value: object,
    label: str,
    platform_name: str,
    *,
    require_existing: bool,
) -> Path:
    _reject_traversal(value, label, platform_name)
    return native_directory_path(
        value,
        label,
        platform_name,
        must_exist=require_existing,
    )


def _models_root_from_keys(
    values: Mapping[str, object],
    keys: Sequence[str],
    platform_name: str,
    *,
    require_existing: bool,
) -> Path | None:
    present = [key for key in keys if key in values]
    if not present:
        return None
    roots = [
        _configured_models_root(
            values[key],
            key,
            platform_name,
            require_existing=require_existing,
        )
        for key in present
    ]
    first_key = canonical_comparison_key(roots[0], platform_name)
    if any(canonical_comparison_key(root, platform_name) != first_key for root in roots[1:]):
        raise PathContractError(
            "PATH_MODELS_CONFLICT",
            "multiple model-directory overrides identify different locations",
        )
    return roots[0]


def resolve_models_root(
    payload: Mapping[str, object],
    extension_dir: Path,
    code_root: Path,
    platform_name: object,
    *,
    payload_keys: Sequence[str],
    environ: Mapping[str, str] | None = None,
    require_existing: bool = True,
) -> Path:
    """Resolve one existing Modly models root with explicit, env, then layout precedence."""

    platform_value = normalize_platform_name(platform_name)
    explicit = _models_root_from_keys(
        payload,
        payload_keys,
        platform_value,
        require_existing=require_existing,
    )
    if explicit is not None:
        return explicit

    environment: Mapping[str, object] = os.environ if environ is None else environ
    overridden = _models_root_from_keys(
        environment,
        MODELS_ENVIRONMENT_KEYS,
        platform_value,
        require_existing=require_existing,
    )
    if overridden is not None:
        return overridden

    verify_extension_identity(extension_dir, code_root)
    extensions_root = extension_dir.parent
    if extensions_root.name.casefold() != "extensions":
        raise PathContractError(
            "PATH_MODELS_LAYOUT_UNAVAILABLE",
            (
                "the Modly models directory cannot be derived from this installation; "
                "install under an extensions directory or set MODLY_MODELS_DIR/MODELS_DIR, "
                "then run Repair"
            ),
        )
    inferred = extensions_root.parent / "models"
    try:
        return native_directory_path(
            str(inferred),
            "conventional Modly models directory",
            platform_value,
            must_exist=True,
        )
    except PathContractError as exc:
        if exc.code not in {"PATH_DIRECTORY_MISSING", "PATH_ABSOLUTE_REQUIRED"}:
            raise
        raise PathContractError(
            "PATH_MODELS_LAYOUT_UNAVAILABLE",
            (
                "the conventional sibling models directory is unavailable; "
                "set MODLY_MODELS_DIR/MODELS_DIR for an independently configured root, "
                "then run Repair"
            ),
        ) from exc


def canonical(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError as exc:
        raise PathContractError("PATH_CANONICALIZE_FAILED", "a configured path could not be verified") from exc


def _lexical_comparison_key(value: os.PathLike[str] | str, platform_name: str) -> str:
    raw = os.fspath(value)
    if platform_name == "win32":
        normalized = ntpath.normcase(ntpath.normpath(raw))
        if not ntpath.isabs(normalized):
            raise PathContractError(
                "PATH_ABSOLUTE_REQUIRED", "storage boundaries require absolute Windows paths"
            )
        return normalized
    normalized = posixpath.normpath(raw)
    if not posixpath.isabs(normalized):
        raise PathContractError(
            "PATH_ABSOLUTE_REQUIRED", "storage boundaries require absolute Linux paths"
        )
    return normalized


def _prospective_native_key(path: Path, platform_name: str) -> str:
    """Resolve the deepest existing ancestor and append missing lexical segments."""

    lexical = Path(os.path.normpath(os.path.abspath(os.fspath(path))))
    missing: list[str] = []
    current = lexical
    while True:
        try:
            current.lstat()
            break
        except FileNotFoundError:
            if current.parent == current:
                raise PathContractError(
                    "PATH_CANONICALIZE_FAILED", "a storage boundary has no verifiable ancestor"
                )
            missing.append(current.name)
            current = current.parent
        except OSError as exc:
            raise PathContractError(
                "PATH_CANONICALIZE_FAILED", "a storage boundary could not be inspected"
            ) from exc
    try:
        resolved = current.resolve(strict=True).joinpath(*reversed(missing))
    except OSError as exc:
        raise PathContractError(
            "PATH_CANONICALIZE_FAILED", "a storage boundary alias could not be verified"
        ) from exc
    raw = os.path.normcase(os.path.normpath(str(resolved)))
    return ntpath.normcase(ntpath.normpath(raw)) if platform_name == "win32" else raw


def canonical_comparison_key(
    value: os.PathLike[str] | str,
    platform_name: object,
) -> str:
    """Return a comparison key for existing or prospective absolute storage paths."""

    platform_value = normalize_platform_name(platform_name)
    lexical = _lexical_comparison_key(value, platform_value)
    if platform_value == current_platform_name():
        return _prospective_native_key(Path(os.fspath(value)), platform_value)
    # Cross-platform static validation cannot inspect foreign filesystems, but
    # still applies that platform's absolute-path, separator, and case rules.
    return lexical


def _keys_overlap(first: str, second: str, platform_name: str) -> bool:
    path_module = ntpath if platform_name == "win32" else posixpath
    try:
        common = path_module.commonpath((first, second))
    except ValueError:
        return False
    return common == first or common == second


def storage_paths_overlap(
    first: os.PathLike[str] | str,
    second: os.PathLike[str] | str,
    platform_name: object,
) -> bool:
    """Detect same/ancestor/descendant overlap lexically and through existing aliases."""

    platform_value = normalize_platform_name(platform_name)
    if platform_value not in {"linux", "win32"}:
        raise PathContractError(
            "PATH_PLATFORM_UNSUPPORTED", "storage boundaries require Windows or Linux"
        )
    first_lexical = _lexical_comparison_key(first, platform_value)
    second_lexical = _lexical_comparison_key(second, platform_value)
    if _keys_overlap(first_lexical, second_lexical, platform_value):
        return True
    first_canonical = canonical_comparison_key(first, platform_value)
    second_canonical = canonical_comparison_key(second, platform_value)
    return _keys_overlap(first_canonical, second_canonical, platform_value)


def require_storage_disjoint(
    model_paths: Iterable[os.PathLike[str] | str],
    mutable_paths: Iterable[os.PathLike[str] | str],
    platform_name: object,
    *,
    code: str,
    public_message: str,
) -> None:
    """Fail with a sanitized stable code when any storage boundary overlaps."""

    models = tuple(model_paths)
    mutable = tuple(mutable_paths)
    for model_path in models:
        for mutable_path in mutable:
            if storage_paths_overlap(model_path, mutable_path, platform_name):
                raise PathContractError(code, public_message)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def _safe_relative_parts(parts: Iterable[str]) -> tuple[str, ...]:
    result = tuple(parts)
    if not result:
        raise PathContractError("PATH_RELATIVE_INVALID", "an owned relative path is empty")
    for part in result:
        parsed = PurePath(part)
        if part in {"", ".", ".."} or parsed.is_absolute() or len(parsed.parts) != 1:
            raise PathContractError("PATH_RELATIVE_INVALID", "an owned path segment is unsafe")
    return result


def ensure_directory_beneath(root: Path, parts: Iterable[str], *, create: bool) -> Path:
    """Permit an aliased root while rejecting every alias below it."""

    if not root.is_dir():
        raise PathContractError("PATH_MODELS_MISSING", "the configured models directory is unavailable")
    root_canonical = canonical(root)
    current = root
    for part in _safe_relative_parts(parts):
        current = current / part
        if current.exists() or current.is_symlink():
            if _is_link_or_reparse(current):
                raise PathContractError(
                    "PATH_OWNED_ALIAS", "an owned model path contains a filesystem alias"
                )
            if not current.is_dir():
                raise PathContractError(
                    "PATH_OWNED_NOT_DIRECTORY", "an owned model path is not a directory"
                )
        elif create:
            try:
                current.mkdir()
            except OSError as exc:
                raise PathContractError(
                    "PATH_OWNED_CREATE_FAILED", "an owned model directory could not be created"
                ) from exc
        if _is_link_or_reparse(current):
            raise PathContractError(
                "PATH_OWNED_ALIAS", "an owned model path contains a filesystem alias"
            )
        if not current.is_dir():
            raise PathContractError(
                "PATH_OWNED_NOT_DIRECTORY", "an owned model path is not a directory"
            )
        resolved = canonical(current)
        if root_canonical not in resolved.parents:
            raise PathContractError(
                "PATH_OWNED_ESCAPE", "an owned model path escapes the configured models directory"
            )
    return current


def owned_model_directory(models_root: Path, *, create: bool) -> Path:
    return ensure_directory_beneath(models_root, (EXTENSION_ID, NODE_ID), create=create)


def safe_asset_file(model_dir: Path, relative_path: str, *, create_parent: bool) -> Path:
    pure = PurePath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise PathContractError("PATH_RELATIVE_INVALID", "an asset path is unsafe")
    parent = model_dir
    if pure.parts[:-1]:
        parent = ensure_directory_beneath(model_dir, pure.parts[:-1], create=create_parent)
    candidate = parent / pure.name
    if candidate.exists() or candidate.is_symlink():
        if _is_link_or_reparse(candidate):
            raise PathContractError("PATH_ASSET_ALIAS", "an asset path is a filesystem alias")
        if not candidate.is_file():
            raise PathContractError("PATH_ASSET_NOT_FILE", "an asset path is not a regular file")
    model_canonical = canonical(model_dir)
    if model_canonical not in canonical(candidate).parents:
        raise PathContractError("PATH_ASSET_ESCAPE", "an asset path escapes its owned directory")
    return candidate
